// Trainable final-layer organ for the latent-flow denoiser (the Krea-2
// family MMDiT output head): zero-centered RMSNorm -> AdaLN affine (scale
// and shift each being time-embedding plus a learned table row) -> linear
// to patch velocity. The organ trains in f64 through the same math the host
// reference computes, over the shared Muon stepper with fully derived
// hyperparameters; gradients stop at the organ input (the head is the
// stack's last layer), and full-denoiser training remains the promotion
// gate beyond it.

package latentimage

import (
	"errors"
	"fmt"
	"math"

	"overgo/internal/checked"
	"overgo/internal/hostmath"
	"overgo/internal/optimizer"
	"overgo/internal/safetensors"
	"overgo/internal/tensor"
	"overgo/internal/trainingprogram"
)

// FinalLayerBinding: the head organ's tensor names in the artifact.
type FinalLayerBinding struct {
	Norm, Table, Linear, Bias string
}

// KreaFinalLayerBinding: the Krea-2-Turbo checkpoint naming.
func KreaFinalLayerBinding() FinalLayerBinding {
	return FinalLayerBinding{
		Norm: "last.norm.scale", Table: "last.modulation.lin",
		Linear: "last.linear.weight", Bias: "last.linear.bias",
	}
}

// WanHeadBinding: the Wan2.1 checkpoint naming. The head norm is a
// non-parametric LayerNorm, so no norm tensor is bound.
func WanHeadBinding() FinalLayerBinding {
	return FinalLayerBinding{
		Table: "head.modulation", Linear: "head.head.weight", Bias: "head.head.bias",
	}
}

// FinalLayerWeights: the loaded organ with derived geometry. A nil Norm
// declares a non-parametric LayerNorm head (the Wan2.1 family); a present
// Norm declares the zero-centered RMSNorm head (the Krea-2 family).
type FinalLayerWeights struct {
	Hidden, Out               int
	Norm, Table, Linear, Bias []float32
}

// LoadFinalLayerWeights derives the head geometry from the artifact's own
// tensor shapes and reads exactly the four organ tensors.
func LoadFinalLayerWeights(src *safetensors.Source, b FinalLayerBinding) (FinalLayerWeights, error) {
	var w FinalLayerWeights
	read := func(name string) ([]float32, safetensors.Tensor, error) {
		tensor, ok := src.Tensors[name]
		if !ok {
			return nil, safetensors.Tensor{}, fmt.Errorf("latentimage final layer: missing tensor %s", name)
		}
		values, err := safetensors.ReadF32(tensor)
		if err != nil {
			return nil, safetensors.Tensor{}, err
		}
		return values, tensor, nil
	}
	linear, linearTensor, err := read(b.Linear)
	if err != nil {
		return w, err
	}
	outputExtent, hiddenExtent, err := safetensors.MatrixShape(linearTensor)
	if err != nil {
		return w, fmt.Errorf("latentimage final layer: %s: %w", b.Linear, err)
	}
	var ok bool
	w.Out, ok = checked.Int(outputExtent)
	if !ok {
		return w, errors.New("latentimage final layer: output extent exceeds host range")
	}
	w.Hidden, ok = checked.Int(hiddenExtent)
	if !ok {
		return w, errors.New("latentimage final layer: hidden extent exceeds host range")
	}
	w.Linear = linear
	if b.Norm != "" {
		if w.Norm, _, err = read(b.Norm); err != nil {
			return w, err
		}
		if len(w.Norm) != w.Hidden {
			return w, fmt.Errorf("latentimage final layer: norm=%d, want %d", len(w.Norm), w.Hidden)
		}
	}
	if w.Table, _, err = read(b.Table); err != nil {
		return w, err
	}
	if w.Bias, _, err = read(b.Bias); err != nil {
		return w, err
	}
	tableElements, ok := checked.MulInt(tensor.PairedExtent, w.Hidden)
	if !ok {
		return w, fmt.Errorf("latentimage final layer: table extent overflows")
	}
	if !checked.Equal(len(w.Table), tableElements) || !checked.Equal(len(w.Bias), w.Out) {
		return w, fmt.Errorf("latentimage final layer: table=%d bias=%d, want %d/%d",
			len(w.Table), len(w.Bias), tableElements, w.Out)
	}
	return w, nil
}

// FinalLayerTrainer: packed head organ over the shared Muon stepper.
type FinalLayerTrainer struct {
	hidden, out               int
	eps                       float64
	normed                    bool
	weights                   []float32
	gradients                 []float32
	norm, table, linear, bias struct{ start, end int }
	stepper                   optimizer.Stepper
	config                    optimizer.Config
	step                      int
}

// NewFinalLayerTrainer packs the organ and compiles the derived Muon plan.
func NewFinalLayerTrainer(w FinalLayerWeights, eps float64, policy trainingprogram.OptimizerPolicy) (*FinalLayerTrainer, error) {
	if !checked.PositiveInts(w.Hidden, w.Out) || !checked.PositiveFinite64(eps) {
		return nil, fmt.Errorf("latentimage final layer: geometry %dx%d eps %g", w.Hidden, w.Out, eps)
	}
	type section struct {
		name       string
		values     []float32
		rows, cols int
		slot       *struct{ start, end int }
	}
	trainerShell := &FinalLayerTrainer{hidden: w.Hidden, out: w.Out, eps: eps, normed: w.Norm != nil}
	sections := []section{
		{"final.table", w.Table, tensor.PairedExtent, w.Hidden, &trainerShell.table},
		{"final.linear", w.Linear, w.Out, w.Hidden, &trainerShell.linear},
		{"final.bias", w.Bias, tensor.SingletonExtent, w.Out, &trainerShell.bias},
	}
	if w.Norm != nil {
		sections = append([]section{{"final.norm", w.Norm, tensor.SingletonExtent, w.Hidden, &trainerShell.norm}}, sections...)
	}
	total := tensor.FirstOffset
	specs := make([]optimizer.GroupSpec, len(sections))
	for i, s := range sections {
		if err := checked.Length(s.values, s.rows, s.cols); err != nil {
			return nil, fmt.Errorf("latentimage final layer: %s: %w", s.name, err)
		}
		specs[i] = optimizer.GroupSpec{Name: s.name, Start: total, End: total + len(s.values), Rows: s.rows, Cols: s.cols}
		total += len(s.values)
	}
	plan, err := optimizer.CompilePlan(total, specs)
	if err != nil {
		return nil, err
	}
	trainer := trainerShell
	trainer.weights = make([]float32, total)
	trainer.gradients = make([]float32, total)
	trainer.config, err = policy.Config(total)
	if err != nil {
		return nil, err
	}
	for i, s := range sections {
		s.slot.start, s.slot.end = specs[i].Start, specs[i].End
		copy(trainer.weights[specs[i].Start:specs[i].End], s.values)
	}
	trainer.stepper, err = optimizer.NewStepper(trainer.weights, trainer.gradients, plan, trainer.config)
	if err != nil {
		return nil, err
	}
	return trainer, nil
}

// ParameterCount reports the packed organ parameter total.
func (t *FinalLayerTrainer) ParameterCount() int { return len(t.weights) }

// Config exposes the derived hyperparameters for evidence.
func (t *FinalLayerTrainer) Config() optimizer.Config { return t.config }

// Close releases the stepper backend.
func (t *FinalLayerTrainer) Close() error { return t.stepper.Close() }

// Velocity runs the organ forward on the packed weights: zero-centered
// RMSNorm, AdaLN affine from temb plus the learned table, linear head.
func (t *FinalLayerTrainer) Velocity(hidden, temb []float64) ([]float64, error) {
	if err := checked.Length(temb, t.hidden); err != nil {
		return nil, fmt.Errorf("latentimage final layer: temb: %w", err)
	}
	rows, err := checked.Rows(hidden, t.hidden)
	if err != nil {
		return nil, err
	}
	_, _, out := t.forwardTrace(hidden, temb, rows)
	return out, nil
}

// Loss evaluates the velocity MSE against the target without training.
func (t *FinalLayerTrainer) Loss(hidden, temb, target []float64) (float64, error) {
	out, err := t.Velocity(hidden, temb)
	if err != nil {
		return 0, err
	}
	if len(target) != len(out) {
		return 0, fmt.Errorf("latentimage final layer: target len=%d, want %d", len(target), len(out))
	}
	invN := 1 / float64(len(target))
	var loss float64
	for i := range out {
		d := out[i] - target[i]
		loss += d * d * invN
	}
	return loss, nil
}

// Step: one observed training step of the velocity MSE objective.
func (t *FinalLayerTrainer) Step(hidden, temb, target []float64) (FinalLayerStepResult, error) {
	loss, gradientL2, err := t.lossAndGradients(hidden, temb, target)
	if err != nil {
		return FinalLayerStepResult{}, err
	}
	return optimizer.Advance(t.stepper, t.config, &t.step, loss, gradientL2)
}

// FinalLayerStepResult: measured facts of one observed organ training step.
type FinalLayerStepResult = optimizer.ObservedStepResult

// forwardTrace: normalized rows (pre-affine), affine rows, head output.
func (t *FinalLayerTrainer) forwardTrace(hidden, temb []float64, rows int) (normed, modulated, out []float64) {
	h, o := t.hidden, t.out
	var normW []float32
	if t.normed {
		normW = t.weights[t.norm.start:t.norm.end]
	}
	table := t.weights[t.table.start:t.table.end]
	linW := t.weights[t.linear.start:t.linear.end]
	linB := t.weights[t.bias.start:t.bias.end]
	if t.normed {
		normed = append([]float64(nil), hidden...)
		hostmath.ZeroCenteredRMSNormF64InPlace(normed, normW, rows, h, t.eps)
	} else {
		normed = hostmath.LayerNormF64(hidden, rows, h, t.eps)
	}
	modulated = hostmath.AdaptiveAffineF64(normed, temb, table, rows, h)
	out = hostmath.LinearFloat64(modulated, linW, linB, rows, h, o)
	return normed, modulated, out
}

// lossAndGradients fills the packed gradient buffer without stepping. The
// organ is the stack's last layer, so no input gradient is produced.
func (t *FinalLayerTrainer) lossAndGradients(hidden, temb, target []float64) (float64, float64, error) {
	if err := checked.Length(temb, t.hidden); err != nil {
		return 0, 0, fmt.Errorf("latentimage final layer: temb: %w", err)
	}
	rows, err := checked.Rows(hidden, t.hidden)
	if err != nil {
		return 0, 0, err
	}
	h, o := t.hidden, t.out
	if err := checked.Length(target, rows, o); err != nil {
		return 0, 0, fmt.Errorf("latentimage final layer: target: %w", err)
	}
	normed, modulated, out := t.forwardTrace(hidden, temb, rows)

	invN := 1 / float64(len(target))
	var loss float64
	dOut := make([]float64, len(out))
	for i := range out {
		d := out[i] - target[i]
		loss += d * d * invN
		dOut[i] = 2 * d * invN
	}

	clear(t.gradients)
	table := t.weights[t.table.start:t.table.end]
	linW := t.weights[t.linear.start:t.linear.end]
	gTable := t.gradients[t.table.start:t.table.end]
	gLin := t.gradients[t.linear.start:t.linear.end]
	gBias := t.gradients[t.bias.start:t.bias.end]

	dModulated := make([]float64, rows*h)
	hostmath.LinearFloat64Backward(dModulated, gLin, gBias, modulated, linW, dOut, rows, h, o)
	// Affine backward: modulated = (1+scale)*n + shift with scale/shift =
	// temb + table rows (temb is data).
	dNormed := make([]float64, rows*h)
	hostmath.AdaptiveAffineBackwardF64(dNormed, gTable, normed, temb, table, dModulated, rows, h)
	// Zero-centered RMSNorm backward: n_i = x_i*inv*(1+w_i); only the weight
	// gradient is needed at the organ boundary. The non-parametric LayerNorm
	// head has no norm parameter, so nothing accumulates.
	if t.normed {
		gNorm := t.gradients[t.norm.start:t.norm.end]
		hostmath.ZeroCenteredRMSNormWeightGradientF64(gNorm, hidden, dNormed, rows, h, t.eps)
	}

	var gradientSquared float64
	for _, g := range t.gradients {
		gradientSquared += float64(g) * float64(g)
	}
	return loss, math.Sqrt(gradientSquared), nil
}
