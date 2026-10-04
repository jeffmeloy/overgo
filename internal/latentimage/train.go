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
	norm, table, linear, bias int
	pack                      *optimizer.TensorPack
	stepper                   optimizer.Stepper
	config                    optimizer.Config
	step                      int
}

// NewFinalLayerTrainer packs the organ and compiles the derived Muon plan.
func NewFinalLayerTrainer(w FinalLayerWeights, eps float64, policy trainingprogram.OptimizerPolicy) (*FinalLayerTrainer, error) {
	if !checked.PositiveInts(w.Hidden, w.Out) || !checked.PositiveFinite64(eps) {
		return nil, fmt.Errorf("latentimage final layer: geometry %dx%d eps %g", w.Hidden, w.Out, eps)
	}
	trainer := &FinalLayerTrainer{hidden: w.Hidden, out: w.Out, eps: eps, normed: w.Norm != nil}
	groups := []*int{&trainer.table, &trainer.linear, &trainer.bias}
	sections := []optimizer.TensorSection{
		{Name: "final.table", Rows: tensor.PairedExtent, Cols: w.Hidden, Values: w.Table},
		{Name: "final.linear", Rows: w.Out, Cols: w.Hidden, Values: w.Linear},
		{Name: "final.bias", Rows: tensor.SingletonExtent, Cols: w.Out, Values: w.Bias},
	}
	if trainer.normed {
		groups = append([]*int{&trainer.norm}, groups...)
		sections = append([]optimizer.TensorSection{{Name: "final.norm", Rows: tensor.SingletonExtent, Cols: w.Hidden, Values: w.Norm}}, sections...)
	}
	for group, slot := range groups {
		*slot = group
	}
	var err error
	if trainer.pack, err = optimizer.NewSectionPack(sections); err != nil {
		return nil, fmt.Errorf("latentimage final layer: %w", err)
	}
	if trainer.config, err = policy.Config(trainer.pack.ParameterCount()); err != nil {
		return nil, err
	}
	if trainer.stepper, err = trainer.pack.NewStepper(trainer.config); err != nil {
		return nil, err
	}
	return trainer, nil
}

// ParameterCount reports the packed organ parameter total.
func (t *FinalLayerTrainer) ParameterCount() int { return t.pack.ParameterCount() }

// Config exposes the derived hyperparameters for evidence.
func (t *FinalLayerTrainer) Config() optimizer.Config { return t.config }

// OptimizerPlan is the compiled Muon plan over the packed organ: one group
// per head tensor.
func (t *FinalLayerTrainer) OptimizerPlan() optimizer.Plan { return t.pack.Plan() }

// Snapshot copies the optimizer state the next step continues from.
func (t *FinalLayerTrainer) Snapshot() optimizer.State { return t.stepper.Snapshot() }

// SaveWeights writes the trained organ at path, the weights Snapshot pairs
// with.
func (t *FinalLayerTrainer) SaveWeights(path string) error { return t.pack.Save(path) }

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
	table := t.pack.Weights(t.table)
	linW := t.pack.Weights(t.linear)
	linB := t.pack.Weights(t.bias)
	if t.normed {
		normed = append([]float64(nil), hidden...)
		hostmath.ZeroCenteredRMSNormF64InPlace(normed, t.pack.Weights(t.norm), rows, h, t.eps)
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

	t.pack.ClearGradients()
	dModulated := make([]float64, rows*h)
	hostmath.LinearFloat64Backward(dModulated, t.pack.Gradients(t.linear), t.pack.Gradients(t.bias), modulated, t.pack.Weights(t.linear), dOut, rows, h, o)
	// Affine backward: modulated = (1+scale)*n + shift with scale/shift =
	// temb + table rows (temb is data).
	dNormed := make([]float64, rows*h)
	hostmath.AdaptiveAffineBackwardF64(dNormed, t.pack.Gradients(t.table), normed, temb, t.pack.Weights(t.table), dModulated, rows, h)
	// Zero-centered RMSNorm backward: n_i = x_i*inv*(1+w_i); only the weight
	// gradient is needed at the organ boundary. The non-parametric LayerNorm
	// head has no norm parameter, so nothing accumulates.
	if t.normed {
		hostmath.ZeroCenteredRMSNormWeightGradientF64(t.pack.Gradients(t.norm), hidden, dNormed, rows, h, t.eps)
	}
	return loss, t.pack.GradientL2(), nil
}
