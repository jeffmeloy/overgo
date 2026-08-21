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
	"fmt"
	"math"

	"overgo/internal/optimizer"
	"overgo/internal/safetensors"
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
	read := func(name string) ([]float32, []int, error) {
		tensor, ok := src.Tensors[name]
		if !ok {
			return nil, nil, fmt.Errorf("latentimage final layer: missing tensor %s", name)
		}
		values, err := safetensors.ReadF32(tensor)
		if err != nil {
			return nil, nil, err
		}
		shape := make([]int, len(tensor.Shape))
		for i, extent := range tensor.Shape {
			shape[i] = int(extent)
		}
		return values, shape, nil
	}
	linear, linearShape, err := read(b.Linear)
	if err != nil {
		return w, err
	}
	if len(linearShape) != 2 || linearShape[0] <= 0 || linearShape[1] <= 0 {
		return w, fmt.Errorf("latentimage final layer: %s shape %v", b.Linear, linearShape)
	}
	w.Out, w.Hidden, w.Linear = linearShape[0], linearShape[1], linear
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
	if len(w.Table) != 2*w.Hidden || len(w.Bias) != w.Out {
		return w, fmt.Errorf("latentimage final layer: table=%d bias=%d, want %d/%d",
			len(w.Table), len(w.Bias), 2*w.Hidden, w.Out)
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
func NewFinalLayerTrainer(w FinalLayerWeights, eps float64) (*FinalLayerTrainer, error) {
	if w.Hidden <= 0 || w.Out <= 0 || eps <= 0 {
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
		{"final.table", w.Table, 2, w.Hidden, &trainerShell.table},
		{"final.linear", w.Linear, w.Out, w.Hidden, &trainerShell.linear},
		{"final.bias", w.Bias, 1, w.Out, &trainerShell.bias},
	}
	if w.Norm != nil {
		sections = append([]section{{"final.norm", w.Norm, 1, w.Hidden, &trainerShell.norm}}, sections...)
	}
	total := 0
	specs := make([]optimizer.GroupSpec, len(sections))
	for i, s := range sections {
		if len(s.values) != s.rows*s.cols {
			return nil, fmt.Errorf("latentimage final layer: %s has %d values, want %d", s.name, len(s.values), s.rows*s.cols)
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
	trainer.config = optimizer.Config{
		BaseLearningRate: trainingprogram.BuiltinOptimizerPolicy().BaseLearningRate(total),
		Momentum:         trainingprogram.BuiltinOptimizerPolicy().Momentum(),
		Schedule:         optimizer.ScheduleConstant,
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
	rows, err := t.rows(hidden, temb)
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
	if err := t.stepper.Step(); err != nil {
		return FinalLayerStepResult{}, err
	}
	t.step++
	return FinalLayerStepResult{
		Loss: loss, Step: t.step,
		LearningRate: t.config.LearningRate(t.step), GradientL2: gradientL2,
	}, nil
}

// FinalLayerStepResult: measured facts of one observed organ training step.
type FinalLayerStepResult struct {
	Loss         float64
	Step         int
	LearningRate float64
	GradientL2   float64
}

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
	normed = make([]float64, rows*h)
	modulated = make([]float64, rows*h)
	for r := 0; r < rows; r++ {
		xr := hidden[r*h : (r+1)*h]
		if t.normed {
			var ss float64
			for i := 0; i < h; i++ {
				ss += xr[i] * xr[i]
			}
			inv := 1 / math.Sqrt(ss/float64(h)+t.eps)
			for i := 0; i < h; i++ {
				normed[r*h+i] = xr[i] * inv * (1 + float64(normW[i]))
			}
		} else {
			// Non-parametric LayerNorm (the Wan2.1 head).
			var mean float64
			for i := 0; i < h; i++ {
				mean += xr[i]
			}
			mean /= float64(h)
			var variance float64
			for i := 0; i < h; i++ {
				d := xr[i] - mean
				variance += d * d
			}
			inv := 1 / math.Sqrt(variance/float64(h)+t.eps)
			for i := 0; i < h; i++ {
				normed[r*h+i] = (xr[i] - mean) * inv
			}
		}
		for i := 0; i < h; i++ {
			scale := temb[i] + float64(table[i])
			shift := temb[i] + float64(table[h+i])
			modulated[r*h+i] = (1+scale)*normed[r*h+i] + shift
		}
	}
	out = make([]float64, rows*o)
	for r := 0; r < rows; r++ {
		mr := modulated[r*h : (r+1)*h]
		for c := 0; c < o; c++ {
			acc := float64(linB[c])
			wRow := linW[c*h : (c+1)*h]
			for i := 0; i < h; i++ {
				acc += mr[i] * float64(wRow[i])
			}
			out[r*o+c] = acc
		}
	}
	return normed, modulated, out
}

// lossAndGradients fills the packed gradient buffer without stepping. The
// organ is the stack's last layer, so no input gradient is produced.
func (t *FinalLayerTrainer) lossAndGradients(hidden, temb, target []float64) (float64, float64, error) {
	rows, err := t.rows(hidden, temb)
	if err != nil {
		return 0, 0, err
	}
	h, o := t.hidden, t.out
	if len(target) != rows*o {
		return 0, 0, fmt.Errorf("latentimage final layer: target len=%d, want %d", len(target), rows*o)
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
	for r := 0; r < rows; r++ {
		mr := modulated[r*h : (r+1)*h]
		dor := dOut[r*o : (r+1)*o]
		for c := 0; c < o; c++ {
			g := dor[c]
			gBias[c] += float32(g)
			wRow := linW[c*h : (c+1)*h]
			gRow := gLin[c*h : (c+1)*h]
			for i := 0; i < h; i++ {
				gRow[i] += float32(g * mr[i])
				dModulated[r*h+i] += g * float64(wRow[i])
			}
		}
	}
	// Affine backward: modulated = (1+scale)*n + shift with scale/shift =
	// temb + table rows (temb is data).
	dNormed := make([]float64, rows*h)
	for r := 0; r < rows; r++ {
		for i := 0; i < h; i++ {
			g := dModulated[r*h+i]
			scale := temb[i] + float64(table[i])
			dNormed[r*h+i] = g * (1 + scale)
			gTable[i] += float32(g * normed[r*h+i]) // dScale
			gTable[h+i] += float32(g)               // dShift
		}
	}
	// Zero-centered RMSNorm backward: n_i = x_i*inv*(1+w_i); only the weight
	// gradient is needed at the organ boundary. The non-parametric LayerNorm
	// head has no norm parameter, so nothing accumulates.
	if t.normed {
		gNorm := t.gradients[t.norm.start:t.norm.end]
		for r := 0; r < rows; r++ {
			xr := hidden[r*h : (r+1)*h]
			var ss float64
			for i := 0; i < h; i++ {
				ss += xr[i] * xr[i]
			}
			inv := 1 / math.Sqrt(ss/float64(h)+t.eps)
			for i := 0; i < h; i++ {
				gNorm[i] += float32(dNormed[r*h+i] * xr[i] * inv)
			}
		}
	}

	var gradientSquared float64
	for _, g := range t.gradients {
		gradientSquared += float64(g) * float64(g)
	}
	return loss, math.Sqrt(gradientSquared), nil
}

func (t *FinalLayerTrainer) rows(hidden, temb []float64) (int, error) {
	if len(temb) != t.hidden {
		return 0, fmt.Errorf("latentimage final layer: temb len=%d, want %d", len(temb), t.hidden)
	}
	if len(hidden) == 0 || len(hidden)%t.hidden != 0 {
		return 0, fmt.Errorf("latentimage final layer: hidden len=%d not divisible by %d", len(hidden), t.hidden)
	}
	return len(hidden) / t.hidden, nil
}
