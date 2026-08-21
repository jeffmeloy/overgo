// Trainable flow-head organ for the routed-LM flow terminal: the fm_head
// MLP (linear -> exact GELU -> linear) trains in f32 through the same math
// the serving forward compiles, over the shared Muon stepper with fully
// derived hyperparameters. The velocity transform v = (x - z)/max(1-t,eps)
// is data-only, so its gradient is the 1/denom scale. Serving applies BF16
// rounding between stages; the trainer's f32 path is documented as the
// training-precision contract, and full-pipeline MoT training remains the
// promotion gate beyond this organ.
package routedlm

import (
	"fmt"
	"math"

	"overgo/internal/hostmath"
	"overgo/internal/optimizer"
	"overgo/internal/tensor/dtype"
	"overgo/internal/trainingprogram"
)

// FlowHeadStepResult: measured facts of one observed organ training step.
type FlowHeadStepResult = optimizer.ObservedStepResult

// bf16Masters decodes BF16 serving storage into f32 master weights -- the
// explicit training-precision contract (the serving artifact stays BF16; the
// trainer owns f32 masters initialized from it, the same master-weight move
// the reference gemma lane makes for its fp8 base).
func bf16Masters(data []uint16) []float32 {
	masters := make([]float32, len(data))
	for i, v := range data {
		masters[i] = dtype.BF16ToFloat32(v)
	}
	return masters
}

// FlowHeadTrainer: packed fm_head organ over the shared Muon stepper.
type FlowHeadTrainer struct {
	plan      FlowPlan
	weights   []float32
	gradients []float32
	w0, b0    span
	w2, b2    span
	stepper   optimizer.Stepper
	config    optimizer.Config
	step      int
}

type span struct{ start, end int }

func (t *FlowHeadTrainer) view(s span) []float32     { return t.weights[s.start:s.end] }
func (t *FlowHeadTrainer) gradView(s span) []float32 { return t.gradients[s.start:s.end] }

// NewFlowHeadTrainer packs the head organ (W0, B0, W2, B2) from the loaded
// terminal weights and compiles the Muon plan. Base LR derives from the
// organ's parameter count; momentum from the CLT effective-samples rule.
func NewFlowHeadTrainer(plan FlowPlan, head FlowMLPWeights) (*FlowHeadTrainer, error) {
	if head.W0.In != plan.Hidden || head.W0.Out != plan.Hidden || head.W2.In != plan.Hidden || head.W2.Out != plan.FlowDim {
		return nil, fmt.Errorf("routed lm flow train: head weights are not [%d->%d->%d]", plan.Hidden, plan.Hidden, plan.FlowDim)
	}
	if len(head.B0) != plan.Hidden || len(head.B2) != plan.FlowDim {
		return nil, fmt.Errorf("routed lm flow train: head biases are %d/%d, want %d/%d", len(head.B0), len(head.B2), plan.Hidden, plan.FlowDim)
	}
	sections := []struct {
		name       string
		values     []float32
		rows, cols int
	}{
		{"fm_head.0.weight", bf16Masters(head.W0.Data), plan.Hidden, plan.Hidden},
		{"fm_head.0.bias", head.B0, 1, plan.Hidden},
		{"fm_head.2.weight", bf16Masters(head.W2.Data), plan.FlowDim, plan.Hidden},
		{"fm_head.2.bias", head.B2, 1, plan.FlowDim},
	}
	total := 0
	specs := make([]optimizer.GroupSpec, len(sections))
	for i, s := range sections {
		if len(s.values) != s.rows*s.cols {
			return nil, fmt.Errorf("routed lm flow train: %s has %d values, want %d", s.name, len(s.values), s.rows*s.cols)
		}
		specs[i] = optimizer.GroupSpec{Name: s.name, Start: total, End: total + len(s.values), Rows: s.rows, Cols: s.cols}
		total += len(s.values)
	}
	compiled, err := optimizer.CompilePlan(total, specs)
	if err != nil {
		return nil, err
	}
	trainer := &FlowHeadTrainer{
		plan:      plan,
		weights:   make([]float32, total),
		gradients: make([]float32, total),
		config: optimizer.Config{
			BaseLearningRate: trainingprogram.BuiltinOptimizerPolicy().BaseLearningRate(total),
			Momentum:         trainingprogram.BuiltinOptimizerPolicy().Momentum(),
			Schedule:         optimizer.ScheduleConstant,
		},
	}
	spans := []*span{&trainer.w0, &trainer.b0, &trainer.w2, &trainer.b2}
	for i, s := range sections {
		*spans[i] = span{specs[i].Start, specs[i].End}
		copy(trainer.weights[specs[i].Start:specs[i].End], s.values)
	}
	trainer.stepper, err = optimizer.NewStepper(trainer.weights, trainer.gradients, compiled, trainer.config)
	if err != nil {
		return nil, err
	}
	return trainer, nil
}

// ParameterCount reports the packed organ parameter total.
func (t *FlowHeadTrainer) ParameterCount() int { return len(t.weights) }

// Config exposes the derived hyperparameters for evidence.
func (t *FlowHeadTrainer) Config() optimizer.Config { return t.config }

// Close releases the stepper backend.
func (t *FlowHeadTrainer) Close() error { return t.stepper.Close() }

// Velocity runs the training-precision (f32) head forward on the packed
// weights: linear0 -> exact GELU -> linear2 -> (x - z)/max(1-t, eps).
func (t *FlowHeadTrainer) Velocity(hidden, z []float32, timestep float64) ([]float32, error) {
	rows, denom, err := t.geometry(hidden, z, timestep)
	if err != nil {
		return nil, err
	}
	h := t.plan.Hidden
	mid := make([]float32, rows*h)
	hostmath.Linear(mid, hidden, t.view(t.w0), rows, h, h)
	for r := 0; r < rows; r++ {
		hostmath.AddBias(mid[r*h:(r+1)*h], t.view(t.b0))
	}
	activated := make([]float32, len(mid))
	for i, v := range mid {
		activated[i] = float32(hostmath.GELUErf(float64(v)))
	}
	out := make([]float32, rows*t.plan.FlowDim)
	hostmath.Linear(out, activated, t.view(t.w2), rows, h, t.plan.FlowDim)
	for r := 0; r < rows; r++ {
		hostmath.AddBias(out[r*t.plan.FlowDim:(r+1)*t.plan.FlowDim], t.view(t.b2))
	}
	for i := range out {
		out[i] = (out[i] - z[i]) / denom
	}
	return out, nil
}

// Step: one observed training step — MSE between the head's velocity and
// the target velocity, full organ VJP, Muon update.
func (t *FlowHeadTrainer) Step(hidden, z, target []float32, timestep float64) (FlowHeadStepResult, error) {
	loss, gradientL2, err := t.lossAndGradients(hidden, z, target, timestep)
	if err != nil {
		return FlowHeadStepResult{}, err
	}
	return optimizer.Advance(t.stepper, t.config, &t.step, loss, gradientL2)
}

// lossAndGradients fills the packed gradient buffer for one pair and returns
// the loss and gradient norm without stepping.
func (t *FlowHeadTrainer) lossAndGradients(hidden, z, target []float32, timestep float64) (float64, float64, error) {
	rows, denom, err := t.geometry(hidden, z, timestep)
	if err != nil {
		return 0, 0, err
	}
	if len(target) != len(z) {
		return 0, 0, fmt.Errorf("routed lm flow train: target len=%d, want %d", len(target), len(z))
	}
	h, fd := t.plan.Hidden, t.plan.FlowDim

	// Recomputed forward with retained pre-activation.
	mid := make([]float32, rows*h)
	hostmath.Linear(mid, hidden, t.view(t.w0), rows, h, h)
	for r := 0; r < rows; r++ {
		hostmath.AddBias(mid[r*h:(r+1)*h], t.view(t.b0))
	}
	activated := make([]float32, len(mid))
	for i, v := range mid {
		activated[i] = float32(hostmath.GELUErf(float64(v)))
	}
	out := make([]float32, rows*fd)
	hostmath.Linear(out, activated, t.view(t.w2), rows, h, fd)
	for r := 0; r < rows; r++ {
		hostmath.AddBias(out[r*fd:(r+1)*fd], t.view(t.b2))
	}

	// Loss over the velocity; the velocity transform contributes 1/denom.
	invN := 1 / float64(len(target))
	var loss float64
	dOut := make([]float32, len(out))
	for i := range out {
		v := (out[i]-z[i])/denom - target[i]
		loss += float64(v) * float64(v) * invN
		dOut[i] = float32(2 * float64(v) * invN / float64(denom))
	}

	clear(t.gradients)
	dActivated := make([]float32, len(activated))
	hostmath.LinearBackward(dActivated, t.gradView(t.w2), t.gradView(t.b2), activated, t.view(t.w2), dOut, rows, h, fd, false)
	dMid := make([]float32, len(mid))
	hostmath.GELUErfBackward(dMid, mid, dActivated)
	dHidden := make([]float32, rows*h)
	hostmath.LinearBackward(dHidden, t.gradView(t.w0), t.gradView(t.b0), hidden, t.view(t.w0), dMid, rows, h, h, false)

	var gradientSquared float64
	for _, g := range t.gradients {
		gradientSquared += float64(g) * float64(g)
	}
	return loss, math.Sqrt(gradientSquared), nil
}

func (t *FlowHeadTrainer) geometry(hidden, z []float32, timestep float64) (rows int, denom float32, err error) {
	if timestep < 0 || timestep >= 1 {
		return 0, 0, fmt.Errorf("routed lm flow train: timestep %g outside [0,1)", timestep)
	}
	if len(hidden) == 0 || len(hidden)%t.plan.Hidden != 0 {
		return 0, 0, fmt.Errorf("routed lm flow train: hidden len=%d not divisible by %d", len(hidden), t.plan.Hidden)
	}
	rows = len(hidden) / t.plan.Hidden
	if len(z) != rows*t.plan.FlowDim {
		return 0, 0, fmt.Errorf("routed lm flow train: z len=%d, want %d", len(z), rows*t.plan.FlowDim)
	}
	return rows, float32(math.Max(1-timestep, t.plan.TEps)), nil
}
