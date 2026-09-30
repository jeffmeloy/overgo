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

	"overgo/internal/hostmath"
	"overgo/internal/optimizer"
	"overgo/internal/trainingprogram"
)

// FlowHeadStepResult: measured facts of one observed organ training step.
type FlowHeadStepResult = optimizer.ObservedStepResult

// FlowHeadTrainer: packed fm_head organ over the shared Muon stepper. The
// BF16 serving weights decode into the f32 master weights the pack owns --
// the explicit training-precision contract (the serving artifact stays
// BF16, the same master-weight move the reference gemma lane makes for its
// fp8 base).
type FlowHeadTrainer struct {
	plan    FlowPlan
	pack    *optimizer.TensorPack
	stepper optimizer.Stepper
	config  optimizer.Config
	step    int
}

// The fm_head groups in pack order.
const (
	flowW0 = iota
	flowB0
	flowW2
	flowB2
)

func (t *FlowHeadTrainer) view(group int) []float32     { return t.pack.Weights(group) }
func (t *FlowHeadTrainer) gradView(group int) []float32 { return t.pack.Gradients(group) }

// NewFlowHeadTrainer packs the head organ (W0, B0, W2, B2) from the loaded
// terminal weights and compiles the Muon plan. Base LR derives from the
// organ's parameter count; momentum from the CLT effective-samples rule.
func NewFlowHeadTrainer(plan FlowPlan, head FlowMLPWeights, policy trainingprogram.OptimizerPolicy) (*FlowHeadTrainer, error) {
	if head.W0.In != plan.Hidden || head.W0.Out != plan.Hidden || head.W2.In != plan.Hidden || head.W2.Out != plan.FlowDim {
		return nil, fmt.Errorf("routed lm flow train: head weights are not [%d->%d->%d]", plan.Hidden, plan.Hidden, plan.FlowDim)
	}
	if len(head.B0) != plan.Hidden || len(head.B2) != plan.FlowDim {
		return nil, fmt.Errorf("routed lm flow train: head biases are %d/%d, want %d/%d", len(head.B0), len(head.B2), plan.Hidden, plan.FlowDim)
	}
	pack, err := optimizer.NewSectionPack([]optimizer.TensorSection{
		{Name: "fm_head.0.weight", Rows: plan.Hidden, Cols: plan.Hidden, BF16: head.W0.Data},
		{Name: "fm_head.0.bias", Rows: 1, Cols: plan.Hidden, Values: head.B0},
		{Name: "fm_head.2.weight", Rows: plan.FlowDim, Cols: plan.Hidden, BF16: head.W2.Data},
		{Name: "fm_head.2.bias", Rows: 1, Cols: plan.FlowDim, Values: head.B2},
	})
	if err != nil {
		return nil, fmt.Errorf("routed lm flow train: %w", err)
	}
	trainer := &FlowHeadTrainer{plan: plan, pack: pack}
	if trainer.config, err = policy.Config(pack.ParameterCount()); err != nil {
		return nil, err
	}
	if trainer.stepper, err = pack.NewStepper(trainer.config); err != nil {
		return nil, err
	}
	return trainer, nil
}

// ParameterCount reports the packed organ parameter total.
func (t *FlowHeadTrainer) ParameterCount() int { return t.pack.ParameterCount() }

// Config exposes the derived hyperparameters for evidence.
func (t *FlowHeadTrainer) Config() optimizer.Config { return t.config }

// OptimizerPlan is the compiled Muon plan over the packed organ: one group
// per fm_head tensor.
func (t *FlowHeadTrainer) OptimizerPlan() optimizer.Plan { return t.pack.Plan() }

// Snapshot copies the optimizer state the next step continues from.
func (t *FlowHeadTrainer) Snapshot() optimizer.State { return t.stepper.Snapshot() }

// SaveWeights writes the trained organ at path, the weights Snapshot pairs
// with.
func (t *FlowHeadTrainer) SaveWeights(path string) error { return t.pack.Save(path) }

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
	hostmath.Linear(mid, hidden, t.view(flowW0), rows, h, h)
	for r := range rows {
		hostmath.AddBias(mid[r*h:(r+1)*h], t.view(flowB0))
	}
	activated := make([]float32, len(mid))
	for i, v := range mid {
		activated[i] = float32(hostmath.GELUErf(float64(v)))
	}
	out := make([]float32, rows*t.plan.FlowDim)
	hostmath.Linear(out, activated, t.view(flowW2), rows, h, t.plan.FlowDim)
	for r := range rows {
		hostmath.AddBias(out[r*t.plan.FlowDim:(r+1)*t.plan.FlowDim], t.view(flowB2))
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
	hostmath.Linear(mid, hidden, t.view(flowW0), rows, h, h)
	for r := range rows {
		hostmath.AddBias(mid[r*h:(r+1)*h], t.view(flowB0))
	}
	activated := make([]float32, len(mid))
	for i, v := range mid {
		activated[i] = float32(hostmath.GELUErf(float64(v)))
	}
	out := make([]float32, rows*fd)
	hostmath.Linear(out, activated, t.view(flowW2), rows, h, fd)
	for r := range rows {
		hostmath.AddBias(out[r*fd:(r+1)*fd], t.view(flowB2))
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

	t.pack.ClearGradients()
	dActivated := make([]float32, len(activated))
	hostmath.LinearBackward(dActivated, t.gradView(flowW2), t.gradView(flowB2), activated, t.view(flowW2), dOut, rows, h, fd, false)
	dMid := make([]float32, len(mid))
	hostmath.GELUErfBackward(dMid, mid, dActivated)
	dHidden := make([]float32, rows*h)
	hostmath.LinearBackward(dHidden, t.gradView(flowW0), t.gradView(flowB0), hidden, t.view(flowW0), dMid, rows, h, h, false)
	return loss, t.pack.GradientL2(), nil
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
	return rows, float32(max(1-timestep, t.plan.TEps)), nil
}
