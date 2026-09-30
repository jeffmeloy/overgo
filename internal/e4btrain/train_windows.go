package e4btrain

import (
	"errors"

	"overgo/internal/cuda/device"
	"overgo/internal/devicegrad"
	"overgo/internal/optimizer"
	"overgo/internal/tensor/dtype"
)

// Trainer takes full-parameter steps of a Model on one device worker: every
// layer's windowed attention VJP runs on the device, and the Muon step
// streams one parameter group at a time across it, so device memory stays
// bounded by the largest tensor however large the model is. The rest of the
// forward and backward runs on the host masters.
type Trainer struct {
	*Model
	worker   *device.Worker
	config   optimizer.Config
	momentum []float32
	step     int
}

// NewTrainer binds model to worker under config, the Muon hyperparameters
// the training program's optimizer policy derives for the model's parameter
// count.
func NewTrainer(worker *device.Worker, model *Model, config optimizer.Config) (*Trainer, error) {
	if worker == nil || model == nil {
		return nil, errors.New("e4b trainer: a device worker and a model are required")
	}
	if err := config.Validate(); err != nil {
		return nil, err
	}
	return &Trainer{Model: model, worker: worker, config: config, momentum: make([]float32, len(model.weights))}, nil
}

// Step takes one full-parameter update on the window and reports the loss
// and gradient norm it stepped from.
func (t *Trainer) Step() (optimizer.ObservedStepResult, error) {
	if t.gradients == nil {
		return optimizer.ObservedStepResult{}, errors.New("e4b trainer: the gradient slab was released; the run takes no more steps")
	}
	loss, err := t.differentiate(t.attention)
	if err != nil {
		return optimizer.ObservedStepResult{}, err
	}
	norm := t.gradientL2()
	if err := optimizer.DeviceMuonStepPlanStreamed(t.worker, t.weights, t.gradients, t.momentum, t.plan, t.step+1, t.config); err != nil {
		return optimizer.ObservedStepResult{}, err
	}
	t.step++
	return optimizer.ObservedStepResult{Loss: loss, Step: t.step, LearningRate: t.config.LearningRate(t.step), GradientL2: norm}, nil
}

// attention is the device's windowed attention VJP.
func (t *Trainer) attention(dq, dk, dv, q, k, v, dOut []float32, seq, heads, kvHeads, headDim, window int) error {
	gq, gk, gv, err := devicegrad.WindowedCausalAttentionBackwardDevice(t.worker, q, k, v, dOut, seq, heads, kvHeads, headDim, window)
	if err != nil {
		return err
	}
	copy(dq, gq)
	copy(dk, gk)
	copy(dv, gv)
	return nil
}

// ReleaseGradients frees the gradient slab once the run takes no more
// steps, so the widened momentum a checkpoint records takes its place in
// host memory rather than joining it; a Step after it refuses.
func (t *Trainer) ReleaseGradients() {
	t.gradients = nil
	clear(t.layerGradients)
	t.globalGradients = globalParameters{}
}

// Snapshot is the optimizer state the next Step continues from, its f32
// momentum widened exactly.
func (t *Trainer) Snapshot() optimizer.State {
	return optimizer.State{
		PlanIdentity: t.plan.Identity(), Config: t.config, Step: t.step,
		Momentum: dtype.ConvertSlice[float64](t.momentum),
	}
}
