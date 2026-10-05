package densecausal

import (
	"errors"
	"fmt"
	"time"

	"overgo/internal/checked"
	"overgo/internal/optimizer"
)

// TrainState: exact step-boundary optimizer state.
type TrainState = optimizer.State

// TrainObserver: completed step; error cancels at the update boundary.
type TrainObserver func(step int, loss float64, stepWall time.Duration) error

// MoERouterObserver receives the exact route used by one layer and update.
// It is observation-only: the delivered slices do not alias training state.
type MoERouterObserver func(step int, observation MoERouterObservation) error

// TrainOptions are what a host training run adds to its token batches.
type TrainOptions struct {
	// Trained holds each batch's flags for the tokens it learns (see
	// MaskedLossAndGrads); nil learns every token.
	Trained [][]bool
	// Average, when positive, is the decay of a moving average of the weights
	// (optimizer.Average) taken after each update; the run ends with the model
	// holding the average, its optimizer state the one training reached.
	Average float64
	// ResumeAverage continues an average an earlier run left (optimizer.RestoreAverage)
	// in place of starting one at Average; the weights it spans must be the model's.
	ResumeAverage *optimizer.Average
	// ObserveRouter, when set, receives each update's MoE routes.
	ObserveRouter MoERouterObserver
	// Checkpoint, when set with a positive CheckpointEvery, is called after every
	// that many updates, counted from the optimizer's first, but not after the last,
	// with the number of updates done, the optimizer's state and the average, nil
	// when the run keeps none; the model's tensors then hold the weights as the
	// update left them, not the average.
	CheckpointEvery int
	Checkpoint      func(done int, state TrainState, average *optimizer.Average) error
}

// Train executes ordered token batches with optional resume and observation.
func (m *Model) Train(batches [][]int, baseLR, mu float64, resume *TrainState, observe TrainObserver) ([]float64, TrainState, error) {
	return m.TrainWith(batches, baseLR, mu, resume, observe, TrainOptions{})
}

// TrainWith executes ordered token batches as Train does, under the options.
func (m *Model) TrainWith(batches [][]int, baseLR, mu float64, resume *TrainState, observe TrainObserver, options TrainOptions) ([]float64, TrainState, error) {
	if options.Trained != nil && len(options.Trained) != len(batches) {
		return nil, TrainState{}, fmt.Errorf("densecausal: trained flags for %d batches, %d given", len(batches), len(options.Trained))
	}
	return m.train(batches, baseLR, mu, resume, observe, options)
}

// TrainWithRouterObservations executes the host/reference trainer while
// exposing the already-computed MoE routes. No routing policy or numeric path
// changes when an observer is present.
func (m *Model) TrainWithRouterObservations(
	batches [][]int,
	baseLR, mu float64,
	resume *TrainState,
	observe TrainObserver,
	observeRouter MoERouterObserver,
) ([]float64, TrainState, error) {
	return m.train(batches, baseLR, mu, resume, observe, TrainOptions{ObserveRouter: observeRouter})
}

func (m *Model) train(batches [][]int, baseLR, mu float64, resume *optimizer.State, observe TrainObserver, options TrainOptions) ([]float64, optimizer.State, error) {
	if err := validateTokenBatches(batches); err != nil {
		return nil, optimizer.State{}, err
	}
	update, err := m.NewDenseUpdate(baseLR, mu, resume)
	if err != nil {
		return nil, optimizer.State{}, err
	}
	weights, _ := update.Pack.Slabs()
	average := options.ResumeAverage
	switch {
	case average != nil:
		if len(average.Values()) != len(weights) {
			return nil, optimizer.State{}, fmt.Errorf("densecausal: a resumed average spans %d weights, the model's are %d", len(average.Values()), len(weights))
		}
	case options.Average > 0:
		if average, err = optimizer.NewAverage(weights, options.Average); err != nil {
			return nil, optimizer.State{}, err
		}
	}

	trajectory := make([]float64, 0, len(batches))
	firstStep := 0
	if resume != nil {
		firstStep = resume.Step
	}
	for step, tokens := range batches {
		stepStarted := time.Now()
		var flags []bool
		if options.Trained != nil {
			flags = options.Trained[step]
		}
		loss, grads, err := m.trainingLossAndGrads(tokens, flags, firstStep+step, options.ObserveRouter)
		if err != nil {
			return nil, optimizer.State{}, err
		}
		trajectory = append(trajectory, loss)
		if _, err := update.step(grads); err != nil {
			return nil, optimizer.State{}, err
		}
		if average != nil {
			if err := average.Update(weights); err != nil {
				return nil, optimizer.State{}, err
			}
		}
		if observe != nil {
			if err := observe(step, loss, time.Since(stepStarted)); err != nil {
				return nil, optimizer.State{}, err
			}
		}
		if done := firstStep + step + 1; options.Checkpoint != nil && options.CheckpointEvery > 0 && done%options.CheckpointEvery == 0 && step+1 < len(batches) {
			if err := options.Checkpoint(done, update.Snapshot(), average); err != nil {
				return nil, optimizer.State{}, err
			}
		}
	}
	if average != nil {
		copy(weights, average.Values())
		update.Pack.Scatter()
	}
	return trajectory, update.Snapshot(), nil
}

// constantMuon: the dense trainers' Muon settings, the recipe-derived base
// rate and momentum under the constant schedule.
func constantMuon(baseLR, momentum float64) optimizer.Config {
	return optimizer.Config{BaseLearningRate: baseLR, Momentum: momentum, Schedule: optimizer.ScheduleConstant}
}

// DenseUpdate is the Muon update the dense trainers share: every model
// tensor packed in name order, stepped under constantMuon.
type DenseUpdate struct {
	Pack *optimizer.TensorPack
	*optimizer.Optimizer
}

// NewDenseUpdate packs the model at the recipe-derived rate and restores
// the optimizer state a resume carries.
func (m *Model) NewDenseUpdate(baseLR, momentum float64, resume *TrainState) (DenseUpdate, error) {
	if baseLR <= 0 {
		return DenseUpdate{}, errors.New("densecausal: positive recipe-derived learning rate required")
	}
	pack, err := m.TrainingPack()
	if err != nil {
		return DenseUpdate{}, err
	}
	update, err := pack.NewOptimizer(constantMuon(baseLR, momentum))
	if err == nil && resume != nil {
		err = update.Restore(*resume)
	}
	return DenseUpdate{Pack: pack, Optimizer: update}, err
}

// step gathers one update's gradients, takes the Muon step, and publishes
// the stepped weights to the model tensors.
func (u DenseUpdate) step(grads Grads) (optimizer.StepResult, error) {
	if err := u.Pack.GatherGradients(grads); err != nil {
		return optimizer.StepResult{}, err
	}
	result := u.Step()
	u.Pack.Scatter()
	return result, nil
}

func validateTokenBatches(batches [][]int) error {
	if len(batches) == 0 {
		return errors.New("densecausal: token batches absent")
	}
	for index, tokens := range batches {
		if len(tokens) < 2 {
			return fmt.Errorf("densecausal: batch %d needs at least two tokens", index)
		}
	}
	return nil
}

// TrainingPlan compiles dense parameter and Muon identity without storage.
func (m *Model) TrainingPlan() (optimizer.Plan, error) {
	if m.frozen != nil {
		return optimizer.Plan{}, errFrozen
	}
	return optimizer.TensorPlan(m.Weights, m.muonGeometry)
}

// TrainingPack packs every model tensor over the TrainingPlan; its Scatter
// publishes the packed weights back to the model tensors.
func (m *Model) TrainingPack() (*optimizer.TensorPack, error) {
	if m.frozen != nil {
		return nil, errFrozen
	}
	return optimizer.NewTensorPack(m.Weights, m.muonGeometry)
}

// muonGeometry: a model tensor's stored shape -> Muon geometry.
func (m *Model) muonGeometry(name string, length int) (int, int, error) {
	return tensorGeometry(m.Shapes[name], length)
}

// tensorGeometry: stored shape -> Muon matrix/vector geometry.
func tensorGeometry(shape []int, length int) (int, int, error) {
	switch len(shape) {
	case 0:
		if length != 1 {
			return 0, 0, fmt.Errorf("scalar shape does not match length %d", length)
		}
		return 1, 1, nil
	case 2:
		if shape[0] <= 0 || shape[1] <= 0 {
			return 0, 0, fmt.Errorf("shape %v has non-positive dimensions", shape)
		}
		elements, valid := checked.Mul64(uint64(shape[0]), uint64(shape[1]))
		if !valid || elements != uint64(length) {
			return 0, 0, fmt.Errorf("shape %v does not match length %d", shape, length)
		}
		return shape[0], shape[1], nil
	case 1:
		if shape[0] <= 0 || shape[0] != length {
			return 0, 0, fmt.Errorf("shape %v does not match length %d", shape, length)
		}
		return shape[0], 1, nil
	default:
		return 0, 0, fmt.Errorf("unsupported tensor rank %d (shape %v)", len(shape), shape)
	}
}
