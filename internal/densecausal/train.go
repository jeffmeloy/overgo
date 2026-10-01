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

// Train executes ordered token batches with optional resume and observation.
func (m *Model) Train(batches [][]int, baseLR, mu float64, resume *TrainState, observe TrainObserver) ([]float64, TrainState, error) {
	return m.train(batches, baseLR, mu, resume, observe, nil)
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
	if observeRouter == nil {
		return m.Train(batches, baseLR, mu, resume, observe)
	}
	return m.train(batches, baseLR, mu, resume, observe, observeRouter)
}

func (m *Model) train(batches [][]int, baseLR, mu float64, resume *optimizer.State, observe TrainObserver, observeRouter MoERouterObserver) ([]float64, optimizer.State, error) {
	if err := validateTokenBatches(batches); err != nil {
		return nil, optimizer.State{}, err
	}
	update, err := m.NewDenseUpdate(baseLR, mu, resume)
	if err != nil {
		return nil, optimizer.State{}, err
	}

	trajectory := make([]float64, 0, len(batches))
	firstStep := 0
	if resume != nil {
		firstStep = resume.Step
	}
	for step, tokens := range batches {
		stepStarted := time.Now()
		loss, grads, err := m.trainingLossAndGrads(tokens, firstStep+step, observeRouter)
		if err != nil {
			return nil, optimizer.State{}, err
		}
		trajectory = append(trajectory, loss)
		if _, err := update.step(grads); err != nil {
			return nil, optimizer.State{}, err
		}
		if observe != nil {
			if err := observe(step, loss, time.Since(stepStarted)); err != nil {
				return nil, optimizer.State{}, err
			}
		}
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
	return optimizer.TensorPlan(m.Weights, m.muonGeometry)
}

// TrainingPack packs every model tensor over the TrainingPlan; its Scatter
// publishes the packed weights back to the model tensors.
func (m *Model) TrainingPack() (*optimizer.TensorPack, error) {
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
