package densecausal

import (
	"errors"
	"fmt"
	"maps"
	"slices"

	"overgo/internal/checked"
	"overgo/internal/optimizer"
)

// TrainBatches applies one Muon update per ordered token batch.
func (m *Model) TrainBatches(batches [][]int, baseLR, mu float64) ([]float64, error) {
	trajectory, _, err := m.train(batches, baseLR, mu, nil)
	return trajectory, err
}

// TrainState: exact step-boundary optimizer state.
type TrainState = optimizer.State

// TrainBatchesResume resumes an ordered sequence at an update boundary.
func (m *Model) TrainBatchesResume(batches [][]int, baseLR, mu float64, resume *TrainState) ([]float64, TrainState, error) {
	return m.train(batches, baseLR, mu, resume)
}

func (m *Model) train(batches [][]int, baseLR, mu float64, resume *optimizer.State) ([]float64, optimizer.State, error) {
	if err := validateTokenBatches(batches); err != nil {
		return nil, optimizer.State{}, err
	}
	names, weights, gradients, plan, resolvedLR, err := m.trainSetup(baseLR)
	if err != nil {
		return nil, optimizer.State{}, err
	}
	opt, err := optimizer.New(weights, gradients, plan, optimizer.Config{
		BaseLearningRate: resolvedLR, Momentum: mu, Schedule: optimizer.ScheduleConstant,
	})
	if err != nil {
		return nil, optimizer.State{}, err
	}
	if resume != nil {
		if err := opt.Restore(*resume); err != nil {
			return nil, optimizer.State{}, err
		}
	}

	trajectory := make([]float64, 0, len(batches))
	for _, tokens := range batches {
		scatter(m, names, weights)
		loss, grads, err := m.trainingLossAndGrads(tokens)
		if err != nil {
			return nil, optimizer.State{}, err
		}
		trajectory = append(trajectory, loss)
		m.gatherGrads(names, gradients, grads)
		opt.Step()
	}
	scatter(m, names, weights)
	return trajectory, opt.Snapshot(), nil
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

// trainSetup: sorted tensors, flat buffers, compiled geometry, resolved LR.
func (m *Model) trainSetup(baseLR float64) (names []string, weights, gradients []float32, plan optimizer.Plan, resolvedLR float64, err error) {
	names, plan, resolvedLR, err = m.trainingPlan(baseLR)
	if err != nil {
		return nil, nil, nil, optimizer.Plan{}, 0, err
	}
	weights = make([]float32, plan.ParameterCount())
	gradients = make([]float32, plan.ParameterCount())
	offset := 0
	for _, name := range names {
		copy(weights[offset:], m.Weights[name])
		offset += len(m.Weights[name])
	}
	return names, weights, gradients, plan, resolvedLR, nil
}

// TrainingPlan compiles dense parameter and Muon identity without storage.
func (m *Model) TrainingPlan(baseLR float64) (optimizer.Plan, float64, error) {
	_, plan, resolvedLR, err := m.trainingPlan(baseLR)
	return plan, resolvedLR, err
}

func (m *Model) trainingPlan(baseLR float64) ([]string, optimizer.Plan, float64, error) {
	names := slices.Sorted(maps.Keys(m.Weights))
	var totalElements uint64
	for _, name := range names {
		var valid bool
		totalElements, valid = checked.Add64(totalElements, uint64(len(m.Weights[name])))
		if !valid {
			return nil, optimizer.Plan{}, 0, fmt.Errorf("densecausal: trainable parameter count overflows uint64")
		}
	}
	total, valid := checked.Int(totalElements)
	if !valid {
		return nil, optimizer.Plan{}, 0, fmt.Errorf("densecausal: trainable parameter count exceeds int")
	}
	if baseLR <= 0 {
		baseLR = optimizer.DeriveBaseLR(total)
	}
	specs := make([]optimizer.GroupSpec, 0, len(names))
	offset := 0
	for _, name := range names {
		values := m.Weights[name]
		rows, cols, err := tensorGeometry(m.Shapes[name], len(values))
		if err != nil {
			return nil, optimizer.Plan{}, 0, fmt.Errorf("densecausal: %s: %w", name, err)
		}
		specs = append(specs, optimizer.GroupSpec{
			Name: name, Start: offset, End: offset + len(values), Rows: rows, Cols: cols,
		})
		offset += len(values)
	}
	plan, err := optimizer.CompilePlan(total, specs)
	return names, plan, baseLR, err
}

// gatherGrads: tensor gradients -> flat plan order; absent tensors zeroed.
func (m *Model) gatherGrads(names []string, gradients []float32, grads Grads) {
	offset := 0
	for _, name := range names {
		n := len(m.Weights[name])
		if g, ok := grads[name]; ok {
			copy(gradients[offset:offset+n], g)
		} else {
			clear(gradients[offset : offset+n])
		}
		offset += n
	}
}

// scatter copies the flat optimizer weight buffer back into the model tensors.
func scatter(m *Model, names []string, weights []float32) {
	offset := 0
	for _, name := range names {
		n := len(m.Weights[name])
		copy(m.Weights[name], weights[offset:offset+n])
		offset += n
	}
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
