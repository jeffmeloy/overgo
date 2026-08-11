package densecausal

import (
	"fmt"

	"overgo/internal/optimizer"
)

// Train: full-parameter Muon updates; pre-update loss trajectory.
func (m *Model) Train(tokens []int, steps int, baseLR, mu float64) ([]float64, error) {
	trajectory, _, err := m.train(tokens, steps, baseLR, mu, nil)
	return trajectory, err
}

// TrainState: resumable Muon state; model weights checkpoint separately.
type TrainState = optimizer.State

// TrainResume: checkpoint-continuable Muon updates.
func (m *Model) TrainResume(tokens []int, steps int, baseLR, mu float64, resume *TrainState) ([]float64, TrainState, error) {
	return m.train(tokens, steps, baseLR, mu, resume)
}

func (m *Model) train(tokens []int, steps int, baseLR, mu float64, resume *optimizer.State) ([]float64, optimizer.State, error) {
	pack, err := optimizer.NewTensorPack(m.Weights, func(name string, length int) (int, int, error) {
		return tensorGeometry(m.Shapes[name], length)
	})
	if err != nil {
		return nil, optimizer.State{}, fmt.Errorf("densecausal: %w", err)
	}
	if baseLR <= 0 {
		baseLR = optimizer.DeriveBaseLR(pack.ParameterCount())
	}
	opt, err := pack.NewOptimizer(optimizer.Config{
		BaseLearningRate: baseLR, Momentum: mu, Schedule: optimizer.ScheduleConstant,
	})
	if err != nil {
		return nil, optimizer.State{}, err
	}
	if resume != nil {
		if err := opt.Restore(*resume); err != nil {
			return nil, optimizer.State{}, err
		}
	}

	trajectory := make([]float64, 0, steps)
	for step := 0; step < steps; step++ {
		pack.Scatter()
		loss, _, grads, err := m.LossAndGrads(tokens)
		if err != nil {
			return nil, optimizer.State{}, err
		}
		trajectory = append(trajectory, loss)
		if err := pack.GatherGradients(grads); err != nil {
			return nil, optimizer.State{}, err
		}
		opt.Step()
	}
	pack.Scatter()
	return trajectory, opt.Snapshot(), nil
}

// tensorGeometry: stored shape to Muon matrix geometry.
func tensorGeometry(shape []int, length int) (int, int, error) {
	switch len(shape) {
	case 2:
		if shape[0]*shape[1] != length {
			return 0, 0, fmt.Errorf("shape %v does not match length %d", shape, length)
		}
		return shape[0], shape[1], nil
	case 1:
		if shape[0] != length {
			return 0, 0, fmt.Errorf("shape %v does not match length %d", shape, length)
		}
		return shape[0], 1, nil
	default:
		return 0, 0, fmt.Errorf("unsupported tensor rank %d (shape %v)", len(shape), shape)
	}
}
