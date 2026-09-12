package hybridtrain

import (
	"errors"
	"fmt"

	"overgo/internal/checked"
	"overgo/internal/optimizer"
)

// TrainingOptions controls optional optimizer continuation and observations.
// Resume requires the matching model weights, input and execution backend.
// Checkpoint runs at the final complete update boundary, including an Observe
// stop, after resident weights have been read back. It owns a portable optimizer
// snapshot; callers retain responsibility for weight and dataset/RNG publication.
// An execution failure inside an update cannot publish a checkpoint.
type TrainingOptions struct {
	Resume     *optimizer.State
	Observe    func(step int, loss float64) error // absolute zero-based update index
	Checkpoint func(optimizer.State) error
}

func (m *Model) resumeParts(steps int, cfg optimizer.Config, options TrainingOptions, matrixF32 bool) (optimizer.State, optimizer.State, error) {
	var mat, vec optimizer.State
	if err := cfg.Validate(); err != nil {
		return mat, vec, err
	}
	start := 0
	if resume := options.Resume; resume != nil {
		if err := optimizer.ValidateState(*resume, m.program.OptimizerIdentity(), len(m.matW)+len(m.vecW)); err != nil {
			return mat, vec, err
		}
		if resume.Config != cfg {
			return mat, vec, errors.New("hybrid resume: optimizer config differs")
		}
		mat, vec = *resume, *resume
		mat.PlanIdentity, vec.PlanIdentity = m.matPlan.Identity(), m.vecPlan.Identity()
		mat.Momentum, vec.Momentum = resume.Momentum[:len(m.matW)], resume.Momentum[len(m.matW):]
		if matrixF32 {
			for index, value := range mat.Momentum {
				if float64(float32(value)) != value {
					return mat, vec, fmt.Errorf("hybrid resume: matrix momentum %d is not exactly representable in f32", index)
				}
			}
		}
		start = resume.Step
	}
	if _, ok := checked.AddInt(start, steps); !ok {
		return mat, vec, errors.New("hybrid training: negative or overflowing update extent")
	}
	return mat, vec, nil
}

func restoreOptimizer(opt *optimizer.Optimizer, state optimizer.State) error {
	if state.PlanIdentity == "" {
		return nil
	}
	return opt.Restore(state)
}

// checkpoint joins the canonical matrix/vector partition only on request.
// Matrix momentum has already been copied or downloaded by the owning backend.
func (m *Model) checkpoint(options TrainingOptions, mat optimizer.State, vec *optimizer.Optimizer) error {
	vector := vec.Snapshot()
	if mat.Step != vector.Step || mat.Config != vector.Config {
		return errors.New("hybrid checkpoint: matrix/vector progress differs")
	}
	mat.PlanIdentity = m.program.OptimizerIdentity()
	mat.Momentum = append(mat.Momentum, vector.Momentum...)
	if err := optimizer.ValidateState(mat, m.program.OptimizerIdentity(), len(m.matW)+len(m.vecW)); err != nil {
		return err
	}
	return options.Checkpoint(mat)
}

func (m *Model) checkpointF32(options TrainingOptions, momentum []float32, step int, cfg optimizer.Config, vec *optimizer.Optimizer) error {
	values := make([]float64, len(momentum), len(momentum)+len(m.vecW))
	for index, value := range momentum {
		values[index] = float64(value)
	}
	return m.checkpoint(options, optimizer.State{Step: step, Config: cfg, Momentum: values}, vec)
}
