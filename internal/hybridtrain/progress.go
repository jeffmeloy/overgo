package hybridtrain

import (
	"errors"
	"fmt"

	"overgo/internal/checked"
	"overgo/internal/optimizer"
	"overgo/internal/safetensors"
)

// TrainingOptions controls optional optimizer continuation and observations.
// Resume requires the matching model weights, input and execution backend.
// Checkpoint runs at the final complete update boundary, including an Observe
// stop, after resident weights have been read back. It owns a portable optimizer
// snapshot; SaveWeights writes the weights it pairs with, and callers retain
// responsibility for their publication and the dataset/RNG state.
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
		var err error
		if mat.Momentum, mat.MomentumF32, vec.Momentum, err = m.splitMomentum(*resume, matrixF32); err != nil {
			return mat, vec, err
		}
		vec.MomentumF32 = nil
		start = resume.Step
	}
	if _, ok := checked.AddInt(start, steps); !ok {
		return mat, vec, errors.New("hybrid training: negative or overflowing update extent")
	}
	return mat, vec, nil
}

// splitMomentum parts a checkpoint's momentum into the matrix slab's, in the
// dtype it is stepped in (f32 where matrixF32), and the vector slab's f64. A
// device stepper's checkpoint holds the matrix momentum in f32, then the
// vector's in f64; one written before held both in f64, whose matrix part
// narrows exactly or is refused.
func (m *Model) splitMomentum(resume optimizer.State, matrixF32 bool) (mat64 []float64, mat32 []float32, vec []float64, err error) {
	matrix := len(m.matW)
	switch len(resume.MomentumF32) {
	case 0:
		mat64, vec = resume.Momentum[:matrix], resume.Momentum[matrix:]
	case matrix:
		mat32, vec = resume.MomentumF32, resume.Momentum
	default:
		return nil, nil, nil, fmt.Errorf("hybrid resume: f32 momentum holds %d values, not the %d matrix parameters", len(resume.MomentumF32), matrix)
	}
	switch {
	case matrixF32 && mat32 == nil:
		mat32 = make([]float32, matrix)
		for index, value := range mat64 {
			if float64(float32(value)) != value {
				return nil, nil, nil, fmt.Errorf("hybrid resume: matrix momentum %d is not exactly representable in f32", index)
			}
			mat32[index] = float32(value)
		}
		mat64 = nil
	case !matrixF32 && mat64 == nil:
		mat64 = make([]float64, matrix)
		for index, value := range mat32 {
			mat64[index] = float64(value)
		}
		mat32 = nil
	}
	return mat64, mat32, vec, nil
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

// SaveWeights writes every trainable group of the compiled program as a named
// F32 safetensors tensor with its [rows, cols] geometry, the weights a
// Checkpoint snapshot's momentum pairs with.
func (m *Model) SaveWeights(path string) error {
	tensors := make(map[string][]float32, m.matPlan.GroupCount()+m.vecPlan.GroupCount())
	shapes := make(map[string][]int, len(tensors))
	for _, slab := range []struct {
		plan   optimizer.Plan
		values []float32
	}{{m.matPlan, m.matW}, {m.vecPlan, m.vecW}} {
		for _, group := range slab.plan.Groups() {
			tensors[group.Name] = slab.values[group.Start:group.End]
			shapes[group.Name] = []int{group.Rows, group.Cols}
		}
	}
	return safetensors.Save(path, tensors, shapes, map[string]string{"program": m.program.ID().String()})
}

// checkpointF32 checkpoints a device stepper's matrix momentum in the f32 it
// is stepped in, before the vector slab's f64. The state holds the stepper's
// own host slab: a checkpoint copies none of it.
func (m *Model) checkpointF32(options TrainingOptions, momentum []float32, step int, cfg optimizer.Config, vec *optimizer.Optimizer) error {
	return m.checkpoint(options, optimizer.State{Step: step, Config: cfg, MomentumF32: momentum}, vec)
}
