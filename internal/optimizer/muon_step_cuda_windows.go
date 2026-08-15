//go:build windows

package optimizer

import (
	"context"
	"errors"
	"fmt"
	"math"

	"overgo/internal/cuda/device"
	"overgo/internal/cuda/driver"
)

// deviceMuonMatrixStep: staged single-matrix Muon update.
func deviceMuonMatrixStep(worker *device.Worker, weights, gradient, momentum []float32, rows, cols int, mu, rate float64) error {
	return worker.Do(context.Background(), func(state *device.State) error {
		ops, err := newDeviceOps(state)
		if err != nil {
			return err
		}
		defer ops.close()
		return ops.muonMatrixGroup(weights, gradient, momentum, rows, cols, mu, rate)
	})
}

// muonMatrixGroupResident: transfer-free update on resident W/G/M.
func (o *deviceOps) muonMatrixGroupResident(dW, dG, dM driver.DevicePtr, rows, cols int, mu, rate float64) error {
	n := rows * cols
	if rows <= 0 || cols <= 0 {
		return fmt.Errorf("muonMatrixGroupResident: bad shape (rows=%d cols=%d)", rows, cols)
	}
	if uint64(n) > math.MaxUint32 {
		return fmt.Errorf("muonMatrixGroupResident: group too large for a 32-bit element count")
	}
	gramN := gramDim(rows, cols)
	gramN *= gramN
	scratch, err := o.allocMuonScratch(n, gramN)
	if err != nil {
		return err
	}
	defer scratch.free(o.lib)
	return o.muonMatrixGroupResidentWithScratch(dW, dG, dM, rows, cols, mu, rate, scratch)
}

type muonScratch struct {
	ns  nsBuffers
	tmp driver.DevicePtr
}

func (o *deviceOps) allocMuonScratch(matrix, square int) (muonScratch, error) {
	ns, err := o.allocBuffers(matrix, square)
	if err != nil {
		return muonScratch{}, err
	}
	tmp, err := o.allocF32(matrix)
	if err != nil {
		ns.free(o.lib)
		return muonScratch{}, err
	}
	return muonScratch{ns: ns, tmp: tmp}, nil
}

func (s muonScratch) free(lib *driver.Library) {
	s.ns.free(lib)
	freeDevicePointers(lib, s.tmp)
}

func (o *deviceOps) muonMatrixGroupResidentWithScratch(
	dW, dG, dM driver.DevicePtr,
	rows, cols int,
	mu, rate float64,
	scratch muonScratch,
) error {
	n := rows * cols
	if rows <= 0 || cols <= 0 || n <= 0 || scratch.ns.dX == 0 || scratch.tmp == 0 {
		return fmt.Errorf("muonMatrixGroupResidentWithScratch: invalid group or scratch")
	}
	scale := math.Sqrt(float64(max(rows, cols))) * stepRMS(mu) * rate

	muF := float32(mu)
	if err := o.scale(dM, dM, muF, n); err != nil {
		return err
	}
	if err := o.add(dM, dG, dM, n); err != nil {
		return err
	}
	if err := o.scale(dM, scratch.ns.dX, muF, n); err != nil {
		return err
	}
	if err := o.add(scratch.ns.dX, dG, scratch.ns.dX, n); err != nil {
		return err
	}
	final, err := o.newtonSchulz(scratch.ns, rows, cols)
	if err != nil {
		return err
	}
	if err := o.scale(final, scratch.tmp, float32(-scale), n); err != nil {
		return err
	}
	return o.add(dW, scratch.tmp, dW, n)
}

// muonMatrixGroup: stage, update, return weights and momentum.
func (o *deviceOps) muonMatrixGroup(weights, gradient, momentum []float32, rows, cols int, mu, rate float64) error {
	n := rows * cols
	if rows <= 0 || cols <= 0 || len(weights) != n || len(gradient) != n || len(momentum) != n {
		return fmt.Errorf("muonMatrixGroup: shape mismatch (n=%d w=%d g=%d m=%d)", n, len(weights), len(gradient), len(momentum))
	}
	buffers, err := o.uploadF32Set(weights, gradient, momentum)
	if err != nil {
		return err
	}
	defer freeDevicePointers(o.lib, buffers...)
	dW, dG, dM := buffers[0], buffers[1], buffers[2]

	if err := o.muonMatrixGroupResident(dW, dG, dM, rows, cols, mu, rate); err != nil {
		return err
	}
	if err := o.lib.StreamSynchronize(o.stream); err != nil {
		return err
	}
	if err := o.lib.MemcpyDtoH(driver.Bytes(weights), dW); err != nil {
		return err
	}
	return o.lib.MemcpyDtoH(driver.Bytes(momentum), dM)
}

// ResidentMatrix is one Muon matrix update target whose weights, gradient and
// momentum already live on the device (caller-owned persistent buffers). Rows/Cols
// are the densecausal matrix geometry (W[out,in]).
type ResidentMatrix struct {
	Weights, Gradient, Momentum driver.DevicePtr
	Rows, Cols                  int
}

// ResidentMuonPlan retains optimizer module, cuBLAS handle, and NS scratch.
type ResidentMuonPlan struct {
	worker  *device.Worker
	plan    Plan
	config  Config
	ops     *deviceOps
	scratch muonScratch
	closed  bool
}

func NewResidentMuonPlan(worker *device.Worker, plan Plan, config Config) (*ResidentMuonPlan, error) {
	if worker == nil || plan.Identity() == "" || plan.ParameterCount() <= 0 {
		return nil, errors.New("resident Muon plan: invalid authority")
	}
	if err := config.validate(); err != nil {
		return nil, err
	}
	session := &ResidentMuonPlan{worker: worker, plan: plan, config: config}
	err := worker.Do(context.Background(), func(state *device.State) error {
		ops, err := newDeviceOps(state)
		if err != nil {
			return err
		}
		var scratch muonScratch
		if plan.maxMatrix > 0 {
			scratch, err = ops.allocMuonScratch(plan.maxMatrix, plan.maxSquare)
			if err != nil {
				ops.close()
				return err
			}
		}
		session.ops, session.scratch = ops, scratch
		return nil
	})
	if err != nil {
		return nil, err
	}
	return session, nil
}

func (s *ResidentMuonPlan) Step(weights, gradients, momentum driver.DevicePtr, step int) error {
	if s == nil || s.closed || s.ops == nil || weights == 0 || gradients == 0 || momentum == 0 || step <= 0 {
		return errors.New("resident Muon plan: unavailable")
	}
	rate := s.config.LearningRate(step)
	return s.worker.Do(context.Background(), func(_ *device.State) error {
		for _, group := range s.plan.groups {
			if group.Frozen {
				continue
			}
			if err := s.ops.muonMatrixGroupResidentWithScratch(
				offsetF32(weights, group.Start), offsetF32(gradients, group.Start), offsetF32(momentum, group.Start),
				group.Rows, group.Cols, s.config.Momentum, rate, s.scratch,
			); err != nil {
				return err
			}
		}
		if err := s.ops.lib.MemsetD32Async(gradients, 0, uint64(s.plan.ParameterCount()), s.ops.stream); err != nil {
			return err
		}
		return s.ops.lib.StreamSynchronize(s.ops.stream)
	})
}

func (s *ResidentMuonPlan) Close() error {
	if s == nil || s.closed {
		return nil
	}
	s.closed = true
	return s.worker.Do(context.Background(), func(_ *device.State) error {
		if s.scratch.tmp != 0 {
			s.scratch.free(s.ops.lib)
		}
		s.ops.close()
		s.ops, s.scratch = nil, muonScratch{}
		return nil
	})
}

// DeviceMuonMatricesResident applies one Muon step to already-resident device
// matrices: nothing is uploaded or downloaded. Weights, gradient and momentum stay
// on the device across calls; the caller uploads them once at the start of training
// and downloads only at checkpoint. This is the same Newton-Schulz update
// muonMatrixGroupResident applies inside DeviceMuonStepPlan, lifted onto persistent
// caller buffers so the across-step training loop never scatters/gathers weights.
func DeviceMuonMatricesResident(worker *device.Worker, matrices []ResidentMatrix, step int, config Config) error {
	rate := config.LearningRate(step)
	return worker.Do(context.Background(), func(state *device.State) error {
		ops, err := newDeviceOps(state)
		if err != nil {
			return err
		}
		defer ops.close()
		for _, m := range matrices {
			if err := ops.muonMatrixGroupResident(m.Weights, m.Gradient, m.Momentum, m.Rows, m.Cols, config.Momentum, rate); err != nil {
				return err
			}
		}
		return ops.lib.StreamSynchronize(ops.stream)
	})
}

// DeviceMuonPlanResident applies and clears one flat resident Muon plan.
func DeviceMuonPlanResident(
	worker *device.Worker,
	weights, gradients, momentum driver.DevicePtr,
	plan Plan,
	step int,
	config Config,
) error {
	if worker == nil || weights == 0 || gradients == 0 || momentum == 0 || plan.Identity() == "" || step <= 0 {
		return fmt.Errorf("DeviceMuonPlanResident: invalid buffer, plan, or step")
	}
	if err := config.validate(); err != nil {
		return err
	}
	session, err := NewResidentMuonPlan(worker, plan, config)
	if err != nil {
		return err
	}
	defer session.Close()
	return session.Step(weights, gradients, momentum, step)
}

// DeviceMuonStepPlan: one flat upload/download; resident matrix updates.
func DeviceMuonStepPlan(worker *device.Worker, weights, gradients, momentum []float32, plan Plan, step int, config Config) error {
	if worker == nil || plan.Identity() == "" || step <= 0 ||
		len(weights) != plan.ParameterCount() || len(gradients) != len(weights) || len(momentum) != len(weights) {
		return errors.New("device Muon step: invalid worker, plan, step, or storage")
	}
	if err := config.validate(); err != nil {
		return err
	}
	rate := config.LearningRate(step)
	return worker.Do(context.Background(), func(state *device.State) error {
		ops, err := newDeviceOps(state)
		if err != nil {
			return err
		}
		defer ops.close()

		buffers, err := ops.uploadF32Set(weights, gradients, momentum)
		if err != nil {
			return err
		}
		defer freeDevicePointers(ops.lib, buffers...)
		dW, dG, dM := buffers[0], buffers[1], buffers[2]

		for _, group := range plan.groups {
			if group.Frozen {
				continue
			}
			if err := ops.muonMatrixGroupResident(
				offsetF32(dW, group.Start), offsetF32(dG, group.Start), offsetF32(dM, group.Start),
				group.Rows, group.Cols, config.Momentum, rate,
			); err != nil {
				return err
			}
		}

		if err := ops.lib.StreamSynchronize(ops.stream); err != nil {
			return err
		}
		if err := ops.lib.MemcpyDtoH(driver.Bytes(weights), dW); err != nil {
			return err
		}
		if err := ops.lib.MemcpyDtoH(driver.Bytes(momentum), dM); err != nil {
			return err
		}
		for _, group := range plan.groups {
			if !group.Frozen {
				clear(gradients[group.Start:group.End])
			}
		}
		return nil
	})
}
