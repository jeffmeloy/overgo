//go:build windows

package optimizer

import (
	"context"
	"errors"

	"overgo/internal/cuda/device"
	"overgo/internal/cuda/driver"
)

// StreamedMuon holds a worker's Muon operations (its cuBLAS library and
// handles) across a training run's streamed steps: opening them took 75 to 86
// ms a step, where a streamed step of a 60M-parameter plan takes about 370.
// Each step still allocates its staging and Newton-Schulz scratch and frees
// them on return, so nothing the session holds sits beside the run's forward
// and backward passes.
type StreamedMuon struct {
	worker *device.Worker
	ops    *deviceOps
}

// NewStreamedMuon opens worker's Muon operations for a run's streamed steps; Close releases them.
func NewStreamedMuon(worker *device.Worker) (*StreamedMuon, error) {
	if worker == nil {
		return nil, errors.New("device Muon streamed step: a worker is required")
	}
	session := &StreamedMuon{worker: worker}
	if err := worker.Do(context.Background(), func(state *device.State) error {
		var err error
		session.ops, err = newDeviceOps(state)
		return err
	}); err != nil {
		return nil, err
	}
	return session, nil
}

// Close releases the session's Muon operations.
func (s *StreamedMuon) Close() error {
	return s.worker.Do(context.Background(), func(*device.State) error {
		s.ops.close()
		return nil
	})
}

// Step applies one Muon step to host-master flat storage by staging one
// matrix group at a time on the device. This is the host-master lane for
// plans whose full weight/gradient/momentum triple does not fit device
// memory: staging and Newton-Schulz scratch are sized once from the plan's
// largest group and reused across groups, so peak device use is bounded by
// the largest matrix, not the parameter count. Weights and momentum stream
// back to the host after each group; host gradients are cleared exactly like
// the flat device step.
func (s *StreamedMuon) Step(weights, gradients, momentum []float32, plan Plan, step int, config Config) error {
	if plan.Identity() == "" || step <= 0 ||
		len(weights) != plan.ParameterCount() || len(gradients) != len(weights) || len(momentum) != len(weights) {
		return errors.New("device Muon streamed step: invalid plan, step, or storage")
	}
	if err := config.Validate(); err != nil {
		return err
	}
	if plan.MaxMatrix() == 0 {
		// Every group is frozen; clearing gradients is the whole step.
		for _, group := range plan.Groups() {
			clear(gradients[group.Start:group.End])
		}
		return nil
	}
	rate := config.LearningRate(step)
	return s.worker.Do(context.Background(), func(*device.State) error {
		ops := s.ops
		staging, err := ops.allocF32Set(plan.MaxMatrix(), plan.MaxMatrix(), plan.MaxMatrix())
		if err != nil {
			return err
		}
		defer freeDevicePointers(ops.lib, staging...)
		dW, dG, dM := staging[0], staging[1], staging[2]

		scratch, err := ops.allocMuonScratch(plan.MaxMatrix(), plan.MaxSquare())
		if err != nil {
			return err
		}
		defer scratch.free(ops.lib)

		for _, group := range plan.Groups() {
			if group.Frozen {
				clear(gradients[group.Start:group.End])
				continue
			}
			w := weights[group.Start:group.End]
			g := gradients[group.Start:group.End]
			m := momentum[group.Start:group.End]
			if err := ops.lib.MemcpyHtoD(dW, driver.Bytes(w)); err != nil {
				return err
			}
			if err := ops.lib.MemcpyHtoD(dG, driver.Bytes(g)); err != nil {
				return err
			}
			if err := ops.lib.MemcpyHtoD(dM, driver.Bytes(m)); err != nil {
				return err
			}
			if err := ops.muonMatrixGroupResidentWithScratch(
				dW, dG, dM, group.Rows, group.Cols, config.Momentum, rate, scratch,
			); err != nil {
				return err
			}
			if err := ops.lib.StreamSynchronize(ops.stream); err != nil {
				return err
			}
			if err := ops.lib.MemcpyDtoH(driver.Bytes(w), dW); err != nil {
				return err
			}
			if err := ops.lib.MemcpyDtoH(driver.Bytes(m), dM); err != nil {
				return err
			}
			clear(g)
		}
		return nil
	})
}
