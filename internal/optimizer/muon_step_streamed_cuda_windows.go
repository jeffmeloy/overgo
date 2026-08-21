//go:build windows

package optimizer

import (
	"context"
	"errors"

	"overgo/internal/cuda/device"
	"overgo/internal/cuda/driver"
)

// DeviceMuonStepPlanStreamed applies one Muon step to host-master flat storage
// by staging one matrix group at a time on the device. This is the
// host-master lane for plans whose full weight/gradient/momentum triple does
// not fit device memory: staging and Newton-Schulz scratch are sized once
// from the plan's largest group and reused across groups, so peak device use
// is bounded by the largest matrix, not the parameter count. Weights and
// momentum stream back to the host after each group; host gradients are
// cleared exactly like the flat device step.
func DeviceMuonStepPlanStreamed(worker *device.Worker, weights, gradients, momentum []float32, plan Plan, step int, config Config) error {
	if worker == nil || plan.Identity() == "" || step <= 0 ||
		len(weights) != plan.ParameterCount() || len(gradients) != len(weights) || len(momentum) != len(weights) {
		return errors.New("device Muon streamed step: invalid worker, plan, step, or storage")
	}
	if err := config.validate(); err != nil {
		return err
	}
	if plan.maxMatrix == 0 {
		// Every group is frozen; clearing gradients is the whole step.
		for _, group := range plan.groups {
			clear(gradients[group.Start:group.End])
		}
		return nil
	}
	rate := config.LearningRate(step)
	return worker.Do(context.Background(), func(state *device.State) error {
		ops, err := newDeviceOps(state)
		if err != nil {
			return err
		}
		defer ops.close()

		staging, err := ops.allocF32Set(plan.maxMatrix, plan.maxMatrix, plan.maxMatrix)
		if err != nil {
			return err
		}
		defer freeDevicePointers(ops.lib, staging...)
		dW, dG, dM := staging[0], staging[1], staging[2]

		scratch, err := ops.allocMuonScratch(plan.maxMatrix, plan.maxSquare)
		if err != nil {
			return err
		}
		defer scratch.free(ops.lib)

		for _, group := range plan.groups {
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
