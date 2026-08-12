//go:build windows

package optimizer

import (
	"context"
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
	scale := math.Sqrt(float64(max(rows, cols))) * stepRMS(mu) * rate

	gramN := gramDim(rows, cols)
	gramN *= gramN
	nsb, err := o.allocBuffers(n, gramN)
	if err != nil {
		return err
	}
	defer nsb.free(o.lib)
	dTmp, err := o.allocF32(n)
	if err != nil {
		return err
	}
	defer freeDevicePointers(o.lib, dTmp)

	muF := float32(mu)
	if err := o.scale(dM, dM, muF, n); err != nil {
		return err
	}
	if err := o.add(dM, dG, dM, n); err != nil {
		return err
	}
	if err := o.scale(dM, nsb.dX, muF, n); err != nil {
		return err
	}
	if err := o.add(nsb.dX, dG, nsb.dX, n); err != nil {
		return err
	}
	final, err := o.newtonSchulz(nsb, rows, cols)
	if err != nil {
		return err
	}
	if err := o.scale(final, dTmp, float32(-scale), n); err != nil {
		return err
	}
	return o.add(dW, dTmp, dW, n)
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

// DeviceMuonStepPlan: one flat upload/download; resident matrix updates.
func DeviceMuonStepPlan(worker *device.Worker, weights, gradients, momentum []float32, plan Plan, step int, config Config) error {
	rate := config.LearningRate(step)
	muF := float32(config.Momentum)

	// Host-only disjoint groups first.
	for _, group := range plan.groups {
		switch {
		case group.Frozen:
			clear(gradients[group.Start:group.End])
		case group.Update == UpdateSign:
			for i := group.Start; i < group.End; i++ {
				next := muF*momentum[i] + gradients[i]
				direction := muF*next + gradients[i]
				momentum[i] = next
				if gradients[i] != 0 {
					switch {
					case direction > 0:
						weights[i] -= float32(rate)
					case direction < 0:
						weights[i] += float32(rate)
					}
				}
				gradients[i] = 0
			}
		}
	}

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
			if group.Frozen || group.Update != UpdateMuon {
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
			if !group.Frozen && group.Update == UpdateMuon {
				clear(gradients[group.Start:group.End])
			}
		}
		return nil
	})
}
