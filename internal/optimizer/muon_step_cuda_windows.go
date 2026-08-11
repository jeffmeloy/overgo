//go:build windows

package optimizer

import (
	"context"
	"fmt"
	"math"

	"overgo/internal/cuda/device"
	"overgo/internal/cuda/driver"
)

// deviceMuonMatrixStep runs one Muon update for a single matrix parameter group
// on the GPU, matching host stepMuon within tolerance. It composes the resident
// device primitives: nesterov momentum (ops_f32 scale/add) -> Newton-Schulz
// orthogonalization (deviceOps.newtonSchulz) -> scaled weight update, all on
// device buffers. weights and momentum are updated in place; gradient is
// consumed. mu is the momentum coefficient, rate the learning rate for this step
// (the caller supplies the schedule's value).
//
//	direction = nesterov(mu, momentum, gradient); momentum := mu*momentum + gradient
//	orthogonalized = NewtonSchulz(direction)
//	weights -= sqrt(max(rows,cols)) * stepRMS(mu) * rate * orthogonalized
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

// muonMatrixGroup runs one Muon update for a matrix group on device buffers,
// reusing the caller's already-open cuBLAS handle + ops_f32 module (so a
// full-plan step opens the context once, not once per group). weights, gradient
// and momentum are host slices for this group; they are staged to device,
// updated, and copied back. momentum becomes the nesterov next state.
func (o *deviceOps) muonMatrixGroup(weights, gradient, momentum []float32, rows, cols int, mu, rate float64) error {
	n := rows * cols
	if rows <= 0 || cols <= 0 || len(weights) != n || len(gradient) != n || len(momentum) != n {
		return fmt.Errorf("muonMatrixGroup: shape mismatch (n=%d w=%d g=%d m=%d)", n, len(weights), len(gradient), len(momentum))
	}
	if uint64(n) > math.MaxUint32 {
		return fmt.Errorf("muonMatrixGroup: group too large for a 32-bit element count")
	}
	scale := math.Sqrt(float64(max(rows, cols))) * stepRMS(mu) * rate

	gramN := gramDim(rows, cols)
	gramN *= gramN
	nsb, err := o.allocBuffers(n, gramN) // nsb.dX carries the direction
	if err != nil {
		return err
	}
	defer nsb.free(o.lib)

	var dW, dG, dM, dTmp driver.DevicePtr
	extra := []*driver.DevicePtr{&dW, &dG, &dM, &dTmp}
	for _, p := range extra {
		ptr, err := o.lib.MemAlloc(uint64(n) * 4)
		if err != nil {
			for _, q := range extra {
				if *q != 0 {
					o.lib.MemFree(*q)
				}
			}
			return err
		}
		*p = ptr
	}
	defer func() {
		for _, p := range extra {
			o.lib.MemFree(*p)
		}
	}()

	if err := o.lib.MemcpyHtoD(dW, driver.Bytes(weights)); err != nil {
		return err
	}
	if err := o.lib.MemcpyHtoD(dG, driver.Bytes(gradient)); err != nil {
		return err
	}
	if err := o.lib.MemcpyHtoD(dM, driver.Bytes(momentum)); err != nil {
		return err
	}

	muF := float32(mu)
	// nesterov: nextM = mu*M + G;  direction = mu*nextM + G.
	if err := o.scale(dM, dM, muF, n); err != nil { // dM = mu*M
		return err
	}
	if err := o.add(dM, dG, dM, n); err != nil { // dM = mu*M + G = nextM
		return err
	}
	if err := o.scale(dM, nsb.dX, muF, n); err != nil { // dX = mu*nextM
		return err
	}
	if err := o.add(nsb.dX, dG, nsb.dX, n); err != nil { // dX = mu*nextM + G = direction
		return err
	}

	final, err := o.newtonSchulz(nsb, rows, cols) // orthogonalize the direction in place
	if err != nil {
		return err
	}

	// weights -= scale * orthogonalized  =>  tmp = (-scale)*final; weights += tmp.
	if err := o.scale(final, dTmp, float32(-scale), n); err != nil {
		return err
	}
	if err := o.add(dW, dTmp, dW, n); err != nil {
		return err
	}

	if err := o.lib.StreamSynchronize(o.stream); err != nil {
		return err
	}
	if err := o.lib.MemcpyDtoH(driver.Bytes(weights), dW); err != nil {
		return err
	}
	return o.lib.MemcpyDtoH(driver.Bytes(momentum), dM) // dM holds nextM
}

// DeviceMuonStepPlan runs one full Muon step for `plan` (fp32, device-native
// momentum), matching host Optimizer.Step within tolerance. Matrix (Muon) groups
// run on the GPU via deviceMuonMatrixStep; the cheap sign and frozen groups run
// host-side in the same pass. weights, gradients (consumed/zeroed) and momentum
// update in place. This is correctness-first: each matrix group takes its own
// device context/transfers; a resident batched step is a later optimization.
func DeviceMuonStepPlan(worker *device.Worker, weights, gradients, momentum []float32, plan Plan, step int, config Config) error {
	rate := config.LearningRate(step)
	muF := float32(config.Momentum)
	return worker.Do(context.Background(), func(state *device.State) error {
		ops, err := newDeviceOps(state) // open cuBLAS + load ops_f32 once for the whole step
		if err != nil {
			return err
		}
		defer ops.close()
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
			case group.Update == UpdateMuon:
				if err := ops.muonMatrixGroup(
					weights[group.Start:group.End],
					gradients[group.Start:group.End],
					momentum[group.Start:group.End],
					group.Rows, group.Cols, config.Momentum, rate,
				); err != nil {
					return err
				}
				clear(gradients[group.Start:group.End])
			}
		}
		return nil
	})
}
