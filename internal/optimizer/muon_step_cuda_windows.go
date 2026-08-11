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
	n := rows * cols
	if rows <= 0 || cols <= 0 || len(weights) != n || len(gradient) != n || len(momentum) != n {
		return fmt.Errorf("deviceMuonMatrixStep: shape mismatch (n=%d w=%d g=%d m=%d)", n, len(weights), len(gradient), len(momentum))
	}
	if uint64(n) > math.MaxUint32 {
		return fmt.Errorf("deviceMuonMatrixStep: group too large for a 32-bit element count")
	}
	scale := math.Sqrt(float64(max(rows, cols))) * stepRMS(mu) * rate

	return worker.Do(context.Background(), func(state *device.State) error {
		ops, err := newDeviceOps(state)
		if err != nil {
			return err
		}
		defer ops.close()

		gramN := gramDim(rows, cols)
		gramN *= gramN
		nsb, err := ops.allocBuffers(n, gramN) // nsb.dX carries the direction
		if err != nil {
			return err
		}
		defer nsb.free(ops.lib)

		var dW, dG, dM, dTmp driver.DevicePtr
		extra := []*driver.DevicePtr{&dW, &dG, &dM, &dTmp}
		for _, p := range extra {
			ptr, err := ops.lib.MemAlloc(uint64(n) * 4)
			if err != nil {
				for _, q := range extra {
					if *q != 0 {
						ops.lib.MemFree(*q)
					}
				}
				return err
			}
			*p = ptr
		}
		defer func() {
			for _, p := range extra {
				ops.lib.MemFree(*p)
			}
		}()

		if err := ops.lib.MemcpyHtoD(dW, driver.Bytes(weights)); err != nil {
			return err
		}
		if err := ops.lib.MemcpyHtoD(dG, driver.Bytes(gradient)); err != nil {
			return err
		}
		if err := ops.lib.MemcpyHtoD(dM, driver.Bytes(momentum)); err != nil {
			return err
		}

		muF := float32(mu)
		// nesterov: nextM = mu*M + G;  direction = mu*nextM + G.
		if err := ops.scale(dM, dM, muF, n); err != nil { // dM = mu*M
			return err
		}
		if err := ops.add(dM, dG, dM, n); err != nil { // dM = mu*M + G = nextM
			return err
		}
		if err := ops.scale(dM, nsb.dX, muF, n); err != nil { // dX = mu*nextM
			return err
		}
		if err := ops.add(nsb.dX, dG, nsb.dX, n); err != nil { // dX = mu*nextM + G = direction
			return err
		}

		final, err := ops.newtonSchulz(nsb, rows, cols) // orthogonalize the direction in place
		if err != nil {
			return err
		}

		// weights -= scale * orthogonalized  =>  tmp = (-scale)*final; weights += tmp.
		if err := ops.scale(final, dTmp, float32(-scale), n); err != nil {
			return err
		}
		if err := ops.add(dW, dTmp, dW, n); err != nil {
			return err
		}

		if err := ops.lib.StreamSynchronize(ops.stream); err != nil {
			return err
		}
		if err := ops.lib.MemcpyDtoH(driver.Bytes(weights), dW); err != nil {
			return err
		}
		return ops.lib.MemcpyDtoH(driver.Bytes(momentum), dM) // dM holds nextM
	})
}
