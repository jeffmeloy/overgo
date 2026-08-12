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

// muonMatrixGroupResident runs one Muon update on RESIDENT device buffers
// dW/dG/dM (already populated -- sub-buffers of a batched flat step, or a single
// staged group). It allocates only the Newton-Schulz scratch, updates dW and dM
// in place, and transfers NOTHING; the caller owns staging. This is the core the
// batched step reuses so a full plan uploads/downloads the flat weight buffer
// once instead of once per group.
//
//	direction = nesterov(mu, momentum, gradient); momentum := mu*momentum + gradient
//	orthogonalized = NewtonSchulz(direction)
//	weights -= sqrt(max(rows,cols)) * stepRMS(mu) * rate * orthogonalized
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
	nsb, err := o.allocBuffers(n, gramN) // nsb.dX carries the direction
	if err != nil {
		return err
	}
	defer nsb.free(o.lib)
	dTmp, err := o.lib.MemAlloc(uint64(n) * 4)
	if err != nil {
		return err
	}
	defer o.lib.MemFree(dTmp)

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
	return o.add(dW, dTmp, dW, n)
}

// muonMatrixGroup stages one group's host weights/gradient/momentum to device,
// runs muonMatrixGroupResident, and copies weights + momentum back. This is the
// single-group path (deviceMuonMatrixStep); the full plan uses the resident core
// directly on one batched upload.
func (o *deviceOps) muonMatrixGroup(weights, gradient, momentum []float32, rows, cols int, mu, rate float64) error {
	n := rows * cols
	if rows <= 0 || cols <= 0 || len(weights) != n || len(gradient) != n || len(momentum) != n {
		return fmt.Errorf("muonMatrixGroup: shape mismatch (n=%d w=%d g=%d m=%d)", n, len(weights), len(gradient), len(momentum))
	}
	var dW, dG, dM driver.DevicePtr
	extra := []*driver.DevicePtr{&dW, &dG, &dM}
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
	if err := o.muonMatrixGroupResident(dW, dG, dM, rows, cols, mu, rate); err != nil {
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
// momentum), matching host Optimizer.Step within tolerance. It is a RESIDENT
// BATCHED step: the flat weights/gradients/momentum upload to device ONCE, every
// Muon matrix group runs on its resident sub-buffer (no per-group transfer), and
// the flat weights/momentum download ONCE. The cheap sign and frozen groups run
// host-side first (disjoint ranges), so their updates are already in the buffers
// the single upload carries. weights, gradients (consumed/zeroed) and momentum
// update in place. Eliminates the per-group round-trips of the old per-group path.
func DeviceMuonStepPlan(worker *device.Worker, weights, gradients, momentum []float32, plan Plan, step int, config Config) error {
	rate := config.LearningRate(step)
	muF := float32(config.Momentum)

	// Sign + frozen groups host-side first (disjoint from the Muon ranges), so the
	// one flat upload already carries their weight/momentum updates.
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
		ops, err := newDeviceOps(state) // open cuBLAS + load ops_f32 once for the whole step
		if err != nil {
			return err
		}
		defer ops.close()

		n := len(weights)
		var dW, dG, dM driver.DevicePtr
		bufs := []*driver.DevicePtr{&dW, &dG, &dM}
		for _, p := range bufs {
			ptr, err := ops.lib.MemAlloc(uint64(n) * 4)
			if err != nil {
				for _, q := range bufs {
					if *q != 0 {
						ops.lib.MemFree(*q)
					}
				}
				return err
			}
			*p = ptr
		}
		defer func() {
			for _, p := range bufs {
				ops.lib.MemFree(*p)
			}
		}()

		// One upload of the whole flat buffers (sign/frozen updates already applied).
		if err := ops.lib.MemcpyHtoD(dW, driver.Bytes(weights)); err != nil {
			return err
		}
		if err := ops.lib.MemcpyHtoD(dG, driver.Bytes(gradients)); err != nil {
			return err
		}
		if err := ops.lib.MemcpyHtoD(dM, driver.Bytes(momentum)); err != nil {
			return err
		}

		off := func(base driver.DevicePtr, elems int) driver.DevicePtr {
			return base + driver.DevicePtr(uint64(elems)*4)
		}
		for _, group := range plan.groups {
			if group.Frozen || group.Update != UpdateMuon {
				continue
			}
			if err := ops.muonMatrixGroupResident(
				off(dW, group.Start), off(dG, group.Start), off(dM, group.Start),
				group.Rows, group.Cols, config.Momentum, rate,
			); err != nil {
				return err
			}
		}

		if err := ops.lib.StreamSynchronize(ops.stream); err != nil {
			return err
		}
		// One download of the whole flat weights + momentum.
		if err := ops.lib.MemcpyDtoH(driver.Bytes(weights), dW); err != nil {
			return err
		}
		if err := ops.lib.MemcpyDtoH(driver.Bytes(momentum), dM); err != nil {
			return err
		}
		// Muon groups' gradients are consumed.
		for _, group := range plan.groups {
			if !group.Frozen && group.Update == UpdateMuon {
				clear(gradients[group.Start:group.End])
			}
		}
		return nil
	})
}
