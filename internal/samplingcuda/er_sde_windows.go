//go:build windows

package samplingcuda

import (
	"errors"
	"fmt"
	"unsafe"

	"overgo/internal/cuda/device"
	"overgo/internal/cuda/driver"
	"overgo/internal/cuda/kernel"
	"overgo/internal/sampling"
)

// erSDEKernel is the native kernel that applies a sampling.ERSDEPlan on the card, and erSDEThreads its block's threads.
const (
	erSDEKernel  = "er_sde_step_f32"
	erSDEThreads = 256
)

// ERSDEDevice takes ER-SDE's steps on the card (er_sde_step_f32, kernels/cuda/ops_native.cu), each as the host step
// internal/sampling's tests hold it to takes it (HostERSDE), bit for bit, so a sampling keeps its latent, its
// predictions, their divided differences and its noise on the card (moss-step-copies). Its handles are valid in the worker that loaded them, which Close unloads them in.
type ERSDEDevice struct {
	module, function uintptr
	loaded           bool
}

// ensureLoaded loads the native kernels and resolves the step's kernel once.
func (d *ERSDEDevice) ensureLoaded(state *device.State) error {
	if d.loaded {
		return nil
	}
	if state == nil || state.Driver == nil {
		return errors.New("samplingcuda: ER-SDE on the card needs a device state")
	}
	if err := kernel.ValidateAssets(); err != nil {
		return err
	}
	module, err := state.Driver.ModuleLoadData(kernel.OpsNativePTX)
	if err != nil {
		return err
	}
	function, err := state.Driver.ModuleFunction(module, erSDEKernel)
	if err != nil {
		return errors.Join(err, state.Driver.ModuleUnload(module))
	}
	d.module, d.function, d.loaded = uintptr(module), uintptr(function), true
	return nil
}

// Step applies plan to count elements on the card: latent holds the step's latent and receives the next, denoised is
// its data prediction and noise its fresh noise; from the plan's second order previous is the last step's prediction
// and first receives the step's first divided difference, and at the third difference holds the last step's. A plan
// to 0 is no step here: its next latent is the prediction itself.
func (d *ERSDEDevice) Step(state *device.State, plan sampling.ERSDEPlan, latent, denoised, noise, previous, first, difference driver.DevicePtr, count int) error {
	if plan.ToZero || plan.Order < 1 || plan.Order > sampling.ERSDEStages || count <= 0 || latent == 0 || denoised == 0 || noise == 0 ||
		plan.Order >= 2 && (previous == 0 || first == 0) || plan.Order >= 3 && difference == 0 {
		return fmt.Errorf("samplingcuda: ER-SDE on the card cannot apply a plan of order %d to %d elements with these buffers", plan.Order, count)
	}
	if err := d.ensureLoaded(state); err != nil {
		return err
	}
	elements, order := uint32(count), uint32(plan.Order)
	keep, data, noiseWeight := plan.Keep, plan.Data, plan.Noise
	divisor, firstWeight, secondWeight, span := plan.Divisor, plan.FirstWeight, plan.SecondWeight, plan.Span
	return state.Driver.LaunchKernel(
		driver.Function(d.function),
		driver.Dim3{X: (elements + erSDEThreads - 1) / erSDEThreads, Y: 1, Z: 1},
		driver.Dim3{X: erSDEThreads, Y: 1, Z: 1},
		0, state.Stream,
		[]unsafe.Pointer{
			unsafe.Pointer(&latent), unsafe.Pointer(&denoised), unsafe.Pointer(&noise), unsafe.Pointer(&previous),
			unsafe.Pointer(&first), unsafe.Pointer(&difference), unsafe.Pointer(&elements), unsafe.Pointer(&order),
			unsafe.Pointer(&keep), unsafe.Pointer(&data), unsafe.Pointer(&noiseWeight), unsafe.Pointer(&divisor),
			unsafe.Pointer(&firstWeight), unsafe.Pointer(&secondWeight), unsafe.Pointer(&span),
		},
	)
}

// Close unloads the native kernels where Step loaded them.
func (d *ERSDEDevice) Close(state *device.State) error {
	if d == nil || !d.loaded {
		return nil
	}
	d.loaded = false
	return state.Driver.ModuleUnload(driver.Module(d.module))
}
