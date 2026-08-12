//go:build windows

package devicemath

import (
	"context"
	"fmt"
	"math"
	"runtime"
	"unsafe"

	"overgo/internal/cuda/device"
	"overgo/internal/cuda/driver"
	"overgo/internal/cuda/kernel"
)

// RMSNormBackward computes the VJP of affine RMSNorm
//
//	y = x * rsqrt(mean(x^2 over d) + eps) * weight
//
// given the output cotangent dY[rows,d]. It returns dx[rows,d] and dscale[d]
// (the weight gradient, summed over rows). Uses the fused rms_norm_backward_f32
// kernel (one thread per row; dscale accumulates via atomicAdd from a zeroed
// buffer).
func RMSNormBackward(worker *device.Worker, x, weight, dY []float32, rows, d int, eps float64) (dx, dscale []float32, err error) {
	if rows <= 0 || d <= 0 || len(x) != rows*d || len(dY) != rows*d || len(weight) != d {
		return nil, nil, fmt.Errorf("RMSNormBackward: shape mismatch (rows=%d d=%d x=%d dY=%d w=%d)", rows, d, len(x), len(dY), len(weight))
	}
	if uint64(rows) > math.MaxUint32 || uint64(d) > math.MaxUint32 {
		return nil, nil, fmt.Errorf("RMSNormBackward: dimensions exceed a 32-bit count")
	}
	dx = make([]float32, rows*d)
	dscale = make([]float32, d) // zeros: the device dscale starts at 0

	err = worker.Do(context.Background(), func(state *device.State) error {
		lib := state.Driver
		module, err := lib.ModuleLoadData(kernel.OpsF32PTX)
		if err != nil {
			return err
		}
		defer lib.ModuleUnload(module)
		fn, err := lib.ModuleFunction(module, "rms_norm_backward_f32")
		if err != nil {
			return err
		}

		var dyPtr, xPtr, wPtr, dxPtr, dsPtr driver.DevicePtr
		specs := []struct {
			ptr *driver.DevicePtr
			n   int
		}{
			{&dyPtr, rows * d}, {&xPtr, rows * d}, {&wPtr, d}, {&dxPtr, rows * d}, {&dsPtr, d},
		}
		for i := range specs {
			p, err := lib.MemAlloc(uint64(specs[i].n) * 4)
			if err != nil {
				for j := range specs {
					if *specs[j].ptr != 0 {
						lib.MemFree(*specs[j].ptr)
					}
				}
				return err
			}
			*specs[i].ptr = p
		}
		defer func() {
			for i := range specs {
				lib.MemFree(*specs[i].ptr)
			}
		}()

		if err := lib.MemcpyHtoD(dyPtr, driver.Bytes(dY)); err != nil {
			return err
		}
		if err := lib.MemcpyHtoD(xPtr, driver.Bytes(x)); err != nil {
			return err
		}
		if err := lib.MemcpyHtoD(wPtr, driver.Bytes(weight)); err != nil {
			return err
		}
		if err := lib.MemcpyHtoD(dsPtr, driver.Bytes(dscale)); err != nil { // zero-init
			return err
		}

		rowsU := uint32(rows)
		dU := uint32(d)
		epsF := float32(eps)
		const threads = uint32(256)
		blocks := (rowsU + threads - 1) / threads
		args := []unsafe.Pointer{
			unsafe.Pointer(&dyPtr), unsafe.Pointer(&xPtr), unsafe.Pointer(&wPtr),
			unsafe.Pointer(&dxPtr), unsafe.Pointer(&dsPtr),
			unsafe.Pointer(&rowsU), unsafe.Pointer(&dU), unsafe.Pointer(&epsF),
		}
		if err := lib.LaunchKernel(fn,
			driver.Dim3{X: blocks, Y: 1, Z: 1},
			driver.Dim3{X: threads, Y: 1, Z: 1},
			0, state.Stream, args); err != nil {
			return err
		}
		runtime.KeepAlive(dyPtr)
		runtime.KeepAlive(xPtr)
		runtime.KeepAlive(wPtr)
		runtime.KeepAlive(dxPtr)
		runtime.KeepAlive(dsPtr)

		if err := lib.StreamSynchronize(state.Stream); err != nil {
			return err
		}
		if err := lib.MemcpyDtoH(driver.Bytes(dx), dxPtr); err != nil {
			return err
		}
		return lib.MemcpyDtoH(driver.Bytes(dscale), dsPtr)
	})
	if err != nil {
		return nil, nil, err
	}
	return dx, dscale, nil
}
