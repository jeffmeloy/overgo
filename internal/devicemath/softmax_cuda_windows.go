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

// SoftmaxBackward computes the VJP of a row-wise softmax. Given the softmax
// output p[rows,d] and its cotangent dp, it returns the gradient w.r.t. the
// softmax input, ds_i = p_i*(dp_i - sum_j p_j*dp_j). Uses the
// softmax_backward_f32 kernel (one thread per row). This is the softmax half of
// attention backward.
func SoftmaxBackward(worker *device.Worker, p, dp []float32, rows, d int) ([]float32, error) {
	if rows <= 0 || d <= 0 || len(p) != rows*d || len(dp) != rows*d {
		return nil, fmt.Errorf("SoftmaxBackward: shape mismatch (rows=%d d=%d p=%d dp=%d)", rows, d, len(p), len(dp))
	}
	if uint64(rows) > math.MaxUint32 || uint64(d) > math.MaxUint32 {
		return nil, fmt.Errorf("SoftmaxBackward: dimensions exceed a 32-bit count")
	}
	ds := make([]float32, rows*d)
	err := worker.Do(context.Background(), func(state *device.State) error {
		lib := state.Driver
		module, err := lib.ModuleLoadData(kernel.OpsF32PTX)
		if err != nil {
			return err
		}
		defer lib.ModuleUnload(module)
		fn, err := lib.ModuleFunction(module, "softmax_backward_f32")
		if err != nil {
			return err
		}

		var pPtr, dpPtr, dsPtr driver.DevicePtr
		bufs := []*driver.DevicePtr{&pPtr, &dpPtr, &dsPtr}
		for _, b := range bufs {
			ptr, err := lib.MemAlloc(uint64(rows*d) * 4)
			if err != nil {
				for _, q := range bufs {
					if *q != 0 {
						lib.MemFree(*q)
					}
				}
				return err
			}
			*b = ptr
		}
		defer func() {
			for _, b := range bufs {
				lib.MemFree(*b)
			}
		}()

		if err := lib.MemcpyHtoD(pPtr, driver.Bytes(p)); err != nil {
			return err
		}
		if err := lib.MemcpyHtoD(dpPtr, driver.Bytes(dp)); err != nil {
			return err
		}

		rowsU := uint32(rows)
		dU := uint32(d)
		const threads = uint32(256)
		blocks := (rowsU + threads - 1) / threads
		args := []unsafe.Pointer{
			unsafe.Pointer(&pPtr), unsafe.Pointer(&dpPtr), unsafe.Pointer(&dsPtr),
			unsafe.Pointer(&rowsU), unsafe.Pointer(&dU),
		}
		if err := lib.LaunchKernel(fn,
			driver.Dim3{X: blocks, Y: 1, Z: 1},
			driver.Dim3{X: threads, Y: 1, Z: 1},
			0, state.Stream, args); err != nil {
			return err
		}
		runtime.KeepAlive(pPtr)
		runtime.KeepAlive(dpPtr)
		runtime.KeepAlive(dsPtr)

		if err := lib.StreamSynchronize(state.Stream); err != nil {
			return err
		}
		return lib.MemcpyDtoH(driver.Bytes(ds), dsPtr)
	})
	if err != nil {
		return nil, err
	}
	return ds, nil
}
