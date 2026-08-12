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

// SiLUBackward computes grad_input = grad_output * silu'(x) elementwise on the
// GPU, where silu(x) = x*sigmoid(x) and silu'(x) = s*(1 + x*(1-s)), s =
// sigmoid(x). Uses the ops_f32 silu_backward_f32 kernel. This is the SiLU/SwiGLU
// activation half of the gated-MLP backward.
func SiLUBackward(worker *device.Worker, x, gradOutput []float32) ([]float32, error) {
	n := len(x)
	if n == 0 || len(gradOutput) != n {
		return nil, fmt.Errorf("SiLUBackward: length mismatch (x=%d dy=%d)", n, len(gradOutput))
	}
	if uint64(n) > math.MaxUint32 {
		return nil, fmt.Errorf("SiLUBackward: input too large for a 32-bit element count")
	}
	gradInput := make([]float32, n)
	err := worker.Do(context.Background(), func(state *device.State) error {
		lib := state.Driver
		module, err := lib.ModuleLoadData(kernel.OpsF32PTX)
		if err != nil {
			return err
		}
		defer lib.ModuleUnload(module)
		fn, err := lib.ModuleFunction(module, "silu_backward_f32")
		if err != nil {
			return err
		}

		var dyPtr, xPtr, dxPtr driver.DevicePtr
		bufs := []*driver.DevicePtr{&dyPtr, &xPtr, &dxPtr}
		for _, p := range bufs {
			ptr, err := lib.MemAlloc(uint64(n) * 4)
			if err != nil {
				for _, q := range bufs {
					if *q != 0 {
						lib.MemFree(*q)
					}
				}
				return err
			}
			*p = ptr
		}
		defer func() {
			for _, p := range bufs {
				lib.MemFree(*p)
			}
		}()

		if err := lib.MemcpyHtoD(dyPtr, driver.Bytes(gradOutput)); err != nil {
			return err
		}
		if err := lib.MemcpyHtoD(xPtr, driver.Bytes(x)); err != nil {
			return err
		}

		count := uint32(n)
		const threads = uint32(256)
		blocks := (count + threads - 1) / threads
		args := []unsafe.Pointer{
			unsafe.Pointer(&dyPtr), unsafe.Pointer(&xPtr), unsafe.Pointer(&dxPtr), unsafe.Pointer(&count),
		}
		if err := lib.LaunchKernel(fn,
			driver.Dim3{X: blocks, Y: 1, Z: 1},
			driver.Dim3{X: threads, Y: 1, Z: 1},
			0, state.Stream, args); err != nil {
			return err
		}
		runtime.KeepAlive(dyPtr)
		runtime.KeepAlive(xPtr)
		runtime.KeepAlive(dxPtr)

		if err := lib.StreamSynchronize(state.Stream); err != nil {
			return err
		}
		return lib.MemcpyDtoH(driver.Bytes(gradInput), dxPtr)
	})
	if err != nil {
		return nil, err
	}
	return gradInput, nil
}
