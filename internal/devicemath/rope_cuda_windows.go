//go:build windows

package devicemath

import (
	"context"
	"fmt"
	"runtime"
	"unsafe"

	"overgo/internal/cuda/device"
	"overgo/internal/cuda/driver"
	"overgo/internal/cuda/kernel"
)

// RoPEHalfForward applies split-half (HF-layout) rotary embedding to a
// [seq, nHeads*hd] tensor in place: each pair (i, i+hd/2) of every head row at
// position pos rotates by angle pos*invFreq[i]. invFreq has hd/2 entries. This
// is the forward whose VJP is RoPEHalfBackward; it feeds the device attention
// forward (q and k rotation).
func RoPEHalfForward(worker *device.Worker, x, invFreq []float32, seq, nHeads, hd int) ([]float32, error) {
	if seq <= 0 || nHeads <= 0 || hd <= 0 || hd%2 != 0 || len(x) != seq*nHeads*hd || len(invFreq) != hd/2 {
		return nil, fmt.Errorf("RoPEHalfForward: shape mismatch (seq=%d nHeads=%d hd=%d x=%d inv=%d)", seq, nHeads, hd, len(x), len(invFreq))
	}
	out := make([]float32, len(x))
	copy(out, x)
	err := worker.Do(context.Background(), func(state *device.State) error {
		lib := state.Driver
		module, err := lib.ModuleLoadData(kernel.OpsF32PTX)
		if err != nil {
			return err
		}
		defer lib.ModuleUnload(module)
		fn, err := lib.ModuleFunction(module, "rope_half_f32")
		if err != nil {
			return err
		}
		var xPtr, invPtr driver.DevicePtr
		bufs := []struct {
			ptr  *driver.DevicePtr
			data []float32
		}{{&xPtr, out}, {&invPtr, invFreq}}
		for i := range bufs {
			p, err := lib.MemAlloc(uint64(len(bufs[i].data)) * 4)
			if err != nil {
				for j := range bufs {
					if *bufs[j].ptr != 0 {
						lib.MemFree(*bufs[j].ptr)
					}
				}
				return err
			}
			*bufs[i].ptr = p
		}
		defer func() {
			for i := range bufs {
				lib.MemFree(*bufs[i].ptr)
			}
		}()
		if err := lib.MemcpyHtoD(xPtr, driver.Bytes(out)); err != nil {
			return err
		}
		if err := lib.MemcpyHtoD(invPtr, driver.Bytes(invFreq)); err != nil {
			return err
		}

		seqU, nhU, hdU := uint32(seq), uint32(nHeads), uint32(hd)
		total := uint32(seq * nHeads * (hd / 2))
		const threads = uint32(256)
		blocks := (total + threads - 1) / threads
		args := []unsafe.Pointer{
			unsafe.Pointer(&xPtr), unsafe.Pointer(&invPtr),
			unsafe.Pointer(&seqU), unsafe.Pointer(&nhU), unsafe.Pointer(&hdU),
		}
		if err := lib.LaunchKernel(fn,
			driver.Dim3{X: blocks, Y: 1, Z: 1},
			driver.Dim3{X: threads, Y: 1, Z: 1},
			0, state.Stream, args); err != nil {
			return err
		}
		runtime.KeepAlive(xPtr)
		runtime.KeepAlive(invPtr)
		if err := lib.StreamSynchronize(state.Stream); err != nil {
			return err
		}
		return lib.MemcpyDtoH(driver.Bytes(out), xPtr)
	})
	if err != nil {
		return nil, err
	}
	return out, nil
}

// RoPEHalfBackward is the VJP of split-half (HF-layout) rotary embedding applied
// to a [seq, nHeads*hd] tensor: it rotates the cotangent gradient by the negated
// angle pos*invFreq[i]. Since RoPE is linear (an orthogonal rotation), this
// depends only on the gradient, not the forward input. invFreq has hd/2 entries.
func RoPEHalfBackward(worker *device.Worker, gradOutput, invFreq []float32, seq, nHeads, hd int) ([]float32, error) {
	if seq <= 0 || nHeads <= 0 || hd <= 0 || hd%2 != 0 || len(gradOutput) != seq*nHeads*hd || len(invFreq) != hd/2 {
		return nil, fmt.Errorf("RoPEHalfBackward: shape mismatch (seq=%d nHeads=%d hd=%d g=%d inv=%d)", seq, nHeads, hd, len(gradOutput), len(invFreq))
	}
	dx := make([]float32, len(gradOutput))
	copy(dx, gradOutput)
	err := worker.Do(context.Background(), func(state *device.State) error {
		lib := state.Driver
		module, err := lib.ModuleLoadData(kernel.OpsF32PTX)
		if err != nil {
			return err
		}
		defer lib.ModuleUnload(module)
		fn, err := lib.ModuleFunction(module, "rope_half_backward_f32")
		if err != nil {
			return err
		}
		var dxPtr, invPtr driver.DevicePtr
		bufs := []struct {
			ptr  *driver.DevicePtr
			data []float32
		}{{&dxPtr, dx}, {&invPtr, invFreq}}
		for i := range bufs {
			p, err := lib.MemAlloc(uint64(len(bufs[i].data)) * 4)
			if err != nil {
				for j := range bufs {
					if *bufs[j].ptr != 0 {
						lib.MemFree(*bufs[j].ptr)
					}
				}
				return err
			}
			*bufs[i].ptr = p
		}
		defer func() {
			for i := range bufs {
				lib.MemFree(*bufs[i].ptr)
			}
		}()
		if err := lib.MemcpyHtoD(dxPtr, driver.Bytes(dx)); err != nil {
			return err
		}
		if err := lib.MemcpyHtoD(invPtr, driver.Bytes(invFreq)); err != nil {
			return err
		}

		seqU, nhU, hdU := uint32(seq), uint32(nHeads), uint32(hd)
		total := uint32(seq * nHeads * (hd / 2))
		const threads = uint32(256)
		blocks := (total + threads - 1) / threads
		args := []unsafe.Pointer{
			unsafe.Pointer(&dxPtr), unsafe.Pointer(&invPtr),
			unsafe.Pointer(&seqU), unsafe.Pointer(&nhU), unsafe.Pointer(&hdU),
		}
		if err := lib.LaunchKernel(fn,
			driver.Dim3{X: blocks, Y: 1, Z: 1},
			driver.Dim3{X: threads, Y: 1, Z: 1},
			0, state.Stream, args); err != nil {
			return err
		}
		runtime.KeepAlive(dxPtr)
		runtime.KeepAlive(invPtr)
		if err := lib.StreamSynchronize(state.Stream); err != nil {
			return err
		}
		return lib.MemcpyDtoH(driver.Bytes(dx), dxPtr)
	})
	if err != nil {
		return nil, err
	}
	return dx, nil
}
