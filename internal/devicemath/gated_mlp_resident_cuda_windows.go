//go:build windows

package devicemath

import (
	"context"
	"fmt"
	"runtime"
	"unsafe"

	"overgo/internal/cuda/cublas"
	"overgo/internal/cuda/device"
	"overgo/internal/cuda/driver"
	"overgo/internal/cuda/kernel"
)

// GatedMLPBackwardTResident is the resident counterpart to GatedMLPBackwardT: it
// runs the entire densecausal-convention SwiGLU MLP backward inside ONE
// worker.Do, with a single cuBLAS handle and ops_f32 module, uploading the
// inputs once and keeping every intermediate gradient (dh, da, du, dg, dXgate,
// dXup) resident on the device -- no per-op host round-trips. Only dX and the
// three weight grads come back. Same math as GatedMLPBackwardT; this addresses
// SQA finding 2/3 (residency) for the MLP block.
func GatedMLPBackwardTResident(worker *device.Worker, x, wGate, wUp, wDown, g, a, u, h, dY []float32, rows, d, inter int) (GatedMLPGrads, error) {
	if rows <= 0 || d <= 0 || inter <= 0 ||
		len(x) != rows*d || len(wGate) != inter*d || len(wUp) != inter*d || len(wDown) != d*inter ||
		len(g) != rows*inter || len(a) != rows*inter || len(u) != rows*inter || len(h) != rows*inter || len(dY) != rows*d {
		return GatedMLPGrads{}, fmt.Errorf("GatedMLPBackwardTResident: shape mismatch (rows=%d d=%d inter=%d)", rows, d, inter)
	}
	dX := make([]float32, rows*d)
	dWGate := make([]float32, inter*d)
	dWUp := make([]float32, inter*d)
	dWDown := make([]float32, d*inter)

	err := worker.Do(context.Background(), func(state *device.State) error {
		lib := state.Driver
		blas, err := cublas.Open()
		if err != nil {
			return err
		}
		defer blas.Close()
		handle, err := blas.Create()
		if err != nil {
			return err
		}
		defer blas.Destroy(handle)
		if err := blas.SetStream(handle, state.Stream); err != nil {
			return err
		}
		module, err := lib.ModuleLoadData(kernel.OpsF32PTX)
		if err != nil {
			return err
		}
		defer lib.ModuleUnload(module)
		mulFn, err := lib.ModuleFunction(module, "multiply_f32")
		if err != nil {
			return err
		}
		addFn, err := lib.ModuleFunction(module, "add_f32")
		if err != nil {
			return err
		}
		siluFn, err := lib.ModuleFunction(module, "silu_backward_f32")
		if err != nil {
			return err
		}

		var frees []driver.DevicePtr
		defer func() {
			for _, p := range frees {
				lib.MemFree(p)
			}
		}()
		alloc := func(n int) (driver.DevicePtr, error) {
			p, err := lib.MemAlloc(uint64(n) * 4)
			if err != nil {
				return 0, err
			}
			frees = append(frees, p)
			return p, nil
		}
		upload := func(data []float32) (driver.DevicePtr, error) {
			p, err := alloc(len(data))
			if err != nil {
				return 0, err
			}
			return p, lib.MemcpyHtoD(p, driver.Bytes(data))
		}

		// Resident inputs (uploaded once).
		hnP, err := upload(x)
		if err != nil {
			return err
		}
		wGateP, err := upload(wGate)
		if err != nil {
			return err
		}
		wUpP, err := upload(wUp)
		if err != nil {
			return err
		}
		wDownP, err := upload(wDown)
		if err != nil {
			return err
		}
		gP, err := upload(g)
		if err != nil {
			return err
		}
		aP, err := upload(a)
		if err != nil {
			return err
		}
		uP, err := upload(u)
		if err != nil {
			return err
		}
		hP, err := upload(h)
		if err != nil {
			return err
		}
		dyP, err := upload(dY)
		if err != nil {
			return err
		}
		// Resident intermediates + outputs.
		dhP, err := alloc(rows * inter)
		if err != nil {
			return err
		}
		daP, err := alloc(rows * inter)
		if err != nil {
			return err
		}
		duP, err := alloc(rows * inter)
		if err != nil {
			return err
		}
		dgP, err := alloc(rows * inter)
		if err != nil {
			return err
		}
		dxGateP, err := alloc(rows * d)
		if err != nil {
			return err
		}
		dxUpP, err := alloc(rows * d)
		if err != nil {
			return err
		}
		dxP, err := alloc(rows * d)
		if err != nil {
			return err
		}
		dwGateP, err := alloc(inter * d)
		if err != nil {
			return err
		}
		dwUpP, err := alloc(inter * d)
		if err != nil {
			return err
		}
		dwDownP, err := alloc(d * inter)
		if err != nil {
			return err
		}

		elemwise := func(fn driver.Function, args []unsafe.Pointer, n int) error {
			const threads = uint32(256)
			blocks := (uint32(n) + threads - 1) / threads
			return lib.LaunchKernel(fn, driver.Dim3{X: blocks, Y: 1, Z: 1}, driver.Dim3{X: threads, Y: 1, Z: 1}, 0, state.Stream, args)
		}
		mul := func(x, y, out driver.DevicePtr, n int) error {
			c := uint32(n)
			err := elemwise(mulFn, []unsafe.Pointer{unsafe.Pointer(&x), unsafe.Pointer(&y), unsafe.Pointer(&out), unsafe.Pointer(&c)}, n)
			runtime.KeepAlive(x)
			runtime.KeepAlive(y)
			runtime.KeepAlive(out)
			return err
		}
		add := func(x, y, out driver.DevicePtr, n int) error {
			c := uint32(n)
			err := elemwise(addFn, []unsafe.Pointer{unsafe.Pointer(&x), unsafe.Pointer(&y), unsafe.Pointer(&out), unsafe.Pointer(&c)}, n)
			runtime.KeepAlive(x)
			runtime.KeepAlive(y)
			runtime.KeepAlive(out)
			return err
		}
		// silu_backward_f32 arg order is (dy, x, out): out = dy * silu'(x).
		silu := func(x, dy, out driver.DevicePtr, n int) error {
			c := uint32(n)
			err := elemwise(siluFn, []unsafe.Pointer{unsafe.Pointer(&dy), unsafe.Pointer(&x), unsafe.Pointer(&out), unsafe.Pointer(&c)}, n)
			runtime.KeepAlive(x)
			runtime.KeepAlive(dy)
			runtime.KeepAlive(out)
			return err
		}

		// Y = h·Wdownᵀ  ->  dh = dY·Wdown ; dWdown = dYᵀ·h.
		if err := blas.RowMajorGEMMExF32(handle, false, false, int32(rows), int32(d), int32(inter), dyP, wDownP, dhP); err != nil {
			return err
		}
		if err := blas.RowMajorGEMMExF32(handle, true, false, int32(d), int32(rows), int32(inter), dyP, hP, dwDownP); err != nil {
			return err
		}
		// da = dh⊙u ; du = dh⊙a ; dg = silu'(g)⊙da.
		if err := mul(dhP, uP, daP, rows*inter); err != nil {
			return err
		}
		if err := mul(dhP, aP, duP, rows*inter); err != nil {
			return err
		}
		if err := silu(gP, daP, dgP, rows*inter); err != nil {
			return err
		}
		// g = X·Wgateᵀ ->  dXgate = dg·Wgate ; dWgate = dgᵀ·X.
		if err := blas.RowMajorGEMMExF32(handle, false, false, int32(rows), int32(inter), int32(d), dgP, wGateP, dxGateP); err != nil {
			return err
		}
		if err := blas.RowMajorGEMMExF32(handle, true, false, int32(inter), int32(rows), int32(d), dgP, hnP, dwGateP); err != nil {
			return err
		}
		// u = X·Wupᵀ ->  dXup = du·Wup ; dWup = duᵀ·X.
		if err := blas.RowMajorGEMMExF32(handle, false, false, int32(rows), int32(inter), int32(d), duP, wUpP, dxUpP); err != nil {
			return err
		}
		if err := blas.RowMajorGEMMExF32(handle, true, false, int32(inter), int32(rows), int32(d), duP, hnP, dwUpP); err != nil {
			return err
		}
		// dX = dXgate + dXup.
		if err := add(dxGateP, dxUpP, dxP, rows*d); err != nil {
			return err
		}

		if err := lib.StreamSynchronize(state.Stream); err != nil {
			return err
		}
		if err := lib.MemcpyDtoH(driver.Bytes(dX), dxP); err != nil {
			return err
		}
		if err := lib.MemcpyDtoH(driver.Bytes(dWGate), dwGateP); err != nil {
			return err
		}
		if err := lib.MemcpyDtoH(driver.Bytes(dWUp), dwUpP); err != nil {
			return err
		}
		return lib.MemcpyDtoH(driver.Bytes(dWDown), dwDownP)
	})
	if err != nil {
		return GatedMLPGrads{}, err
	}
	return GatedMLPGrads{DX: dX, DWGate: dWGate, DWUp: dWUp, DWDown: dWDown}, nil
}
