//go:build windows

// Package devicemath holds GPU (CUDA, cgo-free) counterparts to internal/hostmath
// -- the device backward operators the training loop composes. Each is gated
// against the host by a finite-difference grad-check, matching adaptive_new's
// operators_backward verification.
package devicemath

import (
	"context"
	"fmt"

	"overgo/internal/cuda/cublas"
	"overgo/internal/cuda/device"
	"overgo/internal/cuda/driver"
)

// LinearBackward computes the gradients of the row-major linear map
// Y = X·W (X[rows,in], W[in,out], Y[rows,out]) given the output cotangent
// dY[rows,out], on the GPU in fp32:
//
//	dX = dY·Wᵀ   [rows,in]
//	dW = Xᵀ·dY   [in,out]
//
// Both are cuBLAS GEMMs (no custom kernel). This is the fundamental device
// backward operator; linear/attention/MLP backward compose it.
func LinearBackward(worker *device.Worker, x, w, dY []float32, rows, in, out int) (dX, dW []float32, err error) {
	if rows <= 0 || in <= 0 || out <= 0 || len(x) != rows*in || len(w) != in*out || len(dY) != rows*out {
		return nil, nil, fmt.Errorf("LinearBackward: shape mismatch (rows=%d in=%d out=%d x=%d w=%d dY=%d)", rows, in, out, len(x), len(w), len(dY))
	}
	dX = make([]float32, rows*in)
	dW = make([]float32, in*out)
	err = worker.Do(context.Background(), func(state *device.State) error {
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

		var dxPtr, dwPtr, xPtr, wPtr, dyPtr driver.DevicePtr
		bufs := []struct {
			ptr  *driver.DevicePtr
			data []float32
		}{
			{&dxPtr, dX}, {&dwPtr, dW}, {&xPtr, x}, {&wPtr, w}, {&dyPtr, dY},
		}
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

		if err := lib.MemcpyHtoD(xPtr, driver.Bytes(x)); err != nil {
			return err
		}
		if err := lib.MemcpyHtoD(wPtr, driver.Bytes(w)); err != nil {
			return err
		}
		if err := lib.MemcpyHtoD(dyPtr, driver.Bytes(dY)); err != nil {
			return err
		}

		// dX = dY·Wᵀ : op(A)=dY[rows,out], op(B)=Wᵀ[out,in] -> [rows,in].
		if err := blas.RowMajorGEMMExF32(handle, false, true, int32(rows), int32(out), int32(in), dyPtr, wPtr, dxPtr); err != nil {
			return err
		}
		// dW = Xᵀ·dY : op(A)=Xᵀ[in,rows], op(B)=dY[rows,out] -> [in,out].
		if err := blas.RowMajorGEMMExF32(handle, true, false, int32(in), int32(rows), int32(out), xPtr, dyPtr, dwPtr); err != nil {
			return err
		}

		if err := lib.StreamSynchronize(state.Stream); err != nil {
			return err
		}
		if err := lib.MemcpyDtoH(driver.Bytes(dX), dxPtr); err != nil {
			return err
		}
		return lib.MemcpyDtoH(driver.Bytes(dW), dwPtr)
	})
	if err != nil {
		return nil, nil, err
	}
	return dX, dW, nil
}
