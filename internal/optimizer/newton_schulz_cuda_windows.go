//go:build windows

package optimizer

import (
	"context"
	"fmt"
	"math"
	"runtime"
	"unsafe"

	"overgo/internal/cuda/cublas"
	"overgo/internal/cuda/device"
	"overgo/internal/cuda/driver"
	"overgo/internal/cuda/kernel"
)

// deviceNewtonSchulz runs the Muon Newton-Schulz iteration on the GPU in fp32,
// reproducing host newtonSchulz (fp64 oracle) within tolerance. Every matmul is
// a cuBLAS GEMM; the c0*I term folds into the final matmul
// (X*(c0*I + R) = c0*X + X*R), so no diagonal kernel is needed -- only cuBLAS
// plus the existing ops_f32 scale/add. The Frobenius norm reuses a GEMM
// dot-product. Buffers/handles are per-call (correctness-first; a resident
// device optimizer will cache them).
func deviceNewtonSchulz(worker *device.Worker, input []float32, rows, cols int) ([]float32, error) {
	if rows <= 0 || cols <= 0 || len(input) != rows*cols {
		return nil, fmt.Errorf("deviceNewtonSchulz: len %d != %d*%d", len(input), rows, cols)
	}
	if uint64(len(input)) > math.MaxUint32 {
		return nil, fmt.Errorf("deviceNewtonSchulz: matrix too large for a 32-bit element count")
	}
	result := make([]float32, len(input))
	copy(result, input)
	err := worker.Do(context.Background(), func(state *device.State) error {
		return runDeviceNewtonSchulz(state, result, rows, cols)
	})
	if err != nil {
		return nil, err
	}
	return result, nil
}

func runDeviceNewtonSchulz(state *device.State, data []float32, rows, cols int) error {
	lib := state.Driver
	n := rows * cols
	dim := cols
	tall := rows >= cols
	if !tall {
		dim = rows
	}
	gramN := dim * dim

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
	scaleFn, err := lib.ModuleFunction(module, "scale_f32")
	if err != nil {
		return err
	}
	addFn, err := lib.ModuleFunction(module, "add_f32")
	if err != nil {
		return err
	}

	alloc := func(count int) (driver.DevicePtr, error) { return lib.MemAlloc(uint64(count) * 4) }
	dX, err := alloc(n)
	if err != nil {
		return err
	}
	defer lib.MemFree(dX)
	dXNew, err := alloc(n)
	if err != nil {
		return err
	}
	defer lib.MemFree(dXNew)
	dCX, err := alloc(n)
	if err != nil {
		return err
	}
	defer lib.MemFree(dCX)
	dGram, err := alloc(gramN)
	if err != nil {
		return err
	}
	defer lib.MemFree(dGram)
	dSq, err := alloc(gramN)
	if err != nil {
		return err
	}
	defer lib.MemFree(dSq)
	dRest, err := alloc(gramN)
	if err != nil {
		return err
	}
	defer lib.MemFree(dRest)
	dNorm, err := alloc(1)
	if err != nil {
		return err
	}
	defer lib.MemFree(dNorm)

	if err := lib.MemcpyHtoD(dX, driver.Bytes(data)); err != nil {
		return err
	}

	// scale: out = in * s (count elements). add: out = a + b.
	launchElementwise := func(fn driver.Function, count int, args []unsafe.Pointer) error {
		const threads = uint32(256)
		total := uint32(count)
		blocks := (total + threads - 1) / threads
		return lib.LaunchKernel(fn,
			driver.Dim3{X: blocks, Y: 1, Z: 1},
			driver.Dim3{X: threads, Y: 1, Z: 1},
			0, state.Stream, args)
	}
	scale := func(in, out driver.DevicePtr, s float32, count int) error {
		c := uint32(count)
		err := launchElementwise(scaleFn, count, []unsafe.Pointer{
			unsafe.Pointer(&in), unsafe.Pointer(&out), unsafe.Pointer(&s), unsafe.Pointer(&c),
		})
		runtime.KeepAlive(in)
		runtime.KeepAlive(out)
		return err
	}
	add := func(a, b, out driver.DevicePtr, count int) error {
		c := uint32(count)
		err := launchElementwise(addFn, count, []unsafe.Pointer{
			unsafe.Pointer(&a), unsafe.Pointer(&b), unsafe.Pointer(&out), unsafe.Pointer(&c),
		})
		runtime.KeepAlive(a)
		runtime.KeepAlive(b)
		runtime.KeepAlive(out)
		return err
	}

	// Frobenius normalize: ||X||^2 = X_flat[1,n] . X_flat[n,1] via one GEMM.
	if err := blas.RowMajorGEMMF32(handle, 1, int32(n), 1, dX, dX, dNorm); err != nil {
		return err
	}
	if err := lib.StreamSynchronize(state.Stream); err != nil {
		return err
	}
	normSquared := make([]float32, 1)
	if err := lib.MemcpyDtoH(driver.Bytes(normSquared), dNorm); err != nil {
		return err
	}
	norm := math.Sqrt(float64(normSquared[0]))
	if norm < newtonSchulzFrobeniusGuard {
		return nil // degenerate matrix: leave `data` unchanged, matching host
	}
	if err := scale(dX, dX, float32(1/norm), n); err != nil {
		return err
	}

	iterations := newtonSchulzStage1Iterations + newtonSchulzStage2Iterations
	for iteration := 0; iteration < iterations; iteration++ {
		coefficients := newtonSchulzStage1
		if iteration >= newtonSchulzStage1Iterations {
			coefficients = newtonSchulzStage2
		}
		c0 := float32(coefficients[0])
		c1 := float32(coefficients[1])
		c2 := float32(coefficients[2])

		if tall {
			// gram = XᵀX  [cols x cols]
			if err := blas.RowMajorGEMMExF32(handle, true, false, int32(cols), int32(rows), int32(cols), dX, dX, dGram); err != nil {
				return err
			}
		} else {
			// gram = XXᵀ  [rows x rows]
			if err := blas.RowMajorGEMMExF32(handle, false, true, int32(rows), int32(cols), int32(rows), dX, dX, dGram); err != nil {
				return err
			}
		}
		// sq = gram·gram  [dim x dim]
		if err := blas.RowMajorGEMMF32(handle, int32(dim), int32(dim), int32(dim), dGram, dGram, dSq); err != nil {
			return err
		}
		// rest = c1*gram + c2*sq
		if err := scale(dGram, dRest, c1, gramN); err != nil {
			return err
		}
		if err := scale(dSq, dSq, c2, gramN); err != nil {
			return err
		}
		if err := add(dRest, dSq, dRest, gramN); err != nil {
			return err
		}
		// Xnew = op·rest-product + c0*X  (c0*I folded into the matmul).
		if tall {
			// Xnew = X·rest  [rows x cols]
			if err := blas.RowMajorGEMMF32(handle, int32(rows), int32(cols), int32(cols), dX, dRest, dXNew); err != nil {
				return err
			}
		} else {
			// Xnew = rest·X  [rows x cols]
			if err := blas.RowMajorGEMMF32(handle, int32(rows), int32(rows), int32(cols), dRest, dX, dXNew); err != nil {
				return err
			}
		}
		if err := scale(dX, dCX, c0, n); err != nil {
			return err
		}
		if err := add(dXNew, dCX, dXNew, n); err != nil {
			return err
		}
		dX, dXNew = dXNew, dX
	}

	if err := lib.StreamSynchronize(state.Stream); err != nil {
		return err
	}
	return lib.MemcpyDtoH(driver.Bytes(data), dX)
}
