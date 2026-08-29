//go:build windows

package optimizer

import (
	"context"
	"fmt"
	"math"
	"runtime"
	"unsafe"

	"overgo/internal/checked"
	"overgo/internal/cuda/cublas"
	"overgo/internal/cuda/device"
	"overgo/internal/cuda/driver"
	"overgo/internal/cuda/kernel"
)

const deviceF32Bytes = uint64(4)

// deviceNewtonSchulz runs the Muon Newton-Schulz iteration on the GPU in fp32,
// reproducing host newtonSchulz (fp64 oracle) within tolerance. Every matmul is
// a cuBLAS GEMM; the c0*I term folds into the final matmul
// (X*(c0*I + R) = c0*X + X*R), so no diagonal kernel is needed -- only cuBLAS
// plus the existing ops_f32 scale/add. The Frobenius norm reuses a GEMM
// dot-product. Buffers/handles are per-call (correctness-first; the device Muon
// step reuses the same primitives on resident state).
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
		ops, err := newDeviceOps(state)
		if err != nil {
			return err
		}
		defer ops.close()

		n := rows * cols
		gramN := gramDim(rows, cols)
		gramN *= gramN
		buffers, err := ops.allocBuffers(n, gramN)
		if err != nil {
			return err
		}
		defer buffers.free(ops.lib)

		if err := ops.lib.MemcpyHtoD(buffers.dX, driver.Bytes(result)); err != nil {
			return err
		}
		final, err := ops.newtonSchulz(buffers, rows, cols)
		if err != nil {
			return err
		}
		if err := ops.lib.StreamSynchronize(ops.stream); err != nil {
			return err
		}
		return ops.lib.MemcpyDtoH(driver.Bytes(result), final)
	})
	if err != nil {
		return nil, err
	}
	return result, nil
}

// gramDim returns min(rows, cols), the side of the gram matrix (XᵀX for tall,
// XXᵀ for wide).
func gramDim(rows, cols int) int {
	if rows < cols {
		return rows
	}
	return cols
}

// deviceOps: shared Muon CUDA handles, kernels, and buffer policy.
type deviceOps struct {
	lib     *driver.Library
	blas    *cublas.Library
	handle  cublas.Handle
	tf32    cublas.Handle
	module  driver.Module
	scaleFn driver.Function
	addFn   driver.Function
	stream  driver.Stream
}

func newDeviceOps(state *device.State) (*deviceOps, error) {
	lib := state.Driver
	blas, err := cublas.Open()
	if err != nil {
		return nil, err
	}
	handle, err := blas.Create()
	if err != nil {
		blas.Close()
		return nil, err
	}
	if err := blas.SetStream(handle, state.Stream); err != nil {
		blas.Destroy(handle)
		blas.Close()
		return nil, err
	}
	tf32, err := blas.Create()
	if err != nil {
		blas.Destroy(handle)
		blas.Close()
		return nil, err
	}
	if err := blas.SetStream(tf32, state.Stream); err != nil {
		blas.Destroy(tf32)
		blas.Destroy(handle)
		blas.Close()
		return nil, err
	}
	if err := blas.SetMathMode(tf32, cublas.MathTF32TensorOp); err != nil {
		blas.Destroy(tf32)
		blas.Destroy(handle)
		blas.Close()
		return nil, err
	}
	module, err := lib.ModuleLoadData(kernel.OpsF32PTX)
	if err != nil {
		blas.Destroy(tf32)
		blas.Destroy(handle)
		blas.Close()
		return nil, err
	}
	scaleFn, err := lib.ModuleFunction(module, "scale_f32")
	if err != nil {
		lib.ModuleUnload(module)
		blas.Destroy(tf32)
		blas.Destroy(handle)
		blas.Close()
		return nil, err
	}
	addFn, err := lib.ModuleFunction(module, "add_f32")
	if err != nil {
		lib.ModuleUnload(module)
		blas.Destroy(tf32)
		blas.Destroy(handle)
		blas.Close()
		return nil, err
	}
	return &deviceOps{lib: lib, blas: blas, handle: handle, tf32: tf32, module: module, scaleFn: scaleFn, addFn: addFn, stream: state.Stream}, nil
}

func (o *deviceOps) close() {
	o.lib.ModuleUnload(o.module)
	o.blas.Destroy(o.tf32)
	o.blas.Destroy(o.handle)
	o.blas.Close()
}

func (o *deviceOps) allocF32(count int) (driver.DevicePtr, error) {
	if count <= 0 {
		return 0, fmt.Errorf("device optimizer: invalid f32 allocation count %d", count)
	}
	bytes, valid := checked.Bytes(uint64(count), deviceF32Bytes)
	if !valid {
		return 0, fmt.Errorf("device optimizer: invalid f32 allocation count %d", count)
	}
	return o.lib.MemAlloc(bytes)
}

func (o *deviceOps) allocF32Set(counts ...int) ([]driver.DevicePtr, error) {
	pointers := make([]driver.DevicePtr, len(counts))
	for index, count := range counts {
		pointer, err := o.allocF32(count)
		if err != nil {
			freeDevicePointers(o.lib, pointers...)
			return nil, err
		}
		pointers[index] = pointer
	}
	return pointers, nil
}

func (o *deviceOps) uploadF32Set(values ...[]float32) ([]driver.DevicePtr, error) {
	counts := make([]int, len(values))
	for index := range values {
		counts[index] = len(values[index])
	}
	pointers, err := o.allocF32Set(counts...)
	if err != nil {
		return nil, err
	}
	for index := range values {
		if err := o.lib.MemcpyHtoD(pointers[index], driver.Bytes(values[index])); err != nil {
			freeDevicePointers(o.lib, pointers...)
			return nil, err
		}
	}
	return pointers, nil
}

func freeDevicePointers(library *driver.Library, pointers ...driver.DevicePtr) {
	for _, pointer := range pointers {
		if pointer != 0 {
			library.MemFree(pointer)
		}
	}
}

func offsetF32(pointer driver.DevicePtr, elements int) driver.DevicePtr {
	return pointer + driver.DevicePtr(uint64(elements)*deviceF32Bytes)
}

func (o *deviceOps) launch(fn driver.Function, count int, args []unsafe.Pointer) error {
	const threads = uint32(256)
	total := uint32(count)
	blocks := (total + threads - 1) / threads
	return o.lib.LaunchKernel(fn,
		driver.Dim3{X: blocks, Y: 1, Z: 1},
		driver.Dim3{X: threads, Y: 1, Z: 1},
		0, o.stream, args)
}

// scale: out = in * s (count elements).
func (o *deviceOps) scale(in, out driver.DevicePtr, s float32, count int) error {
	c := uint32(count)
	err := o.launch(o.scaleFn, count, []unsafe.Pointer{
		unsafe.Pointer(&in), unsafe.Pointer(&out), unsafe.Pointer(&s), unsafe.Pointer(&c),
	})
	runtime.KeepAlive(in)
	runtime.KeepAlive(out)
	return err
}

// add: out = a + b (count elements).
func (o *deviceOps) add(a, b, out driver.DevicePtr, count int) error {
	c := uint32(count)
	err := o.launch(o.addFn, count, []unsafe.Pointer{
		unsafe.Pointer(&a), unsafe.Pointer(&b), unsafe.Pointer(&out), unsafe.Pointer(&c),
	})
	runtime.KeepAlive(a)
	runtime.KeepAlive(b)
	runtime.KeepAlive(out)
	return err
}

// nsBuffers: Newton-Schulz matrix, Gram, and norm scratch.
type nsBuffers struct {
	dX, dXNew, dCX    driver.DevicePtr // size n = rows*cols
	dGram, dSq, dRest driver.DevicePtr // size gramN = dim*dim
	dNorm             driver.DevicePtr // size 1
}

func (o *deviceOps) allocBuffers(n, gramN int) (nsBuffers, error) {
	pointers, err := o.allocF32Set(n, n, n, gramN, gramN, gramN, 1)
	if err != nil {
		return nsBuffers{}, err
	}
	return nsBuffers{
		dX: pointers[0], dXNew: pointers[1], dCX: pointers[2],
		dGram: pointers[3], dSq: pointers[4], dRest: pointers[5], dNorm: pointers[6],
	}, nil
}

func (b nsBuffers) free(lib *driver.Library) {
	freeDevicePointers(lib, b.dX, b.dXNew, b.dCX, b.dGram, b.dSq, b.dRest, b.dNorm)
}

const residentNewtonSchulzStage1Iterations = 4

// newtonSchulz matches the host 8+2 FP32 oracle.
func (o *deviceOps) newtonSchulz(b nsBuffers, rows, cols int) (driver.DevicePtr, error) {
	return o.runNewtonSchulz(b, rows, cols, o.handle, newtonSchulzStage1Iterations)
}

// newtonSchulzResident runs the measured 4+2 TF32 training schedule.
func (o *deviceOps) newtonSchulzResident(b nsBuffers, rows, cols int) (driver.DevicePtr, error) {
	return o.runNewtonSchulz(b, rows, cols, o.tf32, residentNewtonSchulzStage1Iterations)
}

func (o *deviceOps) runNewtonSchulz(
	b nsBuffers,
	rows, cols int,
	handle cublas.Handle,
	stage1Iterations int,
) (driver.DevicePtr, error) {
	n := rows * cols
	dim := gramDim(rows, cols)
	gramN := dim * dim
	tall := rows >= cols
	dX, dXNew := b.dX, b.dXNew

	// Frobenius normalize: ||X||^2 = X_flat[1,n] . X_flat[n,1] via one GEMM.
	if err := o.blas.RowMajorGEMMF32(o.handle, 1, int32(n), 1, dX, dX, b.dNorm); err != nil {
		return 0, err
	}
	if err := o.lib.StreamSynchronize(o.stream); err != nil {
		return 0, err
	}
	normSquared := make([]float32, 1)
	if err := o.lib.MemcpyDtoH(driver.Bytes(normSquared), b.dNorm); err != nil {
		return 0, err
	}
	norm := math.Sqrt(float64(normSquared[0]))
	if norm < newtonSchulzFrobeniusGuard {
		return dX, nil
	}
	if err := o.scale(dX, dX, float32(1/norm), n); err != nil {
		return 0, err
	}

	iterations := stage1Iterations + newtonSchulzStage2Iterations
	for iteration := range iterations {
		coefficients := newtonSchulzStage1
		if iteration >= stage1Iterations {
			coefficients = newtonSchulzStage2
		}
		c0 := float32(coefficients[0])
		c1 := float32(coefficients[1])
		c2 := float32(coefficients[2])

		if tall {
			// gram = XᵀX  [cols x cols]
			if err := o.blas.RowMajorGEMMExF32(handle, true, false, int32(cols), int32(rows), int32(cols), dX, dX, b.dGram); err != nil {
				return 0, err
			}
		} else {
			// gram = XXᵀ  [rows x rows]
			if err := o.blas.RowMajorGEMMExF32(handle, false, true, int32(rows), int32(cols), int32(rows), dX, dX, b.dGram); err != nil {
				return 0, err
			}
		}
		// sq = gram·gram  [dim x dim]
		if err := o.blas.RowMajorGEMMF32(handle, int32(dim), int32(dim), int32(dim), b.dGram, b.dGram, b.dSq); err != nil {
			return 0, err
		}
		// rest = c1*gram + c2*sq
		if err := o.scale(b.dGram, b.dRest, c1, gramN); err != nil {
			return 0, err
		}
		if err := o.scale(b.dSq, b.dSq, c2, gramN); err != nil {
			return 0, err
		}
		if err := o.add(b.dRest, b.dSq, b.dRest, gramN); err != nil {
			return 0, err
		}
		// Xnew = op·rest-product + c0*X  (c0*I folded into the matmul).
		if tall {
			// Xnew = X·rest  [rows x cols]
			if err := o.blas.RowMajorGEMMF32(handle, int32(rows), int32(cols), int32(cols), dX, b.dRest, dXNew); err != nil {
				return 0, err
			}
		} else {
			// Xnew = rest·X  [rows x cols]
			if err := o.blas.RowMajorGEMMF32(handle, int32(rows), int32(rows), int32(cols), b.dRest, dX, dXNew); err != nil {
				return 0, err
			}
		}
		if err := o.scale(dX, b.dCX, c0, n); err != nil {
			return 0, err
		}
		if err := o.add(dXNew, b.dCX, dXNew, n); err != nil {
			return 0, err
		}
		dX, dXNew = dXNew, dX
	}
	return dX, nil
}
