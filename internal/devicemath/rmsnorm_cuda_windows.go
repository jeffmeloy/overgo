//go:build windows

package devicemath

import (
	"fmt"
	"math"
	"unsafe"

	"overgo/internal/cuda/device"
	"overgo/internal/cuda/driver"
)

// RMSNormForward applies affine RMS normalization to x[rows,d]:
//
//	y = x * rsqrt(mean(x^2 over d) + eps) * weight
//
// matching hostmath.RMSNormInto within fp32 tolerance. It uses the
// weighted_rms_norm_f32 kernel, which is ONE BLOCK PER ROW (blockDim 256 matches
// its shared reduction), so launch1D is fed rows*256 elements to yield exactly
// `rows` blocks. This is the RMSNorm piece of the device forward.
func RMSNormForward(worker *device.Worker, x, weight []float32, rows, d int, eps float64) ([]float32, error) {
	if rows <= 0 || d <= 0 || len(x) != rows*d || len(weight) != d {
		return nil, fmt.Errorf("RMSNormForward: shape mismatch (rows=%d d=%d x=%d w=%d)", rows, d, len(x), len(weight))
	}
	if uint64(rows) > math.MaxUint32 || uint64(d) > math.MaxUint32 {
		return nil, fmt.Errorf("RMSNormForward: dimensions exceed a 32-bit count")
	}
	out := make([]float32, rows*d)
	err := withCUDA(worker, func(scope *cudaScope) error {
		fn, err := scope.function("weighted_rms_norm_f32")
		if err != nil {
			return err
		}
		var xPtr, wPtr, outPtr driver.DevicePtr
		if err := scope.fill([]deviceSlot{{&xPtr, x, 0}, {&wPtr, weight, 0}, {&outPtr, nil, rows * d}}); err != nil {
			return err
		}
		widthU, rowsU, weightRows, epsF := uint32(d), uint32(rows), uint32(1), float32(eps)
		if err := scope.launch1D(fn, uint32(rows)*deviceBlockThreads,
			unsafe.Pointer(&xPtr), unsafe.Pointer(&wPtr), unsafe.Pointer(&outPtr),
			unsafe.Pointer(&widthU), unsafe.Pointer(&rowsU), unsafe.Pointer(&weightRows), unsafe.Pointer(&epsF),
		); err != nil {
			return err
		}
		return scope.finish(cudaDownload{out, outPtr})
	})
	if err != nil {
		return nil, err
	}
	return out, nil
}

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

	err = withCUDA(worker, func(scope *cudaScope) error {
		fn, err := scope.function("rms_norm_backward_f32")
		if err != nil {
			return err
		}
		var dyPtr, xPtr, wPtr, dxPtr, dsPtr driver.DevicePtr
		if err := scope.fill([]deviceSlot{{&dyPtr, dY, 0}, {&xPtr, x, 0}, {&wPtr, weight, 0}, {&dxPtr, nil, rows * d}, {&dsPtr, dscale, 0}}); err != nil {
			return err
		}

		rowsU := uint32(rows)
		dU := uint32(d)
		epsF := float32(eps)
		if err := scope.launch1D(fn, rowsU,
			unsafe.Pointer(&dyPtr), unsafe.Pointer(&xPtr), unsafe.Pointer(&wPtr),
			unsafe.Pointer(&dxPtr), unsafe.Pointer(&dsPtr),
			unsafe.Pointer(&rowsU), unsafe.Pointer(&dU), unsafe.Pointer(&epsF),
		); err != nil {
			return err
		}
		return scope.finish(cudaDownload{dx, dxPtr}, cudaDownload{dscale, dsPtr})
	})
	if err != nil {
		return nil, nil, err
	}
	return dx, dscale, nil
}
