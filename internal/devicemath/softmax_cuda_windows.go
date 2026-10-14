//go:build windows

package devicemath

import (
	"fmt"
	"math"
	"unsafe"

	"overgo/internal/cuda/device"
	"overgo/internal/cuda/driver"
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
	err := withCUDA(worker, func(scope *cudaScope) error {
		fn, err := scope.function("softmax_backward_f32")
		if err != nil {
			return err
		}
		var pPtr, dpPtr, dsPtr driver.DevicePtr
		if err := scope.fill([]deviceSlot{{&pPtr, p, 0}, {&dpPtr, dp, 0}, {&dsPtr, nil, rows * d}}); err != nil {
			return err
		}

		rowsU := uint32(rows)
		dU := uint32(d)
		if err := scope.launch1D(fn, rowsU,
			unsafe.Pointer(&pPtr), unsafe.Pointer(&dpPtr), unsafe.Pointer(&dsPtr),
			unsafe.Pointer(&rowsU), unsafe.Pointer(&dU),
		); err != nil {
			return err
		}
		return scope.finish(cudaDownload{ds, dsPtr})
	})
	if err != nil {
		return nil, err
	}
	return ds, nil
}
