//go:build windows

package devicemath

import (
	"fmt"
	"unsafe"

	"overgo/internal/cuda/device"
)

// L2NormBackwardDevice is the cgo-free CUDA port of hostmath.L2NormBackward: the
// VJP of the per-row L2 normalization y = x / max(sqrt(sum(x^2)), eps). With
// inv = 1/max(sqrt(sum(x^2)),eps): unclamped (norm>eps)
// dX = inv*dY - inv^3 * x * (dY·x); clamped (norm<=eps) dX = inv*dY. Launches
// l2_norm_backward_f32 with one thread per row (row math in double, mirroring the
// host f64 golden). qwen3.5 L2-normalizes q and k per (token,head) before the
// gated-delta recurrence, so this is the L2 piece of the GDN-mix VJP.
func L2NormBackwardDevice(worker *device.Worker, x, dY []float32, rows, width int, eps float64) ([]float32, error) {
	if rows <= 0 || width <= 0 || len(x) != rows*width || len(dY) != rows*width {
		return nil, fmt.Errorf("L2NormBackwardDevice: shape mismatch (rows=%d width=%d x=%d dY=%d)", rows, width, len(x), len(dY))
	}
	dX := make([]float32, rows*width)
	err := withCUDA(worker, func(scope *cudaScope) error {
		fn, err := scope.function("l2_norm_backward_f32")
		if err != nil {
			return err
		}
		xPtr, err := scope.upload(x)
		if err != nil {
			return err
		}
		dyPtr, err := scope.upload(dY)
		if err != nil {
			return err
		}
		dxPtr, err := scope.alloc(rows * width)
		if err != nil {
			return err
		}
		widthU, rowsU, epsF := uint32(width), uint32(rows), float32(eps)
		if err := scope.launch1D(fn, rowsU,
			unsafe.Pointer(&xPtr), unsafe.Pointer(&dyPtr), unsafe.Pointer(&dxPtr),
			unsafe.Pointer(&widthU), unsafe.Pointer(&rowsU), unsafe.Pointer(&epsF),
		); err != nil {
			return err
		}
		return scope.finish(cudaDownload{dX, dxPtr})
	})
	if err != nil {
		return nil, err
	}
	return dX, nil
}
