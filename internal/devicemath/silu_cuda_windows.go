//go:build windows

package devicemath

import (
	"fmt"
	"math"
	"unsafe"

	"overgo/internal/cuda/device"
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
	err := withCUDA(worker, func(scope *cudaScope) error {
		fn, err := scope.function("silu_backward_f32")
		if err != nil {
			return err
		}
		dyPtr, err := scope.upload(gradOutput)
		if err != nil {
			return err
		}
		xPtr, err := scope.upload(x)
		if err != nil {
			return err
		}
		dxPtr, err := scope.alloc(n)
		if err != nil {
			return err
		}

		count := uint32(n)
		if err := scope.launch1D(fn, count,
			unsafe.Pointer(&dyPtr), unsafe.Pointer(&xPtr), unsafe.Pointer(&dxPtr), unsafe.Pointer(&count),
		); err != nil {
			return err
		}
		return scope.finish(cudaDownload{gradInput, dxPtr})
	})
	if err != nil {
		return nil, err
	}
	return gradInput, nil
}
