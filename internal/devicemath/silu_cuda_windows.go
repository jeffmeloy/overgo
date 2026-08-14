//go:build windows

package devicemath

import (
	"fmt"
	"math"
	"unsafe"

	"overgo/internal/cuda/device"
)

// SiLUGateForward computes the SwiGLU activation forward on the GPU, returning
// BOTH intermediates the gated-MLP cache needs: a = silu(gate) and
// hMLP = a * up (elementwise). silu via silu_f32, the product via multiply_f32,
// in one cudaScope session. Matches layerForwardCached's host computation.
func SiLUGateForward(worker *device.Worker, gate, up []float32) (a, hMLP []float32, err error) {
	n := len(gate)
	if n == 0 || len(up) != n {
		return nil, nil, fmt.Errorf("SiLUGateForward: length mismatch (gate=%d up=%d)", n, len(up))
	}
	if uint64(n) > math.MaxUint32 {
		return nil, nil, fmt.Errorf("SiLUGateForward: input too large for a 32-bit element count")
	}
	a = make([]float32, n)
	hMLP = make([]float32, n)
	err = withCUDA(worker, func(scope *cudaScope) error {
		siluFn, err := scope.function("silu_f32")
		if err != nil {
			return err
		}
		mulFn, err := scope.function("multiply_f32")
		if err != nil {
			return err
		}
		gatePtr, err := scope.upload(gate)
		if err != nil {
			return err
		}
		upPtr, err := scope.upload(up)
		if err != nil {
			return err
		}
		aPtr, err := scope.alloc(n)
		if err != nil {
			return err
		}
		hPtr, err := scope.alloc(n)
		if err != nil {
			return err
		}
		count := uint32(n)
		// a = silu(gate)
		if err := scope.launch1D(siluFn, count,
			unsafe.Pointer(&gatePtr), unsafe.Pointer(&aPtr), unsafe.Pointer(&count),
		); err != nil {
			return err
		}
		// hMLP = a * up
		if err := scope.launchVector3(mulFn, aPtr, upPtr, hPtr, n); err != nil {
			return err
		}
		return scope.finish(cudaDownload{a, aPtr}, cudaDownload{hMLP, hPtr})
	})
	if err != nil {
		return nil, nil, err
	}
	return a, hMLP, nil
}

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
