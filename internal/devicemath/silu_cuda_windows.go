//go:build windows

package devicemath

import (
	"fmt"
	"math"
	"unsafe"

	"overgo/internal/cuda/device"
	"overgo/internal/cuda/driver"
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
		var siluFn, mulFn driver.Function
		if err := scope.functions([]deviceKernel{{&siluFn, "silu_f32"}, {&mulFn, "multiply_f32"}}); err != nil {
			return err
		}
		var gatePtr, upPtr, aPtr, hPtr driver.DevicePtr
		if err := scope.fill([]deviceSlot{{&gatePtr, gate, 0}, {&upPtr, up, 0}, {&aPtr, nil, n}, {&hPtr, nil, n}}); err != nil {
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
