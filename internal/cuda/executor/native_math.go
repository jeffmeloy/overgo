package executor

import (
	"fmt"
	"math/bits"

	"overgo/internal/cuda/device"
	"overgo/internal/cuda/driver"
	"overgo/internal/tensor"
)

// Native reduction geometry follows Torch 2.12 Reduce.cuh's F32 vectorized input path.
// Reconsider these lane, thread and split constants with another source policy.
const (
	nativeMathVector        = 4
	nativeMathWarp          = 32
	nativeMathThreads       = 512
	nativeMathMinimumValues = 16
	nativeMathMaximumValues = 256
)

func nativeRMSGeometry(shape tensor.Shape) (uint32, uint32, error) {
	width, rows, err := rowDimensions32(shape)
	if err != nil {
		return 0, 0, err
	}
	if width < nativeMathWarp*nativeMathVector || width%nativeMathVector != 0 {
		return 0, 0, fmt.Errorf("native RMS width %d requires aligned vectorized rows of at least %d values", width, nativeMathWarp*nativeMathVector)
	}
	power := func(v uint32) uint32 { return 1 << (bits.Len32(v) - 1) }
	x := min(power(width/nativeMathVector), uint32(nativeMathWarp))
	y := min(power(rows), uint32(nativeMathThreads)/x)
	x = min(power(width/nativeMathVector), uint32(nativeMathThreads)/y)
	if 1+(width-1)/x < min(y*nativeMathMinimumValues, uint32(nativeMathMaximumValues)) {
		y = 1
	}
	if 1+(width-1)/(x*y) >= nativeMathMaximumValues {
		return 0, 0, fmt.Errorf("native RMS width %d requires unsupported cross-block reduction", width)
	}
	return x, y, nil
}

func launchNativeRMS(state *device.State, functions functionSet, node *tensor.Tensor, pointers launchPointerFrame) error {
	x, y, err := nativeRMSGeometry(node.Shape)
	if err != nil {
		return err
	}
	width, rows := rowWords(node.Shape)
	input, output := pointers.input(0), pointers.output()
	epsilon := node.Attrs.(tensor.RMSNormAttributes).Epsilon
	return launchGridABI(state, functions[kernelRmsNormNativeF32], driver.Dim3{X: rows, Y: 1, Z: 1}, driver.Dim3{X: x * y, Y: 1, Z: 1}, &input, &output, &width, &rows, &epsilon, &x, &y)
}
