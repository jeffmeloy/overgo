package executor

import (
	"overgo/internal/cuda/device"
	"overgo/internal/cuda/driver"
	"overgo/internal/tensor"
)

func launchNativeNormalization(state *device.State, functions functionSet, node *tensor.Tensor, pointers launchPointerFrame) error {
	bf16 := node.Op == tensor.OpL2Norm
	x, y, err := tensor.NativeReductionGeometry(node.Shape, bf16)
	if err != nil {
		return err
	}
	width, rows := rowWords(node.Shape)
	input, output := pointers.input(0), pointers.output()
	var epsilon float32
	if bf16 {
		epsilon = node.Attrs.(tensor.L2NormAttributes).Epsilon
	} else {
		epsilon = node.Attrs.(tensor.RMSNormAttributes).Epsilon
	}
	mode := kernelBool(bf16)
	return launchGridABI(state, functions[kernelRmsNormNativeF32], driver.Dim3{X: rows, Y: 1, Z: 1}, driver.Dim3{X: x * y, Y: 1, Z: 1}, &input, &output, &width, &rows, &epsilon, &x, &y, &mode)
}
