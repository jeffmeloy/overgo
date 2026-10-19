package executor

import (
	"errors"
	"math"

	"overgo/internal/cuda/device"
	"overgo/internal/cuda/driver"
	"overgo/internal/tensor"
)

// bf16LinearAlignment matches the minimum operand alignment promised to cuBLASLt.
const bf16LinearAlignment = uint64(16)

// bf16LinearBiasOperand follows the weight and activation in the compiled projection.
const bf16LinearBiasOperand = 2

func bf16LinearSpan(elements uint64) uint64 {
	return (elements*bf16ScalarBytes + bf16LinearAlignment - 1) / bf16LinearAlignment * bf16LinearAlignment
}

func bf16LinearStagingBytes(operands []*tensor.Tensor) (uint64, error) {
	inner, rows, tokens := operands[0].Shape.Dims[0], operands[0].Shape.Dims[1], operands[1].Shape.Dims[1]
	if inner > math.MaxInt32 || rows > math.MaxInt32 || tokens > math.MaxInt32 ||
		inner*tokens > math.MaxUint32 || rows*tokens > math.MaxUint32 {
		return 0, errors.New("BF16 linear dimensions exceed cuBLAS or packing limits")
	}
	return ltGemmWorkspaceBytes + bf16LinearSpan(inner*tokens) + bf16LinearSpan(rows) + bf16LinearSpan(rows*tokens), nil
}

func launchBF16Linear(state *device.State, functions functionSet, blas *blasState, operands []*tensor.Tensor, pointers launchPointerFrame) error {
	bytes, err := bf16LinearStagingBytes(operands)
	if err != nil {
		return err
	}
	if blas == nil || blas.lt == 0 || blas.stagingBytes < bytes {
		return errors.New("BF16 linear staging is unavailable")
	}
	inner, rows, tokens := uint32(operands[0].Shape.Dims[0]), uint32(operands[0].Shape.Dims[1]), uint32(operands[1].Shape.Dims[1])
	weight, input, bias, output := pointers.input(0), pointers.input(1), pointers.input(bf16LinearBiasOperand), pointers.output()
	if uint64(weight)%bf16LinearAlignment != 0 {
		return errors.New("BF16 linear weight requires 16-byte alignment")
	}
	activation := blas.staging + driver.DevicePtr(ltGemmWorkspaceBytes)
	packedBias := activation + driver.DevicePtr(bf16LinearSpan(uint64(inner)*uint64(tokens)))
	packedOutput := packedBias + driver.DevicePtr(bf16LinearSpan(uint64(rows)))
	// The shared pack memo belongs to the staging base used by other products.
	blas.stagedNode = nil
	for _, pack := range []struct {
		input, output driver.DevicePtr
		count         uint32
	}{{input, activation, inner * tokens}, {bias, packedBias, rows}} {
		if err := launch1DABI(state, functions[kernelF32ToBf16], pack.count, &pack.input, &pack.output, &pack.count); err != nil {
			return err
		}
	}
	if !traceExternalCall(state, traceTagBF16Matmul, uint64(weight), uint64(activation), uint64(packedOutput), uint64(rows), uint64(tokens), uint64(inner)) {
		plan, err := blas.matmulPlan(inner, rows, tokens, true)
		if err != nil {
			return err
		}
		if err := blas.library.Matmul(blas.lt, plan, weight, activation, packedOutput, packedBias, blas.staging, ltGemmWorkspaceBytes, state.Stream); err != nil {
			return err
		}
	}
	count := rows * tokens
	return launch1DABI(state, functions[kernelBf16ToF32], count, &packedOutput, &output, &count)
}
