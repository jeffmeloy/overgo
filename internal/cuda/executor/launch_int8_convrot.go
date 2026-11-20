package executor

import (
	"errors"
	"math"

	"overgo/internal/cuda/device"
	"overgo/internal/cuda/driver"
	"overgo/internal/cuda/kernel"
	"overgo/internal/tensor"
	"overgo/internal/tensor/dtype"
)

// int8ConvRotGroup and int8ConvRotMaxInner mirror the kernels' INT8_CONVROT_GROUP and INT8_CONVROT_MAX_INNER: the
// width of comfy_kitchen's ConvRot rotation, which an int8 ConvRot product's inner dimension is a multiple of, and the
// widest token column its quantization holds in shared memory.
const (
	int8ConvRotGroup    = uint64(256)
	int8ConvRotMaxInner = uint64(8192)
)

// int8ConvRotMulMat reports whether a mul_mat node multiplies an int8 ConvRot weight (dtype.I8ConvRot), whose only
// arithmetic is comfy_kitchen's int8 product on the tensor cores (launchInt8ConvRotMulMat), in a shape it runs: an F32
// activation over an inner dimension the rotation's groups divide and the quantization holds, and weight rows the
// epilogue's quads divide. A graph with an int8 ConvRot mul_mat of another shape does not compile.
func int8ConvRotMulMat(node *tensor.Tensor) bool {
	if node.Op != tensor.OpMulMat || len(node.Inputs) != 2 || node.Inputs[0].Type != dtype.I8ConvRot ||
		node.Inputs[1].Type != dtype.F32 {
		return false
	}
	inner, leftRows := node.Inputs[0].Shape.Dims[0], node.Inputs[0].Shape.Dims[1]
	return inner%int8ConvRotGroup == 0 && inner <= int8ConvRotMaxInner && leftRows%fp8QuadWidth == 0
}

// int8ConvRotQuantizer writes an int8 ConvRot product's activation side: each token column's int8 values at activation
// and its scale at columnScale.
type int8ConvRotQuantizer func(activation, columnScale driver.DevicePtr) error

// int8ConvRotColumns quantizes the F32 activation at right, rows token columns of inner values
// (quantize_int8_convrot_columns_f32).
func int8ConvRotColumns(state *device.State, functions functionSet, right driver.DevicePtr, inner, rows uint32) int8ConvRotQuantizer {
	return func(activation, columnScale driver.DevicePtr) error {
		return launchGridABI(
			state, functions[kernelQuantizeInt8ConvrotColumnsF32],
			kernel.Grid1D(int(rows)), kernel.DefaultBlock1D(),
			&right, &activation, &columnScale, &inner, &rows,
		)
	}
}

// int8ConvRotEpilogue turns an int8 ConvRot product's exact int32 entries at output, rows a column, into what the
// output holds, from the weight rows' scales and the activation columns' scales, in quads of four entries.
type int8ConvRotEpilogue func(output, weightScale, columnScale driver.DevicePtr, rows, quads uint32) error

// int8ConvRotScale scales each entry by its column's scale and then its row's and rounds it to BF16, as comfy_kitchen's
// int8_linear does (scale_int8_product_bf16_f32).
func int8ConvRotScale(state *device.State, functions functionSet) int8ConvRotEpilogue {
	return func(output, weightScale, columnScale driver.DevicePtr, rows, quads uint32) error {
		return launch1DABI(state, functions[kernelScaleInt8ProductBf16F32], quads, &output, &weightScale, &columnScale, &rows, &quads)
	}
}

// launchInt8ConvRotMulMat multiplies a resident int8 ConvRot weight by its activation as comfy_kitchen's CUDA
// int8_linear does (moss-int8, its product matched bit for bit): each token column rotated by the group's Hadamard
// matrix and quantized to int8 with one scale a column, once per activation node (the memo the FP8 product keeps), one
// cuBLASLt int8 product accumulating exactly into int32 over the output, and an epilogue scaling each entry by its
// column's scale and then its weight row's and rounding it to BF16 in place. The weight's rows are followed in its
// resident buffer by their F32 scales, as an F8E4M3 weight's are. quantize forms the activation side (int8ConvRotColumns, or a
// prologue that computes the activation as it quantizes it, launchInt8Prologue), and epilogue scales the int32 product
// into the output (int8ConvRotScale, or a gated residual sum that reads it alone, launchInt8GatedAdd).
func launchInt8ConvRotMulMat(
	state *device.State,
	functions functionSet,
	blas *blasState,
	rightNode *tensor.Tensor,
	left, weightScale, output driver.DevicePtr,
	quantize int8ConvRotQuantizer,
	epilogue int8ConvRotEpilogue,
	inner, leftRows, rightRows uint32,
) error {
	if blas == nil || blas.lt == 0 || blas.stagingBytes < byteActivationStagingBytes(uint64(inner), uint64(rightRows)) {
		return errors.New("int8 ConvRot mul_mat workspace is unavailable")
	}
	outputElements := uint64(leftRows) * uint64(rightRows)
	if outputElements > math.MaxUint32 {
		return errors.New("int8 ConvRot mul_mat output exceeds uint32")
	}
	for _, pointer := range []driver.DevicePtr{left, weightScale, output} {
		if uint64(pointer)%fp8TensorCoreAlignment != 0 {
			return errors.New("int8 ConvRot mul_mat operand is not 16-byte aligned")
		}
	}
	activation := blas.staging + driver.DevicePtr(ltGemmWorkspaceBytes)
	columnScale := activation + driver.DevicePtr(uint64(inner)*uint64(rightRows))
	if blas.stagedNode != rightNode || blas.stagedType != dtype.I8ConvRot {
		if err := quantize(activation, columnScale); err != nil {
			return err
		}
		blas.stagedNode, blas.stagedType = rightNode, dtype.I8ConvRot
	}
	if !traceExternalCall(
		state, traceTagInt8Matmul,
		uint64(left), uint64(activation), uint64(output),
		uint64(leftRows), uint64(rightRows), uint64(inner),
	) {
		plan, err := blas.matmulPlan(inner, leftRows, rightRows, matmulInt8)
		if err != nil {
			return err
		}
		if err := blas.library.Matmul(
			blas.lt, plan, left, activation, output, 0, blas.staging, ltGemmWorkspaceBytes, state.Stream,
		); err != nil {
			return err
		}
	}
	return epilogue(output, weightScale, columnScale, leftRows, uint32(outputElements/fp8QuadWidth))
}
