package executor

import (
	"errors"

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

// int8QuantizeGroups mirrors the kernels' INT8_QUANTIZE_GROUPS: the most rotation groups a warp of the activation
// quantizer holds.
const int8QuantizeGroups = uint32(4)

// int8QuantizeBlock is the activation quantizer's block for token columns of inner values: a power of two of warps,
// the fewest of which none holds more than int8QuantizeGroups of a column's rotation groups.
func int8QuantizeBlock(inner uint32) driver.Dim3 {
	warps := uint32(1)
	for warps*int8QuantizeGroups*uint32(int8ConvRotGroup) < inner {
		warps *= 2
	}
	return kernel.Grid1D(int(warps) * kernel.WarpThreads())
}

// int8ConvRotColumns quantizes the F32 activation at right, rows token columns of inner values
// (quantize_int8_convrot_columns_f32).
func int8ConvRotColumns(state *device.State, functions functionSet, right driver.DevicePtr, inner, rows uint32) int8ConvRotQuantizer {
	return func(activation, columnScale driver.DevicePtr) error {
		return launchGridABI(
			state, functions[kernelQuantizeInt8ConvrotColumnsF32],
			kernel.Grid1D(int(rows)), int8QuantizeBlock(inner),
			&right, &activation, &columnScale, &inner, &rows,
		)
	}
}

// int8ProductTokens, int8ProductFeatures, int8ProductThreads and int8ProductSharedBytes mirror the kernels'
// INT8_PRODUCT_TOKENS, INT8_PRODUCT_FEATURES, INT8_PRODUCT_THREADS and INT8_PRODUCT_SHARED_BYTES: the token and feature
// rows a block of the int8 ConvRot product multiplies, its threads, and the dynamic shared memory its three stages hold.
const (
	int8ProductTokens      = uint32(128)
	int8ProductFeatures    = uint32(256)
	int8ProductThreads     = 256
	int8ProductSharedBytes = uint32(3 * (128 + 256) * 64)
)

// int8ConvRotProduct multiplies an int8 ConvRot weight by the quantized activation at activation, its column scales at
// columnScale, and writes the output its epilogue makes (int8_convrot_product_bf16_f32 and its gated sum and head
// norm), the scales applied in the product as comfy_kitchen's CUTLASS product applies them.
type int8ConvRotProduct func(activation, columnScale driver.DevicePtr) error

// launchInt8ConvRotProduct launches an int8 ConvRot product kernel over leftRows weight rows and rightRows token
// columns: a grid's x the token tiles, its y the feature tiles, so the blocks reading one tile of the weight run
// together.
func launchInt8ConvRotProduct(state *device.State, function boundKernel, leftRows, rightRows uint32, arguments ...any) error {
	grid := driver.Dim3{
		X: (rightRows + int8ProductTokens - 1) / int8ProductTokens,
		Y: (leftRows + int8ProductFeatures - 1) / int8ProductFeatures,
		Z: 1,
	}
	return launchGridSharedABI(state, function, grid, kernel.Grid1D(int8ProductThreads), int8ProductSharedBytes, arguments...)
}

// int8ConvRotScaled is the int8 ConvRot product whose epilogue writes each entry scaled by its column's scale and then
// its row's and rounded to BF16, as comfy_kitchen's int8_linear does (int8_convrot_product_bf16_f32): the weight at left,
// its rows' F32 scales after them.
func int8ConvRotScaled(state *device.State, functions functionSet, left, output driver.DevicePtr, inner, leftRows, rightRows uint32) int8ConvRotProduct {
	weightScale := left + driver.DevicePtr(uint64(inner)*uint64(leftRows))
	return func(activation, columnScale driver.DevicePtr) error {
		return launchInt8ConvRotProduct(state, functions[kernelInt8ConvrotProductBf16F32], leftRows, rightRows,
			&activation, &columnScale, &left, &weightScale, &output, &inner, &leftRows, &rightRows)
	}
}

// launchInt8ConvRotMulMat multiplies a resident int8 ConvRot weight at left by its activation as comfy_kitchen's CUDA
// int8_linear does (moss-int8, its product matched bit for bit): each token column rotated by the group's Hadamard
// matrix and quantized to int8 with one scale a column, once per activation node (the memo the FP8 product keeps), then
// one product on the int8 tensor cores whose epilogue scales each exact int32 entry by its column's scale and then its
// weight row's and rounds it to BF16 before it writes it. The weight's rows are followed in its resident buffer by their
// F32 scales, as an F8E4M3 weight's are. quantize forms the activation side (int8ConvRotColumns, or a prologue that
// computes the activation as it quantizes it, launchInt8Prologue), and product writes the output (int8ConvRotScaled, or
// a gated residual sum or head norm that reads it alone, launchInt8GatedAdd and launchInt8NormRotary).
func launchInt8ConvRotMulMat(
	blas *blasState,
	rightNode *tensor.Tensor,
	left, output driver.DevicePtr,
	quantize int8ConvRotQuantizer,
	product int8ConvRotProduct,
	inner, rightRows uint32,
) error {
	if blas == nil || blas.stagingBytes < byteActivationStagingBytes(uint64(inner), uint64(rightRows)) {
		return errors.New("int8 ConvRot mul_mat workspace is unavailable")
	}
	for _, pointer := range []driver.DevicePtr{left, output} {
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
	return product(activation, columnScale)
}
