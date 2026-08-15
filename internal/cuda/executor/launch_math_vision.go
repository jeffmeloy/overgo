package executor

import (
	"errors"
	"fmt"
	"math"

	"overgo/internal/cuda/cublas"
	"overgo/internal/cuda/device"
	"overgo/internal/cuda/driver"
	"overgo/internal/tensor"
)

func launchMathVision(
	state *device.State,
	functions functionSet,
	blas *blasState,
	node *tensor.Tensor,
	pointers launchPointerFrame,
	attributePointers devicePointerTable,
) error {
	output := pointers.output()
	switch node.Op {
	case tensor.OpLoRAMerge:
		attributes, ok := node.Attrs.(tensor.LoRAMergeAttributes)
		if !ok || len(node.Inputs) != 3 {
			return errors.New("invalid LoRA merge attributes")
		}
		count, err := elementCount32(node.Shape)
		if err != nil {
			return err
		}
		inner, err := uint32Checked(node.Shape.Dims[0], "LoRA merge inner width")
		if err != nil {
			return err
		}
		rows, err := uint32Checked(node.Shape.Dims[1], "LoRA merge row count")
		if err != nil {
			return err
		}
		rank, err := uint32Checked(node.Inputs[1].Shape.Dims[1], "LoRA merge rank")
		if err != nil {
			return err
		}
		groups := uint32(1)
		if node.Shape.Rank == 3 {
			groups, err = uint32Checked(node.Shape.Dims[2], "LoRA merge group count")
			if err != nil {
				return err
			}
		}
		base, a, b := pointers.input(0), pointers.input(1), pointers.input(2)
		scale := attributes.Scale
		return launch1DABI(
			state, functions[kernelLoraMergeF32], count,
			&base, &a, &b, &output, &inner, &rows, &rank, &groups, &scale, &count,
		)
	case tensor.OpAdd, tensor.OpMultiply, tensor.OpDivide:
		count, err := elementCount32(node.Shape)
		if err != nil {
			return err
		}
		left := pointers.input(0)
		right := pointers.input(1)
		if node.Inputs[0].Shape.Equal(node.Shape) && node.Inputs[1].Shape.Equal(node.Shape) {
			function := functions[kernelAddF32]
			if node.Op == tensor.OpMultiply {
				function = functions[kernelMultiplyF32]
			} else if node.Op == tensor.OpDivide {
				function = functions[kernelDivideF32]
			}
			return launch1DABI(state, function, count, &left, &right, &output, &count)
		}
		leftDimensions, err := shapeDimensions32(node.Inputs[0].Shape)
		if err != nil {
			return err
		}
		rightDimensions, err := shapeDimensions32(node.Inputs[1].Shape)
		if err != nil {
			return err
		}
		outputDimensions, err := shapeDimensions32(node.Shape)
		if err != nil {
			return err
		}
		function := functions[kernelBroadcastAddF32]
		if node.Op == tensor.OpMultiply {
			function = functions[kernelBroadcastMultiplyF32]
		} else if node.Op == tensor.OpDivide {
			function = functions[kernelBroadcastDivideF32]
		}
		return launch1DABI(
			state, function, count, &left, &right, &output, &count,
			&leftDimensions[0], &leftDimensions[1], &leftDimensions[2], &leftDimensions[3],
			&rightDimensions[0], &rightDimensions[1], &rightDimensions[2], &rightDimensions[3],
			&outputDimensions[0], &outputDimensions[1], &outputDimensions[2],
		)
	case tensor.OpScale:
		count, err := elementCount32(node.Shape)
		if err != nil {
			return err
		}
		attributes, ok := node.Attrs.(tensor.ScaleAttributes)
		if !ok {
			return errors.New("invalid scale attributes")
		}
		input := pointers.input(0)
		scale := attributes.Value
		return launch1DABI(state, functions[kernelScaleF32], count, &input, &output, &scale, &count)
	case tensor.OpClamp:
		count, err := elementCount32(node.Shape)
		if err != nil {
			return err
		}
		attributes, ok := node.Attrs.(tensor.ClampAttributes)
		if !ok {
			return errors.New("invalid clamp attributes")
		}
		input := pointers.input(0)
		minimum, maximum := attributes.Minimum, attributes.Maximum
		return launch1DABI(
			state, functions[kernelClampF32], count, &input, &output, &minimum, &maximum, &count,
		)
	case tensor.OpBF16Round:
		count, err := elementCount32(node.Shape)
		if err != nil {
			return err
		}
		input := pointers.input(0)
		return launch1DABI(state, functions[kernelBf16RoundF32], count, &input, &output, &count)
	case tensor.OpSiLU, tensor.OpGELU, tensor.OpGELUErf, tensor.OpReLU, tensor.OpReLUSquared, tensor.OpSigmoid, tensor.OpSoftplus, tensor.OpTanh, tensor.OpAtan, tensor.OpExp:
		count, err := elementCount32(node.Shape)
		if err != nil {
			return err
		}
		input := pointers.input(0)
		function := functions[kernelSiluF32]
		if node.Op == tensor.OpGELU {
			function = functions[kernelGeluF32]
		} else if node.Op == tensor.OpGELUErf {
			function = functions[kernelGeluErfF32]
		} else if node.Op == tensor.OpReLU {
			function = functions[kernelReluF32]
		} else if node.Op == tensor.OpReLUSquared {
			function = functions[kernelReluSquaredF32]
		} else if node.Op == tensor.OpSigmoid {
			function = functions[kernelSigmoidF32]
		} else if node.Op == tensor.OpSoftplus {
			function = functions[kernelSoftplusF32]
		} else if node.Op == tensor.OpTanh {
			function = functions[kernelTanhF32]
		} else if node.Op == tensor.OpAtan {
			function = functions[kernelAtanF32]
		} else if node.Op == tensor.OpExp {
			function = functions[kernelExpF32]
		}
		return launch1DABI(state, function, count, &input, &output, &count)
	case tensor.OpXIELU:
		count, err := elementCount32(node.Shape)
		if err != nil {
			return err
		}
		attributes, ok := node.Attrs.(tensor.XIELUAttributes)
		if !ok {
			return errors.New("invalid xIELU attributes")
		}
		input := pointers.input(0)
		alphaN := attributes.AlphaN
		alphaP := attributes.AlphaP
		beta := attributes.Beta
		epsilon := attributes.Epsilon
		return launch1DABI(
			state, functions[kernelXieluF32], count,
			&input, &output, &alphaN, &alphaP, &beta, &epsilon, &count,
		)
	case tensor.OpConv1DSame:
		attributes, ok := node.Attrs.(tensor.Conv1DAttributes)
		if !ok {
			return errors.New("invalid same Conv1D attributes")
		}
		count, err := elementCount32(node.Shape)
		if err != nil {
			return err
		}
		input, weight, bias := pointers.input(0), pointers.input(1), pointers.input(2)
		channelsIn := uint32(node.Inputs[0].Shape.Dims[0])
		tokens := uint32(node.Inputs[0].Shape.Dims[1])
		kernelWidth := uint32(node.Inputs[1].Shape.Dims[0])
		channelsOut := uint32(node.Shape.Dims[0])
		depthwise := kernelBool(attributes.Depthwise)
		return launch1DABI(
			state, functions[kernelConv1dSameF32], count,
			&input, &weight, &bias, &output, &channelsIn, &tokens, &kernelWidth,
			&channelsOut, &depthwise, &count,
		)
	case tensor.OpConv2D:
		attributes, ok := node.Attrs.(tensor.Conv2DAttributes)
		if !ok {
			return errors.New("invalid Conv2D attributes")
		}
		count, err := elementCount32(node.Shape)
		if err != nil {
			return err
		}
		input, weight := pointers.input(0), pointers.input(1)
		bias := input
		if attributes.HasBias {
			bias = pointers.input(2)
		}
		channelsIn := uint32(node.Inputs[0].Shape.Dims[0])
		inputW, inputH := uint32(node.Inputs[0].Shape.Dims[1]), uint32(node.Inputs[0].Shape.Dims[2])
		kernelW, kernelH := uint32(node.Inputs[1].Shape.Dims[0]), uint32(node.Inputs[1].Shape.Dims[1])
		weightChannels, channelsOut := uint32(node.Inputs[1].Shape.Dims[2]), uint32(node.Shape.Dims[0])
		outputW, outputH := uint32(node.Shape.Dims[1]), uint32(node.Shape.Dims[2])
		strideX, strideY := attributes.StrideX, attributes.StrideY
		padLeft, padTop := attributes.PadLeft, attributes.PadTop
		var depthwise, hasBias uint32
		if attributes.Depthwise {
			depthwise = 1
		}
		if attributes.HasBias {
			hasBias = 1
		}
		if depthwise == 0 && blas != nil && blas.staging != 0 {
			return launchBlasConv2D(
				state, functions, blas, input, weight, bias, output,
				channelsIn, inputW, inputH, kernelW, kernelH, channelsOut, outputW, outputH,
				strideX, strideY, padLeft, padTop, hasBias,
			)
		}
		return launch1DABI(
			state, functions[kernelConv2dF32], count,
			&input, &weight, &bias, &output, &channelsIn, &inputW, &inputH,
			&kernelW, &kernelH, &weightChannels, &channelsOut, &outputW, &outputH,
			&strideX, &strideY, &padLeft, &padTop, &depthwise, &hasBias, &count,
		)
	case tensor.OpWindowPartition2D, tensor.OpWindowUnpartition2D:
		attributes, ok := node.Attrs.(tensor.Window2DAttributes)
		if !ok {
			return errors.New("invalid window 2D attributes")
		}
		count, err := elementCount32(node.Shape)
		if err != nil {
			return err
		}
		input := pointers.input(0)
		channels := uint32(node.Shape.Dims[0])
		width, height, window := attributes.Width, attributes.Height, attributes.Window
		function := functions[kernelWindowPartition2dF32]
		if node.Op == tensor.OpWindowUnpartition2D {
			function = functions[kernelWindowUnpartition2dF32]
		}
		return launch1DABI(
			state, function, count, &input, &output, &channels, &width, &height, &window, &count,
		)
	case tensor.OpSAMAttention:
		attributes, ok := node.Attrs.(tensor.SAMAttentionAttributes)
		if !ok {
			return errors.New("invalid SAM attention attributes")
		}
		count, err := elementCount32(node.Shape)
		if err != nil {
			return err
		}
		query, key, value := pointers.input(0), pointers.input(1), pointers.input(2)
		relativeW, relativeH := pointers.input(3), pointers.input(4)
		keyWidth := uint32(node.Inputs[0].Shape.Dims[0])
		valueWidth := uint32(node.Inputs[2].Shape.Dims[0])
		queryHeads := uint32(node.Inputs[0].Shape.Dims[1])
		keyHeads := uint32(node.Inputs[1].Shape.Dims[1])
		tokens := uint32(node.Inputs[0].Shape.Dims[2])
		batches := uint32(node.Inputs[0].Shape.Dims[3])
		spatialSize := attributes.SpatialSize
		relativeWLength := uint32(node.Inputs[3].Shape.Dims[1])
		relativeHLength := uint32(node.Inputs[4].Shape.Dims[1])
		scale, relativeScale := attributes.Scale, attributes.RelativeScale
		return launch1DABI(
			state, functions[kernelSamAttentionF32], count,
			&query, &key, &value, &relativeW, &relativeH, &output,
			&keyWidth, &valueWidth, &queryHeads, &keyHeads, &tokens, &batches,
			&spatialSize, &relativeWLength, &relativeHLength, &scale, &relativeScale, &count,
		)
	case tensor.OpGroupNorm:
		attributes, ok := node.Attrs.(tensor.GroupNormAttributes)
		if !ok {
			return errors.New("invalid group norm attributes")
		}
		count, err := elementCount32(node.Shape)
		if err != nil {
			return err
		}
		input, weight, bias := pointers.input(0), pointers.input(1), pointers.input(2)
		channels := uint32(node.Shape.Dims[0])
		tokens := uint32(node.Shape.Dims[1])
		groups := attributes.Groups
		epsilon := attributes.Epsilon
		return launch1DABI(
			state, functions[kernelGroupNormF32], count,
			&input, &weight, &bias, &output, &channels, &tokens, &groups, &epsilon, &count,
		)
	case tensor.OpL2Norm:
		attributes, ok := node.Attrs.(tensor.L2NormAttributes)
		if !ok {
			return errors.New("invalid L2Norm attributes")
		}
		width, rows, err := rowDimensions32(node.Shape)
		if err != nil {
			return err
		}
		input := pointers.input(0)
		epsilon := attributes.Epsilon
		return launchNormalizationABI(state, functions[kernelL2NormF32], rows, &input, &output, &width, &rows, &epsilon)
	default:
		return fmt.Errorf("unsupported CUDA operation %s", node.Op)
	}
}

func launchBlasConv2D(
	state *device.State,
	functions functionSet,
	blas *blasState,
	input, weight, bias, output driver.DevicePtr,
	channelsIn, inputW, inputH, kernelW, kernelH, channelsOut, outputW, outputH,
	strideX, strideY, padLeft, padTop, hasBias uint32,
) error {
	inner64 := uint64(channelsIn) * uint64(kernelW) * uint64(kernelH)
	positions64 := uint64(outputW) * uint64(outputH)
	if inner64 == 0 || positions64 == 0 || inner64 > math.MaxUint32 || positions64 > math.MaxUint32 {
		return errors.New("Conv2D cuBLAS geometry overflows")
	}
	capacity := blas.stagingBytes / (inner64 * 4)
	if capacity == 0 {
		return errors.New("Conv2D cuBLAS workspace is too small")
	}
	inner, positions := uint32(inner64), uint32(positions64)
	for start := uint32(0); start < positions; {
		columns := min(uint32(min(capacity, uint64(math.MaxUint32))), positions-start)
		count64 := uint64(inner) * uint64(columns)
		if count64 > math.MaxUint32 {
			return errors.New("Conv2D im2col launch overflows")
		}
		count := uint32(count64)
		if err := launch1DABI(
			state, functions[kernelConv2dIm2colF32], count,
			&input, &blas.staging, &channelsIn, &inputW, &inputH,
			&kernelW, &kernelH, &outputW, &outputH, &strideX, &strideY,
			&padLeft, &padTop, &start, &columns, &inner, &count,
		); err != nil {
			return err
		}
		tileOutput := output + driver.DevicePtr(uint64(start)*uint64(channelsOut)*4)
		if !traceExternalCall(
			state, traceTagSGEMM,
			uint64(weight), uint64(blas.staging), uint64(tileOutput),
			uint64(channelsOut), uint64(columns), uint64(inner),
		) {
			if err := blas.library.SGEMM(
				blas.handle, cublas.OperationTranspose, cublas.OperationNone,
				int32(channelsOut), int32(columns), int32(inner), 1,
				weight, int32(inner), blas.staging, int32(inner), 0,
				tileOutput, int32(channelsOut),
			); err != nil {
				return err
			}
		}
		if hasBias != 0 {
			tileCount := channelsOut * columns
			one := uint32(1)
			if err := launch1DABI(
				state, functions[kernelBroadcastAddF32], tileCount,
				&tileOutput, &bias, &tileOutput, &tileCount,
				&channelsOut, &columns, &one, &one,
				&channelsOut, &one, &one, &one,
				&channelsOut, &columns, &one,
			); err != nil {
				return err
			}
		}
		start += columns
	}
	blas.stagedNode = nil
	return nil
}
