package executor

import (
	"errors"
	"fmt"
	"runtime"
	"unsafe"

	"llamacpp2go/internal/cuda/device"
	"llamacpp2go/internal/cuda/driver"
	"llamacpp2go/internal/tensor"
)

func launchMathVision(
	state *device.State,
	functions functionSet,
	blas *blasState,
	node *tensor.Tensor,
	pointers map[*tensor.Tensor]driver.DevicePtr,
	attributePointers map[*tensor.Tensor]driver.DevicePtr,
) error {
	output := pointers[node]
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
		base, a, b := pointers[node.Inputs[0]], pointers[node.Inputs[1]], pointers[node.Inputs[2]]
		scale := attributes.Scale
		args := []unsafe.Pointer{
			unsafe.Pointer(&base), unsafe.Pointer(&a), unsafe.Pointer(&b), unsafe.Pointer(&output),
			unsafe.Pointer(&inner), unsafe.Pointer(&rows), unsafe.Pointer(&rank), unsafe.Pointer(&groups),
			unsafe.Pointer(&scale), unsafe.Pointer(&count),
		}
		err = launch1D(state, functions.loraMerge, count, args)
		runtime.KeepAlive(args)
		return err
	case tensor.OpAdd, tensor.OpMultiply, tensor.OpDivide:
		count, err := elementCount32(node.Shape)
		if err != nil {
			return err
		}
		left := pointers[node.Inputs[0]]
		right := pointers[node.Inputs[1]]
		if node.Inputs[0].Shape.Equal(node.Shape) && node.Inputs[1].Shape.Equal(node.Shape) {
			function := functions.add
			if node.Op == tensor.OpMultiply {
				function = functions.multiply
			} else if node.Op == tensor.OpDivide {
				function = functions.divide
			}
			args := []unsafe.Pointer{
				unsafe.Pointer(&left),
				unsafe.Pointer(&right),
				unsafe.Pointer(&output),
				unsafe.Pointer(&count),
			}
			err = launch1D(state, function, count, args)
			runtime.KeepAlive(left)
			runtime.KeepAlive(right)
			runtime.KeepAlive(output)
			runtime.KeepAlive(count)
			return err
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
		function := functions.broadcastAdd
		if node.Op == tensor.OpMultiply {
			function = functions.broadcastMultiply
		} else if node.Op == tensor.OpDivide {
			function = functions.broadcastDivide
		}
		args := []unsafe.Pointer{
			unsafe.Pointer(&left),
			unsafe.Pointer(&right),
			unsafe.Pointer(&output),
			unsafe.Pointer(&count),
			unsafe.Pointer(&leftDimensions[0]),
			unsafe.Pointer(&leftDimensions[1]),
			unsafe.Pointer(&leftDimensions[2]),
			unsafe.Pointer(&leftDimensions[3]),
			unsafe.Pointer(&rightDimensions[0]),
			unsafe.Pointer(&rightDimensions[1]),
			unsafe.Pointer(&rightDimensions[2]),
			unsafe.Pointer(&rightDimensions[3]),
			unsafe.Pointer(&outputDimensions[0]),
			unsafe.Pointer(&outputDimensions[1]),
			unsafe.Pointer(&outputDimensions[2]),
		}
		err = launch1D(state, function, count, args)
		runtime.KeepAlive(left)
		runtime.KeepAlive(right)
		runtime.KeepAlive(output)
		runtime.KeepAlive(count)
		runtime.KeepAlive(leftDimensions)
		runtime.KeepAlive(rightDimensions)
		runtime.KeepAlive(outputDimensions)
		return err
	case tensor.OpScale:
		count, err := elementCount32(node.Shape)
		if err != nil {
			return err
		}
		attributes, ok := node.Attrs.(tensor.ScaleAttributes)
		if !ok {
			return errors.New("invalid scale attributes")
		}
		input := pointers[node.Inputs[0]]
		scale := attributes.Value
		args := []unsafe.Pointer{
			unsafe.Pointer(&input),
			unsafe.Pointer(&output),
			unsafe.Pointer(&scale),
			unsafe.Pointer(&count),
		}
		err = launch1D(state, functions.scale, count, args)
		runtime.KeepAlive(input)
		runtime.KeepAlive(output)
		runtime.KeepAlive(scale)
		runtime.KeepAlive(count)
		return err
	case tensor.OpClamp:
		count, err := elementCount32(node.Shape)
		if err != nil {
			return err
		}
		attributes, ok := node.Attrs.(tensor.ClampAttributes)
		if !ok {
			return errors.New("invalid clamp attributes")
		}
		input := pointers[node.Inputs[0]]
		minimum, maximum := attributes.Minimum, attributes.Maximum
		args := []unsafe.Pointer{
			unsafe.Pointer(&input), unsafe.Pointer(&output), unsafe.Pointer(&minimum),
			unsafe.Pointer(&maximum), unsafe.Pointer(&count),
		}
		err = launch1D(state, functions.clamp, count, args)
		runtime.KeepAlive(input)
		runtime.KeepAlive(output)
		runtime.KeepAlive(minimum)
		runtime.KeepAlive(maximum)
		runtime.KeepAlive(count)
		return err
	case tensor.OpBF16Round:
		count, err := elementCount32(node.Shape)
		if err != nil {
			return err
		}
		input := pointers[node.Inputs[0]]
		args := []unsafe.Pointer{
			unsafe.Pointer(&input), unsafe.Pointer(&output), unsafe.Pointer(&count),
		}
		err = launch1D(state, functions.bf16Round, count, args)
		runtime.KeepAlive(args)
		return err
	case tensor.OpSiLU, tensor.OpGELU, tensor.OpGELUErf, tensor.OpReLU, tensor.OpReLUSquared, tensor.OpSigmoid, tensor.OpSoftplus, tensor.OpTanh, tensor.OpExp:
		count, err := elementCount32(node.Shape)
		if err != nil {
			return err
		}
		input := pointers[node.Inputs[0]]
		args := []unsafe.Pointer{
			unsafe.Pointer(&input),
			unsafe.Pointer(&output),
			unsafe.Pointer(&count),
		}
		function := functions.silu
		if node.Op == tensor.OpGELU {
			function = functions.gelu
		} else if node.Op == tensor.OpGELUErf {
			function = functions.geluErf
		} else if node.Op == tensor.OpReLU {
			function = functions.relu
		} else if node.Op == tensor.OpReLUSquared {
			function = functions.reluSquared
		} else if node.Op == tensor.OpSigmoid {
			function = functions.sigmoid
		} else if node.Op == tensor.OpSoftplus {
			function = functions.softplus
		} else if node.Op == tensor.OpTanh {
			function = functions.tanh
		} else if node.Op == tensor.OpExp {
			function = functions.exp
		}
		err = launch1D(state, function, count, args)
		runtime.KeepAlive(input)
		runtime.KeepAlive(output)
		runtime.KeepAlive(count)
		return err
	case tensor.OpXIELU:
		count, err := elementCount32(node.Shape)
		if err != nil {
			return err
		}
		attributes, ok := node.Attrs.(tensor.XIELUAttributes)
		if !ok {
			return errors.New("invalid xIELU attributes")
		}
		input := pointers[node.Inputs[0]]
		alphaN := attributes.AlphaN
		alphaP := attributes.AlphaP
		beta := attributes.Beta
		epsilon := attributes.Epsilon
		args := []unsafe.Pointer{
			unsafe.Pointer(&input),
			unsafe.Pointer(&output),
			unsafe.Pointer(&alphaN),
			unsafe.Pointer(&alphaP),
			unsafe.Pointer(&beta),
			unsafe.Pointer(&epsilon),
			unsafe.Pointer(&count),
		}
		err = launch1D(state, functions.xielu, count, args)
		runtime.KeepAlive(input)
		runtime.KeepAlive(output)
		runtime.KeepAlive(alphaN)
		runtime.KeepAlive(alphaP)
		runtime.KeepAlive(beta)
		runtime.KeepAlive(epsilon)
		runtime.KeepAlive(count)
		return err
	case tensor.OpConv1DSame:
		attributes, ok := node.Attrs.(tensor.Conv1DAttributes)
		if !ok {
			return errors.New("invalid same Conv1D attributes")
		}
		count, err := elementCount32(node.Shape)
		if err != nil {
			return err
		}
		input, weight, bias := pointers[node.Inputs[0]], pointers[node.Inputs[1]], pointers[node.Inputs[2]]
		channelsIn := uint32(node.Inputs[0].Shape.Dims[0])
		tokens := uint32(node.Inputs[0].Shape.Dims[1])
		kernelWidth := uint32(node.Inputs[1].Shape.Dims[0])
		channelsOut := uint32(node.Shape.Dims[0])
		var depthwise uint32
		if attributes.Depthwise {
			depthwise = 1
		}
		args := []unsafe.Pointer{
			unsafe.Pointer(&input), unsafe.Pointer(&weight), unsafe.Pointer(&bias), unsafe.Pointer(&output),
			unsafe.Pointer(&channelsIn), unsafe.Pointer(&tokens), unsafe.Pointer(&kernelWidth),
			unsafe.Pointer(&channelsOut), unsafe.Pointer(&depthwise), unsafe.Pointer(&count),
		}
		err = launch1D(state, functions.conv1DSame, count, args)
		runtime.KeepAlive(args)
		return err
	case tensor.OpConv2D:
		attributes, ok := node.Attrs.(tensor.Conv2DAttributes)
		if !ok {
			return errors.New("invalid Conv2D attributes")
		}
		count, err := elementCount32(node.Shape)
		if err != nil {
			return err
		}
		input, weight := pointers[node.Inputs[0]], pointers[node.Inputs[1]]
		bias := input
		if attributes.HasBias {
			bias = pointers[node.Inputs[2]]
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
		args := []unsafe.Pointer{
			unsafe.Pointer(&input), unsafe.Pointer(&weight), unsafe.Pointer(&bias), unsafe.Pointer(&output),
			unsafe.Pointer(&channelsIn), unsafe.Pointer(&inputW), unsafe.Pointer(&inputH),
			unsafe.Pointer(&kernelW), unsafe.Pointer(&kernelH), unsafe.Pointer(&weightChannels),
			unsafe.Pointer(&channelsOut), unsafe.Pointer(&outputW), unsafe.Pointer(&outputH),
			unsafe.Pointer(&strideX), unsafe.Pointer(&strideY), unsafe.Pointer(&padLeft), unsafe.Pointer(&padTop),
			unsafe.Pointer(&depthwise), unsafe.Pointer(&hasBias), unsafe.Pointer(&count),
		}
		err = launch1D(state, functions.conv2D, count, args)
		runtime.KeepAlive(args)
		return err
	case tensor.OpWindowPartition2D, tensor.OpWindowUnpartition2D:
		attributes, ok := node.Attrs.(tensor.Window2DAttributes)
		if !ok {
			return errors.New("invalid window 2D attributes")
		}
		count, err := elementCount32(node.Shape)
		if err != nil {
			return err
		}
		input := pointers[node.Inputs[0]]
		channels := uint32(node.Shape.Dims[0])
		width, height, window := attributes.Width, attributes.Height, attributes.Window
		function := functions.windowPartition2D
		if node.Op == tensor.OpWindowUnpartition2D {
			function = functions.windowUnpartition2D
		}
		args := []unsafe.Pointer{
			unsafe.Pointer(&input), unsafe.Pointer(&output), unsafe.Pointer(&channels),
			unsafe.Pointer(&width), unsafe.Pointer(&height), unsafe.Pointer(&window), unsafe.Pointer(&count),
		}
		err = launch1D(state, function, count, args)
		runtime.KeepAlive(args)
		return err
	case tensor.OpSAMAttention:
		attributes, ok := node.Attrs.(tensor.SAMAttentionAttributes)
		if !ok {
			return errors.New("invalid SAM attention attributes")
		}
		count, err := elementCount32(node.Shape)
		if err != nil {
			return err
		}
		query, key, value := pointers[node.Inputs[0]], pointers[node.Inputs[1]], pointers[node.Inputs[2]]
		relativeW, relativeH := pointers[node.Inputs[3]], pointers[node.Inputs[4]]
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
		args := []unsafe.Pointer{
			unsafe.Pointer(&query), unsafe.Pointer(&key), unsafe.Pointer(&value),
			unsafe.Pointer(&relativeW), unsafe.Pointer(&relativeH), unsafe.Pointer(&output),
			unsafe.Pointer(&keyWidth), unsafe.Pointer(&valueWidth), unsafe.Pointer(&queryHeads),
			unsafe.Pointer(&keyHeads), unsafe.Pointer(&tokens), unsafe.Pointer(&batches),
			unsafe.Pointer(&spatialSize), unsafe.Pointer(&relativeWLength), unsafe.Pointer(&relativeHLength),
			unsafe.Pointer(&scale), unsafe.Pointer(&relativeScale), unsafe.Pointer(&count),
		}
		err = launch1D(state, functions.samAttention, count, args)
		runtime.KeepAlive(args)
		return err
	case tensor.OpGroupNorm:
		attributes, ok := node.Attrs.(tensor.GroupNormAttributes)
		if !ok {
			return errors.New("invalid group norm attributes")
		}
		count, err := elementCount32(node.Shape)
		if err != nil {
			return err
		}
		input, weight, bias := pointers[node.Inputs[0]], pointers[node.Inputs[1]], pointers[node.Inputs[2]]
		channels := uint32(node.Shape.Dims[0])
		tokens := uint32(node.Shape.Dims[1])
		groups := attributes.Groups
		epsilon := attributes.Epsilon
		args := []unsafe.Pointer{
			unsafe.Pointer(&input), unsafe.Pointer(&weight), unsafe.Pointer(&bias), unsafe.Pointer(&output),
			unsafe.Pointer(&channels), unsafe.Pointer(&tokens), unsafe.Pointer(&groups),
			unsafe.Pointer(&epsilon), unsafe.Pointer(&count),
		}
		err = launch1D(state, functions.groupNorm, count, args)
		runtime.KeepAlive(args)
		return err
	case tensor.OpL2Norm:
		attributes, ok := node.Attrs.(tensor.L2NormAttributes)
		if !ok {
			return errors.New("invalid L2Norm attributes")
		}
		width, rows, err := rowDimensions32(node.Shape)
		if err != nil {
			return err
		}
		input := pointers[node.Inputs[0]]
		epsilon := attributes.Epsilon
		args := []unsafe.Pointer{
			unsafe.Pointer(&input),
			unsafe.Pointer(&output),
			unsafe.Pointer(&width),
			unsafe.Pointer(&rows),
			unsafe.Pointer(&epsilon),
		}
		err = launch1D(state, functions.l2Norm, rows, args)
		runtime.KeepAlive(input)
		runtime.KeepAlive(output)
		runtime.KeepAlive(width)
		runtime.KeepAlive(rows)
		runtime.KeepAlive(epsilon)
		return err
	default:
		return fmt.Errorf("unsupported CUDA operation %s", node.Op)
	}
}
