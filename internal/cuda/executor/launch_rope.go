package executor

import (
	"errors"
	"fmt"

	"overgo/internal/cuda/device"
	"overgo/internal/cuda/driver"
	"overgo/internal/tensor"
)

func launchRoPE(
	state *device.State,
	functions functionSet,
	blas *blasState,
	_ *q8InputState,
	node *tensor.Tensor,
	runtimeAttributes tensor.Attributes,
	pointers launchPointerFrame,
) error {
	output := pointers.output()
	switch node.Op {
	case tensor.OpRoPENeoX, tensor.OpRoPENormal:
		attributes := runtimeAttributes.(tensor.RoPEAttributes)
		count := shapeCount32(node.Shape)
		width, heads, tokens := uint32(node.Shape.Dims[0]), uint32(node.Shape.Dims[1]), uint32(node.Shape.Dims[2])
		input := pointers.input(0)
		var frequencyFactors driver.DevicePtr
		if len(node.Inputs) == 2 {
			frequencyFactors = pointers.input(1)
		}
		positions := pointers.attribute
		if positions == 0 {
			return errors.New("RoPE position storage is unavailable")
		}
		rotary := attributes.RotaryDimensions
		frequencyBase := attributes.FrequencyBase
		frequencyScale := attributes.FrequencyScale
		originalContext := attributes.OriginalContext
		extFactor := attributes.ExtFactor
		attentionFactor := attributes.AttentionFactor
		betaFast := attributes.BetaFast
		betaSlow := attributes.BetaSlow
		function := functions[kernelRopeNeoxF32]
		if node.Op == tensor.OpRoPENormal {
			function = functions[kernelRopeNormalF32]
		}
		return launch1DABI(
			state, function, count,
			&input, &positions, &frequencyFactors, &output, &width, &heads, &tokens, &rotary,
			&frequencyBase, &frequencyScale, &originalContext, &extFactor, &attentionFactor,
			&betaFast, &betaSlow, &count,
		)
	case tensor.OpRoPEMulti:
		attributes := runtimeAttributes.(tensor.RoPEMultiAttributes)
		count := shapeCount32(node.Shape)
		width, heads, tokens := uint32(node.Shape.Dims[0]), uint32(node.Shape.Dims[1]), uint32(node.Shape.Dims[2])
		input := pointers.input(0)
		positions := pointers.attribute
		if positions == 0 {
			return errors.New("RoPE multi position storage is unavailable")
		}
		rotary := attributes.RotaryDimensions
		frequencyBase := attributes.FrequencyBase
		frequencyScale := attributes.FrequencyScale
		var sections [4]uint32
		for index, section := range attributes.Sections {
			if section < 0 {
				return errors.New("RoPE multi section count is negative")
			}
			sections[index] = uint32(section)
		}
		interleaved := kernelBool(attributes.InterleavedSections)
		splitHalf := kernelBool(attributes.Layout == tensor.RoPELayoutNeoX)
		if attributes.Layout > tensor.RoPELayoutNeoX {
			return errors.New("RoPE multi layout is invalid")
		}
		return launch1DABI(
			state, functions[kernelRopeMultiF32], count,
			&input, &positions, &output, &width, &heads, &tokens, &rotary,
			&frequencyBase, &frequencyScale,
			&sections[0], &sections[1], &sections[2], &sections[3], &interleaved, &splitHalf, &count,
		)
	default:
		return fmt.Errorf("unsupported CUDA operation %s", node.Op)
	}
}
