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
	node *tensor.Tensor,
	runtimeAttributes tensor.Attributes,
	pointers launchPointerFrame,
	attributePointers devicePointerTable,
) error {
	output := pointers.output()
	switch node.Op {
	case tensor.OpRoPENeoX, tensor.OpRoPENormal:
		attributes, ok := runtimeAttributes.(tensor.RoPEAttributes)
		if !ok {
			return errors.New("invalid RoPE attributes")
		}
		count, err := elementCount32(node.Shape)
		if err != nil {
			return err
		}
		width, err := uint32Checked(node.Shape.Dims[0], "RoPE width")
		if err != nil {
			return err
		}
		heads, err := uint32Checked(node.Shape.Dims[1], "RoPE heads")
		if err != nil {
			return err
		}
		tokens, err := uint32Checked(node.Shape.Dims[2], "RoPE tokens")
		if err != nil {
			return err
		}
		input := pointers.input(0)
		var frequencyFactors driver.DevicePtr
		if len(node.Inputs) == 2 {
			frequencyFactors = pointers.input(1)
		}
		positions, ok := attributePointers.lookup(node)
		if !ok {
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
		attributes, ok := runtimeAttributes.(tensor.RoPEMultiAttributes)
		if !ok {
			return errors.New("invalid rope_multi attributes")
		}
		count, err := elementCount32(node.Shape)
		if err != nil {
			return err
		}
		width, err := uint32Checked(node.Shape.Dims[0], "RoPE multi width")
		if err != nil {
			return err
		}
		heads, err := uint32Checked(node.Shape.Dims[1], "RoPE multi heads")
		if err != nil {
			return err
		}
		tokens, err := uint32Checked(node.Shape.Dims[2], "RoPE multi tokens")
		if err != nil {
			return err
		}
		input := pointers.input(0)
		positions, ok := attributePointers.lookup(node)
		if !ok {
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
		return launch1DABI(
			state, functions[kernelRopeMultiF32], count,
			&input, &positions, &output, &width, &heads, &tokens, &rotary,
			&frequencyBase, &frequencyScale,
			&sections[0], &sections[1], &sections[2], &sections[3], &count,
		)
	default:
		return fmt.Errorf("unsupported CUDA operation %s", node.Op)
	}
}
