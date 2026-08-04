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

func launchRoPE(
	state *device.State,
	functions functionSet,
	blas *blasState,
	node *tensor.Tensor,
	pointers map[*tensor.Tensor]driver.DevicePtr,
	attributePointers map[*tensor.Tensor]driver.DevicePtr,
) error {
	output := pointers[node]
	switch node.Op {
	case tensor.OpRoPENeoX, tensor.OpRoPENormal:
		attributes, ok := node.Attrs.(tensor.RoPEAttributes)
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
		input := pointers[node.Inputs[0]]
		var frequencyFactors driver.DevicePtr
		if len(node.Inputs) == 2 {
			frequencyFactors = pointers[node.Inputs[1]]
		}
		positions, ok := attributePointers[node]
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
		args := []unsafe.Pointer{
			unsafe.Pointer(&input),
			unsafe.Pointer(&positions),
			unsafe.Pointer(&frequencyFactors),
			unsafe.Pointer(&output),
			unsafe.Pointer(&width),
			unsafe.Pointer(&heads),
			unsafe.Pointer(&tokens),
			unsafe.Pointer(&rotary),
			unsafe.Pointer(&frequencyBase),
			unsafe.Pointer(&frequencyScale),
			unsafe.Pointer(&originalContext),
			unsafe.Pointer(&extFactor),
			unsafe.Pointer(&attentionFactor),
			unsafe.Pointer(&betaFast),
			unsafe.Pointer(&betaSlow),
			unsafe.Pointer(&count),
		}
		function := functions.ropeNeoX
		if node.Op == tensor.OpRoPENormal {
			function = functions.ropeNormal
		}
		err = launch1D(state, function, count, args)
		runtime.KeepAlive(input)
		runtime.KeepAlive(positions)
		runtime.KeepAlive(frequencyFactors)
		runtime.KeepAlive(output)
		runtime.KeepAlive(width)
		runtime.KeepAlive(heads)
		runtime.KeepAlive(tokens)
		runtime.KeepAlive(rotary)
		runtime.KeepAlive(frequencyBase)
		runtime.KeepAlive(frequencyScale)
		runtime.KeepAlive(originalContext)
		runtime.KeepAlive(extFactor)
		runtime.KeepAlive(attentionFactor)
		runtime.KeepAlive(betaFast)
		runtime.KeepAlive(betaSlow)
		runtime.KeepAlive(count)
		return err
	case tensor.OpRoPEMulti:
		attributes, ok := node.Attrs.(tensor.RoPEMultiAttributes)
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
		input := pointers[node.Inputs[0]]
		positions, ok := attributePointers[node]
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
		args := []unsafe.Pointer{
			unsafe.Pointer(&input),
			unsafe.Pointer(&positions),
			unsafe.Pointer(&output),
			unsafe.Pointer(&width),
			unsafe.Pointer(&heads),
			unsafe.Pointer(&tokens),
			unsafe.Pointer(&rotary),
			unsafe.Pointer(&frequencyBase),
			unsafe.Pointer(&frequencyScale),
			unsafe.Pointer(&sections[0]),
			unsafe.Pointer(&sections[1]),
			unsafe.Pointer(&sections[2]),
			unsafe.Pointer(&sections[3]),
			unsafe.Pointer(&count),
		}
		err = launch1D(state, functions.ropeMulti, count, args)
		runtime.KeepAlive(input)
		runtime.KeepAlive(positions)
		runtime.KeepAlive(output)
		runtime.KeepAlive(width)
		runtime.KeepAlive(heads)
		runtime.KeepAlive(tokens)
		runtime.KeepAlive(rotary)
		runtime.KeepAlive(frequencyBase)
		runtime.KeepAlive(frequencyScale)
		runtime.KeepAlive(sections)
		runtime.KeepAlive(count)
		return err
	default:
		return fmt.Errorf("unsupported CUDA operation %s", node.Op)
	}
}
