package executor

import (
	"errors"
	"fmt"
	"math"

	"overgo/internal/cuda/device"
	"overgo/internal/cuda/driver"
	"overgo/internal/tensor"
	"overgo/internal/tensor/dtype"
)

func launchAttentionLayout(
	state *device.State,
	functions functionSet,
	blas *blasState,
	node *tensor.Tensor,
	runtimeAttributes tensor.Attributes,
	pointers map[*tensor.Tensor]driver.DevicePtr,
	attributePointers map[*tensor.Tensor]driver.DevicePtr,
) error {
	output := pointers[node]
	switch node.Op {
	case tensor.OpReshape:
		count, err := elementCount32(node.Shape)
		if err != nil {
			return err
		}
		input := pointers[node.Inputs[0]]
		return launch1DABI(state, functions[kernelCopyF32], count, &input, &output, &count)
	case tensor.OpAttention:
		attributes, ok := runtimeAttributes.(tensor.AttentionAttributes)
		if !ok {
			return errors.New("invalid attention attributes")
		}
		count, err := elementCount32(node.Shape)
		if err != nil {
			return err
		}
		queryNode := node.Inputs[0]
		keyNode := node.Inputs[1]
		valueNode := node.Inputs[2]
		keyWidth, err := uint32Checked(queryNode.Shape.Dims[0], "attention key width")
		if err != nil {
			return err
		}
		valueWidth, err := uint32Checked(valueNode.Shape.Dims[0], "attention value width")
		if err != nil {
			return err
		}
		queryHeads, err := uint32Checked(queryNode.Shape.Dims[1], "attention query heads")
		if err != nil {
			return err
		}
		keyValueHeads, err := uint32Checked(keyNode.Shape.Dims[1], "attention KV heads")
		if err != nil {
			return err
		}
		queryTokens, err := uint32Checked(queryNode.Shape.Dims[2], "attention query tokens")
		if err != nil {
			return err
		}
		keyCapacityTokens, err := uint32Checked(keyNode.Shape.Dims[2], "attention KV capacity")
		if err != nil {
			return err
		}
		keyValueTokens := keyCapacityTokens
		if attributes.KeyValueTokens != 0 {
			keyValueTokens = attributes.KeyValueTokens
		}
		if keyValueTokens > keyCapacityTokens {
			return errors.New("attention logical KV tokens exceed capacity")
		}
		sequences := uint32(1)
		if queryNode.Shape.Rank == 4 {
			sequences, err = uint32Checked(queryNode.Shape.Dims[3], "attention sequences")
			if err != nil {
				return err
			}
		}
		if sequences > 1 && keyValueTokens != keyCapacityTokens {
			return errors.New("parameterized attention capacity requires one sequence")
		}
		query := pointers[queryNode]
		key := pointers[keyNode]
		value := pointers[valueNode]
		var relativeBias driver.DevicePtr
		var sinks driver.DevicePtr
		var blockIDs driver.DevicePtr
		if len(node.Inputs) == 4 && attributes.HasBlockMask {
			blockIDs = pointers[node.Inputs[3]]
		} else if len(node.Inputs) == 4 && attributes.HasSinks {
			sinks = pointers[node.Inputs[3]]
		} else if len(node.Inputs) == 4 {
			relativeBias = pointers[node.Inputs[3]]
		}
		relativeBuckets := attributes.RelativeBuckets
		relativeBidirectional := kernelBool(attributes.RelativeBidirectional)
		scale := attributes.Scale
		softcap := attributes.Softcap
		maxALiBiBias := attributes.MaxALiBiBias
		causal := kernelBool(attributes.Causal)
		queryStart := attributes.QueryStart
		window := attributes.Window
		var symmetricWindow uint32
		if attributes.SymmetricWindow {
			const symmetricWindowABI = 1
			symmetricWindow = symmetricWindowABI
		} else if attributes.ChunkedWindow {
			const chunkedWindowABI = 2
			symmetricWindow = chunkedWindowABI
		}
		const (
			attentionDecodeThreads       = uint32(256)
			attentionDecodeSharedLimit   = uint64(48 * 1024)
			attentionDecodePartialFloats = uint64(attentionDecodeThreads)
			f32Bytes                     = uint64(4)
		)
		sharedBytes := (uint64(keyValueTokens) + attentionDecodePartialFloats) * f32Bytes
		if queryTokens == 1 && causal != 0 && queryStart+1 == keyValueTokens &&
			relativeBias == 0 && sinks == 0 && blockIDs == 0 && softcap == 0 &&
			maxALiBiBias == 0 && window == 0 && sharedBytes <= attentionDecodeSharedLimit {
			blocks := uint64(queryHeads) * uint64(sequences)
			if blocks > math.MaxUint32 {
				return errors.New("decode attention launch size exceeds uint32")
			}
			return launchGridSharedABI(
				state, functions[kernelAttentionDecodeF32],
				driver.Dim3{X: uint32(blocks), Y: 1, Z: 1},
				driver.Dim3{X: attentionDecodeThreads, Y: 1, Z: 1}, uint32(sharedBytes),
				&query, &key, &value, &output, &keyWidth, &valueWidth,
				&queryHeads, &keyValueHeads, &keyValueTokens, &sequences, &scale,
			)
		}
		const attentionOnlineThreads = uint32(256)
		if valueWidth <= attentionOnlineThreads {
			blocks := uint64(queryHeads) * uint64(queryTokens) * uint64(sequences)
			if blocks > math.MaxUint32 {
				return errors.New("online attention launch size exceeds uint32")
			}
			return launchGridABI(
				state, functions[kernelAttentionOnlineF32],
				driver.Dim3{X: uint32(blocks), Y: 1, Z: 1},
				driver.Dim3{X: attentionOnlineThreads, Y: 1, Z: 1},
				&query, &key, &value, &relativeBias, &sinks, &blockIDs, &output,
				&keyWidth, &valueWidth, &queryHeads, &keyValueHeads, &queryTokens, &keyValueTokens, &sequences,
				&scale, &softcap, &maxALiBiBias, &causal, &queryStart, &window, &symmetricWindow,
				&relativeBuckets, &relativeBidirectional,
			)
		}
		return launch1DABI(
			state, functions[kernelAttentionF32], count,
			&query, &key, &value, &relativeBias, &sinks, &blockIDs, &output,
			&keyWidth, &valueWidth, &queryHeads, &keyValueHeads, &queryTokens, &keyValueTokens, &sequences,
			&scale, &softcap, &maxALiBiBias, &causal, &queryStart, &window, &symmetricWindow,
			&relativeBuckets, &relativeBidirectional, &count,
		)
	case tensor.OpConcat:
		attributes, ok := runtimeAttributes.(tensor.ConcatAttributes)
		if !ok || attributes.Axis >= uint32(node.Shape.Rank) {
			return errors.New("invalid concat attributes")
		}
		count, err := elementCount32(node.Shape)
		if err != nil {
			return err
		}
		left := pointers[node.Inputs[0]]
		right := pointers[node.Inputs[1]]
		if output == left && attributes.Axis+1 == uint32(node.Shape.Rank) {
			leftBytes, sizeErr := node.Inputs[0].Shape.Bytes(dtype.F32)
			if sizeErr != nil || uint64(output) > math.MaxUint64-leftBytes {
				return errors.New("concat append offset overflows")
			}
			rightCount, countErr := elementCount32(node.Inputs[1].Shape)
			if countErr != nil {
				return countErr
			}
			destination := output + driver.DevicePtr(leftBytes)
			return launch1DABI(
				state, functions[kernelCopyF32], rightCount,
				&right, &destination, &rightCount,
			)
		}
		var inner uint64 = 1
		for dimension := uint32(0); dimension < attributes.Axis; dimension++ {
			inner *= node.Shape.Dims[dimension]
		}
		innerSize, err := uint32Checked(inner, "concat inner size")
		if err != nil {
			return err
		}
		leftAxis, err := uint32Checked(node.Inputs[0].Shape.Dims[attributes.Axis], "concat left axis")
		if err != nil {
			return err
		}
		rightAxis, err := uint32Checked(node.Inputs[1].Shape.Dims[attributes.Axis], "concat right axis")
		if err != nil {
			return err
		}
		axis := attributes.Axis
		return launch1DABI(
			state, functions[kernelConcatF32], count,
			&left, &right, &output, &innerSize, &leftAxis, &rightAxis, &axis, &count,
		)
	case tensor.OpCacheAppend:
		attributes, ok := runtimeAttributes.(tensor.CacheAppendAttributes)
		if !ok || attributes.Axis+1 != uint32(node.Shape.Rank) {
			return errors.New("invalid cache append attributes")
		}
		left := pointers[node.Inputs[0]]
		right := pointers[node.Inputs[1]]
		leftCount, err := elementCount32(node.Inputs[0].Shape)
		if err != nil {
			return err
		}
		rightCount, err := elementCount32(node.Inputs[1].Shape)
		if err != nil {
			return err
		}
		outputCount, err := elementCount32(node.Shape)
		if err != nil {
			return err
		}
		if output != left {
			if err := launch1DABI(
				state, functions[kernelCopyF32], leftCount, &left, &output, &leftCount,
			); err != nil {
				return err
			}
		}
		inner := uint64(1)
		for dimension := uint32(0); dimension < attributes.Axis; dimension++ {
			if inner > math.MaxUint64/node.Shape.Dims[dimension] {
				return errors.New("cache append offset overflows")
			}
			inner *= node.Shape.Dims[dimension]
		}
		if inner != 0 && uint64(attributes.Offset) > math.MaxUint64/inner {
			return errors.New("cache append offset overflows")
		}
		offset := uint64(attributes.Offset) * inner
		traits, _ := dtype.F32.Traits()
		if offset > uint64(outputCount) || uint64(rightCount) > uint64(outputCount)-offset ||
			offset > math.MaxUint64/traits.TypeSize {
			return errors.New("cache append range exceeds capacity")
		}
		byteOffset := offset * traits.TypeSize
		if uint64(output) > math.MaxUint64-byteOffset {
			return errors.New("cache append destination overflows")
		}
		destination := output + driver.DevicePtr(byteOffset)
		return launch1DABI(
			state, functions[kernelCopyF32], rightCount, &right, &destination, &rightCount,
		)
	default:
		return fmt.Errorf("unsupported CUDA operation %s", node.Op)
	}
}
