package executor

import (
	"errors"
	"fmt"

	"llamacpp2go/internal/cuda/device"
	"llamacpp2go/internal/cuda/driver"
	"llamacpp2go/internal/tensor"
)

func launchAttentionLayout(
	state *device.State,
	functions functionSet,
	blas *blasState,
	node *tensor.Tensor,
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
		return launch1DABI(state, functions.copy, count, &input, &output, &count)
	case tensor.OpAttention:
		attributes, ok := node.Attrs.(tensor.AttentionAttributes)
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
		keyValueTokens, err := uint32Checked(keyNode.Shape.Dims[2], "attention KV tokens")
		if err != nil {
			return err
		}
		sequences := uint32(1)
		if queryNode.Shape.Rank == 4 {
			sequences, err = uint32Checked(queryNode.Shape.Dims[3], "attention sequences")
			if err != nil {
				return err
			}
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
		return launch1DABI(
			state, functions.attention, count,
			&query, &key, &value, &relativeBias, &sinks, &blockIDs, &output,
			&keyWidth, &valueWidth, &queryHeads, &keyValueHeads, &queryTokens, &keyValueTokens, &sequences,
			&scale, &softcap, &maxALiBiBias, &causal, &queryStart, &window, &symmetricWindow,
			&relativeBuckets, &relativeBidirectional, &count,
		)
	case tensor.OpConcat:
		attributes, ok := node.Attrs.(tensor.ConcatAttributes)
		if !ok || attributes.Axis >= uint32(node.Shape.Rank) {
			return errors.New("invalid concat attributes")
		}
		count, err := elementCount32(node.Shape)
		if err != nil {
			return err
		}
		left := pointers[node.Inputs[0]]
		right := pointers[node.Inputs[1]]
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
			state, functions.concat, count,
			&left, &right, &output, &innerSize, &leftAxis, &rightAxis, &axis, &count,
		)
	default:
		return fmt.Errorf("unsupported CUDA operation %s", node.Op)
	}
}
