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
		args := []unsafe.Pointer{
			unsafe.Pointer(&input),
			unsafe.Pointer(&output),
			unsafe.Pointer(&count),
		}
		err = launch1D(state, functions.copy, count, args)
		runtime.KeepAlive(input)
		runtime.KeepAlive(output)
		runtime.KeepAlive(count)
		return err
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
		var relativeBidirectional uint32
		if attributes.RelativeBidirectional {
			relativeBidirectional = 1
		}
		scale := attributes.Scale
		softcap := attributes.Softcap
		maxALiBiBias := attributes.MaxALiBiBias
		var causal uint32
		if attributes.Causal {
			causal = 1
		}
		queryStart := attributes.QueryStart
		window := attributes.Window
		var symmetricWindow uint32
		if attributes.SymmetricWindow {
			symmetricWindow = 1
		} else if attributes.ChunkedWindow {
			symmetricWindow = 2
		}
		args := []unsafe.Pointer{
			unsafe.Pointer(&query),
			unsafe.Pointer(&key),
			unsafe.Pointer(&value),
			unsafe.Pointer(&relativeBias),
			unsafe.Pointer(&sinks),
			unsafe.Pointer(&blockIDs),
			unsafe.Pointer(&output),
			unsafe.Pointer(&keyWidth),
			unsafe.Pointer(&valueWidth),
			unsafe.Pointer(&queryHeads),
			unsafe.Pointer(&keyValueHeads),
			unsafe.Pointer(&queryTokens),
			unsafe.Pointer(&keyValueTokens),
			unsafe.Pointer(&scale),
			unsafe.Pointer(&softcap),
			unsafe.Pointer(&maxALiBiBias),
			unsafe.Pointer(&causal),
			unsafe.Pointer(&queryStart),
			unsafe.Pointer(&window),
			unsafe.Pointer(&symmetricWindow),
			unsafe.Pointer(&relativeBuckets),
			unsafe.Pointer(&relativeBidirectional),
			unsafe.Pointer(&count),
		}
		err = launch1D(state, functions.attention, count, args)
		runtime.KeepAlive(query)
		runtime.KeepAlive(key)
		runtime.KeepAlive(value)
		runtime.KeepAlive(relativeBias)
		runtime.KeepAlive(sinks)
		runtime.KeepAlive(blockIDs)
		runtime.KeepAlive(output)
		runtime.KeepAlive(keyWidth)
		runtime.KeepAlive(valueWidth)
		runtime.KeepAlive(queryHeads)
		runtime.KeepAlive(keyValueHeads)
		runtime.KeepAlive(queryTokens)
		runtime.KeepAlive(keyValueTokens)
		runtime.KeepAlive(scale)
		runtime.KeepAlive(softcap)
		runtime.KeepAlive(maxALiBiBias)
		runtime.KeepAlive(causal)
		runtime.KeepAlive(queryStart)
		runtime.KeepAlive(window)
		runtime.KeepAlive(symmetricWindow)
		runtime.KeepAlive(relativeBuckets)
		runtime.KeepAlive(relativeBidirectional)
		runtime.KeepAlive(count)
		return err
	case tensor.OpConcat:
		attributes, ok := node.Attrs.(tensor.ConcatAttributes)
		if !ok || (attributes.Axis != 0 && attributes.Axis != uint32(node.Shape.Rank-1)) {
			return errors.New("invalid concat attributes")
		}
		count, err := elementCount32(node.Shape)
		if err != nil {
			return err
		}
		leftCount, err := elementCount32(node.Inputs[0].Shape)
		if err != nil {
			return err
		}
		left := pointers[node.Inputs[0]]
		right := pointers[node.Inputs[1]]
		leftWidth, err := uint32Checked(node.Inputs[0].Shape.Dims[0], "concat left width")
		if err != nil {
			return err
		}
		rightWidth, err := uint32Checked(node.Inputs[1].Shape.Dims[0], "concat right width")
		if err != nil {
			return err
		}
		axis := attributes.Axis
		args := []unsafe.Pointer{
			unsafe.Pointer(&left),
			unsafe.Pointer(&right),
			unsafe.Pointer(&output),
			unsafe.Pointer(&leftCount),
			unsafe.Pointer(&leftWidth),
			unsafe.Pointer(&rightWidth),
			unsafe.Pointer(&axis),
			unsafe.Pointer(&count),
		}
		err = launch1D(state, functions.concat, count, args)
		runtime.KeepAlive(left)
		runtime.KeepAlive(right)
		runtime.KeepAlive(output)
		runtime.KeepAlive(leftCount)
		runtime.KeepAlive(leftWidth)
		runtime.KeepAlive(rightWidth)
		runtime.KeepAlive(axis)
		runtime.KeepAlive(count)
		return err
	default:
		return fmt.Errorf("unsupported CUDA operation %s", node.Op)
	}
}
