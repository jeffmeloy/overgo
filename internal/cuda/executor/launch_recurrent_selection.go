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

func launchRecurrentSelection(
	state *device.State,
	functions functionSet,
	blas *blasState,
	node *tensor.Tensor,
	pointers map[*tensor.Tensor]driver.DevicePtr,
	attributePointers map[*tensor.Tensor]driver.DevicePtr,
) error {
	output := pointers[node]
	switch node.Op {
	case tensor.OpSSMConv:
		count, err := elementCount32(node.Shape)
		if err != nil {
			return err
		}
		window, err := uint32Checked(node.Inputs[0].Shape.Dims[0], "SSMConv window")
		if err != nil {
			return err
		}
		channels, err := uint32Checked(node.Shape.Dims[0], "SSMConv channels")
		if err != nil {
			return err
		}
		tokens, err := uint32Checked(node.Shape.Dims[1], "SSMConv tokens")
		if err != nil {
			return err
		}
		input := pointers[node.Inputs[0]]
		weights := pointers[node.Inputs[1]]
		args := []unsafe.Pointer{
			unsafe.Pointer(&input),
			unsafe.Pointer(&weights),
			unsafe.Pointer(&output),
			unsafe.Pointer(&window),
			unsafe.Pointer(&channels),
			unsafe.Pointer(&tokens),
			unsafe.Pointer(&count),
		}
		err = launch1D(state, functions.ssmConv, count, args)
		runtime.KeepAlive(input)
		runtime.KeepAlive(weights)
		runtime.KeepAlive(output)
		runtime.KeepAlive(window)
		runtime.KeepAlive(channels)
		runtime.KeepAlive(tokens)
		runtime.KeepAlive(count)
		return err
	case tensor.OpSSMScan:
		stateWidth, err := uint32Checked(node.Inputs[0].Shape.Dims[0], "SSMScan state width")
		if err != nil {
			return err
		}
		dimension, err := uint32Checked(node.Inputs[0].Shape.Dims[1], "SSMScan head width")
		if err != nil {
			return err
		}
		heads, err := uint32Checked(node.Inputs[0].Shape.Dims[2], "SSMScan heads")
		if err != nil {
			return err
		}
		tokens, err := uint32Checked(node.Inputs[1].Shape.Dims[2], "SSMScan tokens")
		if err != nil {
			return err
		}
		sequences, err := uint32Checked(node.Inputs[0].Shape.Dims[3], "SSMScan sequences")
		if err != nil {
			return err
		}
		groups, err := uint32Checked(node.Inputs[4].Shape.Dims[1], "SSMScan groups")
		if err != nil {
			return err
		}
		if uint64(heads)*uint64(sequences) > uint64(^uint32(0)) {
			return errors.New("SSMScan launch count exceeds uint32")
		}
		aWidth, err := uint32Checked(node.Inputs[3].Shape.Dims[0], "SSMScan A width")
		if err != nil {
			return err
		}
		inputState := pointers[node.Inputs[0]]
		x := pointers[node.Inputs[1]]
		dt := pointers[node.Inputs[2]]
		a := pointers[node.Inputs[3]]
		beta := pointers[node.Inputs[4]]
		c := pointers[node.Inputs[5]]
		args := []unsafe.Pointer{
			unsafe.Pointer(&inputState), unsafe.Pointer(&x), unsafe.Pointer(&dt),
			unsafe.Pointer(&a), unsafe.Pointer(&beta), unsafe.Pointer(&c),
			unsafe.Pointer(&output), unsafe.Pointer(&stateWidth), unsafe.Pointer(&dimension),
			unsafe.Pointer(&heads), unsafe.Pointer(&tokens), unsafe.Pointer(&sequences),
			unsafe.Pointer(&groups), unsafe.Pointer(&aWidth),
		}
		err = launch1D(state, functions.ssmScan, heads*sequences, args)
		runtime.KeepAlive(inputState)
		runtime.KeepAlive(x)
		runtime.KeepAlive(dt)
		runtime.KeepAlive(a)
		runtime.KeepAlive(beta)
		runtime.KeepAlive(c)
		runtime.KeepAlive(output)
		runtime.KeepAlive(stateWidth)
		runtime.KeepAlive(dimension)
		runtime.KeepAlive(heads)
		runtime.KeepAlive(tokens)
		runtime.KeepAlive(sequences)
		runtime.KeepAlive(groups)
		runtime.KeepAlive(aWidth)
		return err
	case tensor.OpGatedDeltaNet:
		attributes := node.Attrs.(tensor.GatedDeltaNetAttributes)
		size, err := uint32Checked(node.Inputs[2].Shape.Dims[0], "GatedDeltaNet state width")
		if err != nil {
			return err
		}
		qHeads, err := uint32Checked(node.Inputs[0].Shape.Dims[1], "GatedDeltaNet Q heads")
		if err != nil {
			return err
		}
		kHeads, err := uint32Checked(node.Inputs[1].Shape.Dims[1], "GatedDeltaNet K heads")
		if err != nil {
			return err
		}
		heads, err := uint32Checked(node.Inputs[2].Shape.Dims[1], "GatedDeltaNet value heads")
		if err != nil {
			return err
		}
		tokens, err := uint32Checked(node.Inputs[2].Shape.Dims[2], "GatedDeltaNet tokens")
		if err != nil {
			return err
		}
		sequences, err := uint32Checked(node.Inputs[2].Shape.Dims[3], "GatedDeltaNet sequences")
		if err != nil {
			return err
		}
		gateWidth, err := uint32Checked(node.Inputs[3].Shape.Dims[0], "GatedDeltaNet gate width")
		if err != nil {
			return err
		}
		if uint64(heads)*uint64(sequences) > uint64(^uint32(0)) {
			return errors.New("GatedDeltaNet launch count exceeds uint32")
		}
		count := heads * sequences
		var repeatInterleave uint32
		if attributes.RepeatInterleave {
			repeatInterleave = 1
		}
		query := pointers[node.Inputs[0]]
		key := pointers[node.Inputs[1]]
		value := pointers[node.Inputs[2]]
		gate := pointers[node.Inputs[3]]
		beta := pointers[node.Inputs[4]]
		stateInput := pointers[node.Inputs[5]]
		args := []unsafe.Pointer{
			unsafe.Pointer(&query),
			unsafe.Pointer(&key),
			unsafe.Pointer(&value),
			unsafe.Pointer(&gate),
			unsafe.Pointer(&beta),
			unsafe.Pointer(&stateInput),
			unsafe.Pointer(&output),
			unsafe.Pointer(&size),
			unsafe.Pointer(&qHeads),
			unsafe.Pointer(&kHeads),
			unsafe.Pointer(&heads),
			unsafe.Pointer(&tokens),
			unsafe.Pointer(&sequences),
			unsafe.Pointer(&gateWidth),
			unsafe.Pointer(&repeatInterleave),
		}
		err = launch1D(state, functions.gatedDeltaNet, count, args)
		runtime.KeepAlive(query)
		runtime.KeepAlive(key)
		runtime.KeepAlive(value)
		runtime.KeepAlive(gate)
		runtime.KeepAlive(beta)
		runtime.KeepAlive(stateInput)
		runtime.KeepAlive(output)
		runtime.KeepAlive(size)
		runtime.KeepAlive(qHeads)
		runtime.KeepAlive(kHeads)
		runtime.KeepAlive(heads)
		runtime.KeepAlive(tokens)
		runtime.KeepAlive(sequences)
		runtime.KeepAlive(gateWidth)
		runtime.KeepAlive(repeatInterleave)
		return err
	case tensor.OpGatedLinearAttention:
		attributes, ok := node.Attrs.(tensor.GatedLinearAttentionAttributes)
		if !ok {
			return errors.New("invalid GatedLinearAttention attributes")
		}
		width, err := uint32Checked(node.Inputs[0].Shape.Dims[0], "GatedLinearAttention width")
		if err != nil {
			return err
		}
		keyHeads, err := uint32Checked(node.Inputs[0].Shape.Dims[1], "GatedLinearAttention key heads")
		if err != nil {
			return err
		}
		heads, err := uint32Checked(node.Inputs[2].Shape.Dims[1], "GatedLinearAttention heads")
		if err != nil {
			return err
		}
		tokens, err := uint32Checked(node.Inputs[0].Shape.Dims[2], "GatedLinearAttention tokens")
		if err != nil {
			return err
		}
		sequences, err := uint32Checked(node.Inputs[0].Shape.Dims[3], "GatedLinearAttention sequences")
		if err != nil {
			return err
		}
		if uint64(heads)*uint64(sequences) > uint64(^uint32(0)) {
			return errors.New("GatedLinearAttention launch count exceeds uint32")
		}
		key := pointers[node.Inputs[0]]
		value := pointers[node.Inputs[1]]
		receptance := pointers[node.Inputs[2]]
		decay := pointers[node.Inputs[3]]
		inputState := pointers[node.Inputs[4]]
		scale := attributes.Scale
		args := []unsafe.Pointer{
			unsafe.Pointer(&key), unsafe.Pointer(&value), unsafe.Pointer(&receptance),
			unsafe.Pointer(&decay), unsafe.Pointer(&inputState), unsafe.Pointer(&output),
			unsafe.Pointer(&width), unsafe.Pointer(&keyHeads), unsafe.Pointer(&heads), unsafe.Pointer(&tokens),
			unsafe.Pointer(&sequences), unsafe.Pointer(&scale),
		}
		err = launch1D(state, functions.gatedLinearAttn, heads*sequences, args)
		runtime.KeepAlive(key)
		runtime.KeepAlive(value)
		runtime.KeepAlive(receptance)
		runtime.KeepAlive(decay)
		runtime.KeepAlive(inputState)
		runtime.KeepAlive(output)
		runtime.KeepAlive(width)
		runtime.KeepAlive(keyHeads)
		runtime.KeepAlive(heads)
		runtime.KeepAlive(tokens)
		runtime.KeepAlive(sequences)
		runtime.KeepAlive(scale)
		return err
	case tensor.OpRWKV6:
		width, err := uint32Checked(node.Inputs[0].Shape.Dims[0], "RWKV6 width")
		if err != nil {
			return err
		}
		heads, err := uint32Checked(node.Inputs[0].Shape.Dims[1], "RWKV6 heads")
		if err != nil {
			return err
		}
		tokens, err := uint32Checked(node.Inputs[0].Shape.Dims[2], "RWKV6 tokens")
		if err != nil {
			return err
		}
		sequences, err := uint32Checked(node.Inputs[0].Shape.Dims[3], "RWKV6 sequences")
		if err != nil {
			return err
		}
		if uint64(heads)*uint64(sequences) > uint64(^uint32(0)) {
			return errors.New("RWKV6 launch count exceeds uint32")
		}
		key := pointers[node.Inputs[0]]
		value := pointers[node.Inputs[1]]
		receptance := pointers[node.Inputs[2]]
		first := pointers[node.Inputs[3]]
		decay := pointers[node.Inputs[4]]
		inputState := pointers[node.Inputs[5]]
		args := []unsafe.Pointer{
			unsafe.Pointer(&key), unsafe.Pointer(&value), unsafe.Pointer(&receptance),
			unsafe.Pointer(&first), unsafe.Pointer(&decay), unsafe.Pointer(&inputState),
			unsafe.Pointer(&output), unsafe.Pointer(&width), unsafe.Pointer(&heads),
			unsafe.Pointer(&tokens), unsafe.Pointer(&sequences),
		}
		err = launch1D(state, functions.rwkv6, heads*sequences, args)
		runtime.KeepAlive(key)
		runtime.KeepAlive(value)
		runtime.KeepAlive(receptance)
		runtime.KeepAlive(first)
		runtime.KeepAlive(decay)
		runtime.KeepAlive(inputState)
		runtime.KeepAlive(output)
		runtime.KeepAlive(width)
		runtime.KeepAlive(heads)
		runtime.KeepAlive(tokens)
		runtime.KeepAlive(sequences)
		return err
	case tensor.OpSumRows:
		width, err := uint32Checked(node.Inputs[0].Shape.Dims[0], "SumRows width")
		if err != nil {
			return err
		}
		elements, err := node.Inputs[0].Shape.Elements()
		if err != nil || elements/uint64(width) > uint64(^uint32(0)) {
			return errors.New("SumRows row count exceeds uint32")
		}
		rows := uint32(elements / uint64(width))
		input := pointers[node.Inputs[0]]
		args := []unsafe.Pointer{
			unsafe.Pointer(&input), unsafe.Pointer(&output), unsafe.Pointer(&width), unsafe.Pointer(&rows),
		}
		err = launch1D(state, functions.sumRows, rows, args)
		runtime.KeepAlive(input)
		runtime.KeepAlive(output)
		runtime.KeepAlive(width)
		runtime.KeepAlive(rows)
		return err
	case tensor.OpFWHT:
		width, rows, err := rowDimensions32(node.Inputs[0].Shape)
		if err != nil {
			return err
		}
		input := pointers[node.Inputs[0]]
		args := []unsafe.Pointer{
			unsafe.Pointer(&input), unsafe.Pointer(&output), unsafe.Pointer(&width), unsafe.Pointer(&rows),
		}
		err = launch1D(state, functions.fwht, rows, args)
		runtime.KeepAlive(args)
		return err
	case tensor.OpTopK:
		attributes, ok := node.Attrs.(tensor.TopKAttributes)
		if !ok || attributes.K == 0 {
			return errors.New("invalid TopK attributes")
		}
		width, rows, err := rowDimensions32(node.Inputs[0].Shape)
		if err != nil {
			return err
		}
		input := pointers[node.Inputs[0]]
		k := attributes.K
		args := []unsafe.Pointer{
			unsafe.Pointer(&input), unsafe.Pointer(&output), unsafe.Pointer(&width),
			unsafe.Pointer(&k), unsafe.Pointer(&rows),
		}
		err = launch1D(state, functions.topK, rows, args)
		runtime.KeepAlive(args)
		return err
	case tensor.OpGatherLast:
		inputNode, indicesNode := node.Inputs[0], node.Inputs[1]
		inputRows, err := uint32Checked(
			inputNode.Shape.Dims[inputNode.Shape.Rank-1], "GatherLast input rows",
		)
		if err != nil {
			return err
		}
		inputElements, err := inputNode.Shape.Elements()
		if err != nil || inputElements%uint64(inputRows) != 0 {
			return errors.New("GatherLast input dimensions are invalid")
		}
		inner, err := uint32Checked(inputElements/uint64(inputRows), "GatherLast inner size")
		if err != nil {
			return err
		}
		indexElements, err := indicesNode.Shape.Elements()
		if err != nil {
			return err
		}
		indexCount, err := uint32Checked(indexElements, "GatherLast index count")
		if err != nil {
			return err
		}
		count, err := elementCount32(node.Shape)
		if err != nil {
			return err
		}
		input, indices := pointers[inputNode], pointers[indicesNode]
		args := []unsafe.Pointer{
			unsafe.Pointer(&input), unsafe.Pointer(&indices), unsafe.Pointer(&output),
			unsafe.Pointer(&inner), unsafe.Pointer(&indexCount), unsafe.Pointer(&inputRows),
			unsafe.Pointer(&count),
		}
		err = launch1D(state, functions.gatherLast, count, args)
		runtime.KeepAlive(args)
		return err
	case tensor.OpSparseAttention:
		attributes, ok := node.Attrs.(tensor.SparseAttentionAttributes)
		if !ok {
			return errors.New("invalid SparseAttention attributes")
		}
		keyWidth, err := uint32Checked(node.Inputs[0].Shape.Dims[0], "SparseAttention key width")
		if err != nil {
			return err
		}
		valueWidth, err := uint32Checked(node.Inputs[2].Shape.Dims[0], "SparseAttention value width")
		if err != nil {
			return err
		}
		queryHeads, err := uint32Checked(node.Inputs[0].Shape.Dims[1], "SparseAttention query heads")
		if err != nil {
			return err
		}
		keyValueHeads, err := uint32Checked(node.Inputs[1].Shape.Dims[1], "SparseAttention KV heads")
		if err != nil {
			return err
		}
		queryTokens, err := uint32Checked(node.Inputs[0].Shape.Dims[2], "SparseAttention query tokens")
		if err != nil {
			return err
		}
		keyValueTokens, err := uint32Checked(node.Inputs[1].Shape.Dims[2], "SparseAttention KV tokens")
		if err != nil {
			return err
		}
		selected, err := uint32Checked(node.Inputs[3].Shape.Dims[0], "SparseAttention selected tokens")
		if err != nil {
			return err
		}
		count, err := elementCount32(node.Shape)
		if err != nil {
			return err
		}
		query, key := pointers[node.Inputs[0]], pointers[node.Inputs[1]]
		value, indices := pointers[node.Inputs[2]], pointers[node.Inputs[3]]
		scale := attributes.Scale
		var causal uint32
		if attributes.Causal {
			causal = 1
		}
		queryStart := attributes.QueryStart
		args := []unsafe.Pointer{
			unsafe.Pointer(&query), unsafe.Pointer(&key), unsafe.Pointer(&value), unsafe.Pointer(&indices),
			unsafe.Pointer(&output), unsafe.Pointer(&keyWidth), unsafe.Pointer(&valueWidth),
			unsafe.Pointer(&queryHeads), unsafe.Pointer(&keyValueHeads), unsafe.Pointer(&queryTokens),
			unsafe.Pointer(&keyValueTokens), unsafe.Pointer(&selected), unsafe.Pointer(&scale),
			unsafe.Pointer(&causal), unsafe.Pointer(&queryStart), unsafe.Pointer(&count),
		}
		err = launch1D(state, functions.sparseAttention, count, args)
		runtime.KeepAlive(args)
		return err
	case tensor.OpIndexerScore:
		attributes, ok := node.Attrs.(tensor.IndexerScoreAttributes)
		if !ok {
			return errors.New("invalid IndexerScore attributes")
		}
		width, err := uint32Checked(node.Inputs[0].Shape.Dims[0], "IndexerScore width")
		if err != nil {
			return err
		}
		heads, err := uint32Checked(node.Inputs[0].Shape.Dims[1], "IndexerScore heads")
		if err != nil {
			return err
		}
		queryTokens, err := uint32Checked(node.Inputs[0].Shape.Dims[2], "IndexerScore query tokens")
		if err != nil {
			return err
		}
		keyTokens, err := uint32Checked(node.Inputs[1].Shape.Dims[2], "IndexerScore key tokens")
		if err != nil {
			return err
		}
		count, err := elementCount32(node.Shape)
		if err != nil {
			return err
		}
		query, key, weights := pointers[node.Inputs[0]], pointers[node.Inputs[1]], pointers[node.Inputs[2]]
		scale, queryStart := attributes.Scale, attributes.QueryStart
		args := []unsafe.Pointer{
			unsafe.Pointer(&query), unsafe.Pointer(&key), unsafe.Pointer(&weights), unsafe.Pointer(&output),
			unsafe.Pointer(&width), unsafe.Pointer(&heads), unsafe.Pointer(&queryTokens),
			unsafe.Pointer(&keyTokens), unsafe.Pointer(&scale), unsafe.Pointer(&queryStart), unsafe.Pointer(&count),
		}
		err = launch1D(state, functions.indexerScore, count, args)
		runtime.KeepAlive(args)
		return err
	case tensor.OpRWKV7:
		width, err := uint32Checked(node.Inputs[0].Shape.Dims[0], "RWKV7 width")
		if err != nil {
			return err
		}
		heads, err := uint32Checked(node.Inputs[0].Shape.Dims[1], "RWKV7 heads")
		if err != nil {
			return err
		}
		tokens, err := uint32Checked(node.Inputs[0].Shape.Dims[2], "RWKV7 tokens")
		if err != nil {
			return err
		}
		sequences, err := uint32Checked(node.Inputs[0].Shape.Dims[3], "RWKV7 sequences")
		if err != nil {
			return err
		}
		if uint64(heads)*uint64(sequences) > uint64(^uint32(0)) {
			return errors.New("RWKV7 launch count exceeds uint32")
		}
		receptance := pointers[node.Inputs[0]]
		decay := pointers[node.Inputs[1]]
		key := pointers[node.Inputs[2]]
		value := pointers[node.Inputs[3]]
		a := pointers[node.Inputs[4]]
		bVector := pointers[node.Inputs[5]]
		inputState := pointers[node.Inputs[6]]
		args := []unsafe.Pointer{
			unsafe.Pointer(&receptance), unsafe.Pointer(&decay), unsafe.Pointer(&key), unsafe.Pointer(&value),
			unsafe.Pointer(&a), unsafe.Pointer(&bVector), unsafe.Pointer(&inputState), unsafe.Pointer(&output),
			unsafe.Pointer(&width), unsafe.Pointer(&heads), unsafe.Pointer(&tokens), unsafe.Pointer(&sequences),
		}
		err = launch1D(state, functions.rwkv7, heads*sequences, args)
		runtime.KeepAlive(receptance)
		runtime.KeepAlive(decay)
		runtime.KeepAlive(key)
		runtime.KeepAlive(value)
		runtime.KeepAlive(a)
		runtime.KeepAlive(bVector)
		runtime.KeepAlive(inputState)
		runtime.KeepAlive(output)
		runtime.KeepAlive(width)
		runtime.KeepAlive(heads)
		runtime.KeepAlive(tokens)
		runtime.KeepAlive(sequences)
		return err
	default:
		return fmt.Errorf("unsupported CUDA operation %s", node.Op)
	}
}
