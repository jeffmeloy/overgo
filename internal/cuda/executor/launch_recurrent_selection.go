package executor

import (
	"errors"
	"fmt"

	"overgo/internal/cuda/device"
	"overgo/internal/cuda/driver"
	"overgo/internal/tensor"
	"overgo/internal/tensor/dtype"
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
		return launch1DABI(
			state, functions[kernelSsmConvF32], count,
			&input, &weights, &output, &window, &channels, &tokens, &count,
		)
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
		return launch1DABI(
			state, functions[kernelSsmScanF32], heads*sequences,
			&inputState, &x, &dt, &a, &beta, &c, &output, &stateWidth, &dimension,
			&heads, &tokens, &sequences, &groups, &aWidth,
		)
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
		const gatedDeltaNetThreads = uint32(1024)
		repeatInterleave := kernelBool(attributes.RepeatInterleave)
		query := pointers[node.Inputs[0]]
		key := pointers[node.Inputs[1]]
		value := pointers[node.Inputs[2]]
		gate := pointers[node.Inputs[3]]
		beta := pointers[node.Inputs[4]]
		stateInput := pointers[node.Inputs[5]]
		return launchGridABI(
			state, functions[kernelGatedDeltaNetF32],
			driver.Dim3{X: count, Y: 1, Z: 1},
			driver.Dim3{X: gatedDeltaNetThreads, Y: 1, Z: 1},
			&query, &key, &value, &gate, &beta, &stateInput, &output,
			&size, &qHeads, &kHeads, &heads, &tokens, &sequences, &gateWidth, &repeatInterleave,
		)
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
		return launch1DABI(
			state, functions[kernelGatedLinearAttentionF32], heads*sequences,
			&key, &value, &receptance, &decay, &inputState, &output,
			&width, &keyHeads, &heads, &tokens, &sequences, &scale,
		)
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
		return launch1DABI(
			state, functions[kernelRwkv6F32], heads*sequences,
			&key, &value, &receptance, &first, &decay, &inputState, &output,
			&width, &heads, &tokens, &sequences,
		)
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
		return launch1DABI(state, functions[kernelSumRowsF32], rows, &input, &output, &width, &rows)
	case tensor.OpFWHT:
		width, rows, err := rowDimensions32(node.Inputs[0].Shape)
		if err != nil {
			return err
		}
		input := pointers[node.Inputs[0]]
		return launch1DABI(state, functions[kernelFwhtF32], rows, &input, &output, &width, &rows)
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
		if k == 1 {
			return launchGridABI(
				state, functions[kernelArgmaxF32],
				driver.Dim3{X: rows, Y: 1, Z: 1},
				driver.Dim3{X: 256, Y: 1, Z: 1},
				&input, &output, &width, &rows,
			)
		}
		return launch1DABI(state, functions[kernelTopKF32], rows, &input, &output, &width, &k, &rows)
	case tensor.OpTopKPairs:
		attributes, ok := node.Attrs.(tensor.TopKAttributes)
		if !ok || attributes.K == 0 {
			return errors.New("invalid TopKPairs attributes")
		}
		chunks, err := uint32Checked(node.Inputs[0].Shape.Dims[2], "TopKPairs chunk count")
		if err != nil {
			return err
		}
		outputElements, err := node.Shape.Elements()
		if err != nil || outputElements%(uint64(attributes.K)*2) != 0 {
			return errors.New("TopKPairs output dimensions are invalid")
		}
		rows := uint32(outputElements / (uint64(attributes.K) * 2))
		input := pointers[node.Inputs[0]]
		k := attributes.K
		candidates := k * chunks
		return launchGridABI(
			state, functions[kernelTopKPairsF32],
			driver.Dim3{X: rows, Y: 1, Z: 1}, driver.Dim3{X: 256, Y: 1, Z: 1},
			&input, &output, &candidates, &k, &rows,
		)
	case tensor.OpTopKPartials:
		attributes, ok := node.Attrs.(tensor.TopKAttributes)
		if !ok || attributes.K == 0 || attributes.Chunk == 0 {
			return errors.New("invalid TopKPartials attributes")
		}
		width, rows, err := rowDimensions32(node.Inputs[0].Shape)
		if err != nil {
			return err
		}
		chunks, err := uint32Checked(node.Shape.Dims[2], "TopKPartials chunk count")
		if err != nil || uint64(chunks)*uint64(rows) > uint64(^uint32(0)) {
			return errors.New("TopKPartials launch count exceeds uint32")
		}
		input := pointers[node.Inputs[0]]
		k, chunk := attributes.K, attributes.Chunk
		return launchGridABI(
			state, functions[kernelTopKPartialsF32],
			driver.Dim3{X: chunks * rows, Y: 1, Z: 1},
			driver.Dim3{X: 32, Y: 1, Z: 1},
			&input, &output, &width, &k, &chunk, &chunks, &rows,
		)
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
		function := functions[kernelGatherLastF32]
		if inputNode.Type == dtype.Q8_0 {
			traits, _ := inputNode.Type.Traits()
			if uint64(inner)%traits.BlockSize != 0 {
				return errors.New("Q8_0 GatherLast inner size is not block aligned")
			}
			function = functions[kernelGatherLastQ80F32]
		}
		return launch1DABI(
			state, function, count,
			&input, &indices, &output, &inner, &indexCount, &inputRows, &count,
		)
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
		causal := kernelBool(attributes.Causal)
		queryStart := attributes.QueryStart
		return launch1DABI(
			state, functions[kernelSparseAttentionF32], count,
			&query, &key, &value, &indices, &output, &keyWidth, &valueWidth,
			&queryHeads, &keyValueHeads, &queryTokens, &keyValueTokens, &selected,
			&scale, &causal, &queryStart, &count,
		)
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
		return launch1DABI(
			state, functions[kernelIndexerScoreF32], count,
			&query, &key, &weights, &output, &width, &heads, &queryTokens,
			&keyTokens, &scale, &queryStart, &count,
		)
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
		return launch1DABI(
			state, functions[kernelRwkv7F32], heads*sequences,
			&receptance, &decay, &key, &value, &a, &bVector, &inputState, &output,
			&width, &heads, &tokens, &sequences,
		)
	default:
		return fmt.Errorf("unsupported CUDA operation %s", node.Op)
	}
}
