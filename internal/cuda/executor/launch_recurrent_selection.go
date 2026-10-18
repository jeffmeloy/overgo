package executor

import (
	"errors"
	"fmt"

	"overgo/internal/cuda/device"
	"overgo/internal/cuda/driver"
	"overgo/internal/cuda/kernel"
	"overgo/internal/tensor"
	"overgo/internal/tensor/dtype"
)

const topKPairWidth = 2

func launchRecurrentSelection(
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
	case tensor.OpSSMConv:
		count := shapeCount32(node.Shape)
		window := uint32(node.Inputs[0].Shape.Dims[0])
		channels, tokens := uint32(node.Shape.Dims[0]), uint32(node.Shape.Dims[1])
		input := pointers.input(0)
		weights := pointers.input(1)
		return launch1DABI(
			state, functions[kernelSsmConvF32], count,
			&input, &weights, &output, &window, &channels, &tokens, &count,
		)
	case tensor.OpSSMScan:
		stateDimensions := shapeDimensions32(node.Inputs[0].Shape)
		stateWidth, dimension, heads, sequences := stateDimensions[0], stateDimensions[1], stateDimensions[2], stateDimensions[3]
		tokens := uint32(node.Inputs[1].Shape.Dims[2])
		groups := uint32(node.Inputs[4].Shape.Dims[1])
		if uint64(heads)*uint64(sequences) > uint64(^uint32(0)) {
			return errors.New("SSMScan launch count exceeds uint32")
		}
		aWidth := uint32(node.Inputs[3].Shape.Dims[0])
		inputState := pointers.input(0)
		x := pointers.input(1)
		dt := pointers.input(2)
		a := pointers.input(3)
		beta := pointers.input(4)
		c := pointers.input(5)
		return launch1DABI(
			state, functions[kernelSsmScanF32], heads*sequences,
			&inputState, &x, &dt, &a, &beta, &c, &output, &stateWidth, &dimension,
			&heads, &tokens, &sequences, &groups, &aWidth,
		)
	case tensor.OpGatedDeltaNet:
		attributes := node.Attrs.(tensor.GatedDeltaNetAttributes)
		valueDimensions := shapeDimensions32(node.Inputs[2].Shape)
		size, heads, tokens, sequences := valueDimensions[0], valueDimensions[1], valueDimensions[2], valueDimensions[3]
		qHeads, kHeads := uint32(node.Inputs[0].Shape.Dims[1]), uint32(node.Inputs[1].Shape.Dims[1])
		gateWidth := uint32(node.Inputs[3].Shape.Dims[0])
		if uint64(heads)*uint64(sequences) > uint64(^uint32(0)) {
			return errors.New("GatedDeltaNet launch count exceeds uint32")
		}
		count := heads * sequences
		const gatedDeltaNetThreads = uint32(1024)
		repeatInterleave := kernelBool(attributes.RepeatInterleave)
		query := pointers.input(0)
		key := pointers.input(1)
		value := pointers.input(2)
		gate := pointers.input(3)
		beta := pointers.input(4)
		stateInput := pointers.input(5)
		return launchGridABI(
			state, functions[kernelGatedDeltaNetF32],
			driver.Dim3{X: count, Y: 1, Z: 1},
			driver.Dim3{X: gatedDeltaNetThreads, Y: 1, Z: 1},
			&query, &key, &value, &gate, &beta, &stateInput, &output,
			&size, &qHeads, &kHeads, &heads, &tokens, &sequences, &gateWidth, &repeatInterleave,
		)
	case tensor.OpGatedLinearAttention:
		keyDimensions := shapeDimensions32(node.Inputs[0].Shape)
		width, keyHeads, tokens, sequences := keyDimensions[0], keyDimensions[1], keyDimensions[2], keyDimensions[3]
		heads := uint32(node.Inputs[2].Shape.Dims[1])
		if uint64(heads)*uint64(sequences) > uint64(^uint32(0)) {
			return errors.New("GatedLinearAttention launch count exceeds uint32")
		}
		key := pointers.input(0)
		value := pointers.input(1)
		receptance := pointers.input(2)
		decay := pointers.input(3)
		inputState := pointers.input(4)
		scale := node.Attrs.(tensor.GatedLinearAttentionAttributes).Scale
		return launch1DABI(
			state, functions[kernelGatedLinearAttentionF32], heads*sequences,
			&key, &value, &receptance, &decay, &inputState, &output,
			&width, &keyHeads, &heads, &tokens, &sequences, &scale,
		)
	case tensor.OpWKV6:
		dimensions := shapeDimensions32(node.Inputs[0].Shape)
		width, heads, tokens, sequences := dimensions[0], dimensions[1], dimensions[2], dimensions[3]
		if uint64(heads)*uint64(sequences) > uint64(^uint32(0)) {
			return errors.New("WKV6 launch count exceeds uint32")
		}
		key := pointers.input(0)
		value := pointers.input(1)
		receptance := pointers.input(2)
		first := pointers.input(3)
		decay := pointers.input(4)
		inputState := pointers.input(5)
		return launch1DABI(
			state, functions[kernelRwkv6F32], heads*sequences,
			&key, &value, &receptance, &first, &decay, &inputState, &output,
			&width, &heads, &tokens, &sequences,
		)
	case tensor.OpSumRows:
		width := uint32(node.Inputs[0].Shape.Dims[0])
		elements, err := node.Inputs[0].Shape.Elements()
		if err != nil || elements/uint64(width) > uint64(^uint32(0)) {
			return errors.New("SumRows row count exceeds uint32")
		}
		rows := uint32(elements / uint64(width))
		input := pointers.input(0)
		return launch1DABI(state, functions[kernelSumRowsF32], rows, &input, &output, &width, &rows)
	case tensor.OpFWHT:
		width, rows := rowWords(node.Inputs[0].Shape)
		input := pointers.input(0)
		return launch1DABI(state, functions[kernelFwhtF32], rows, &input, &output, &width, &rows)
	case tensor.OpTopK:
		attributes := runtimeAttributes.(tensor.TopKAttributes)
		if attributes.K == 0 {
			return errors.New("invalid TopK attributes")
		}
		width, rows := rowWords(node.Inputs[0].Shape)
		input := pointers.input(0)
		k := attributes.K
		if k == 1 && attributes.BlockSize == 0 {
			return launchGridABI(
				state, functions[kernelArgmaxF32],
				kernel.Grid1D(int(rows)),
				kernel.DefaultBlock1D(),
				&input, &output, &width, &rows,
			)
		}
		blockSize, queryStart, outputWidth := attributes.BlockSize, attributes.QueryStart, uint32(node.Shape.Dims[0])
		return launch1DABI(state, functions[kernelTopKF32], rows, &input, &output, &width, &k, &rows, &blockSize, &queryStart, &outputWidth)
	case tensor.OpTopKPairs:
		attributes := node.Attrs.(tensor.TopKAttributes)
		if attributes.K == 0 {
			return errors.New("invalid TopKPairs attributes")
		}
		chunks := uint32(node.Inputs[0].Shape.Dims[2])
		outputElements, err := node.Shape.Elements()
		if err != nil || outputElements%(uint64(attributes.K)*topKPairWidth) != 0 {
			return errors.New("TopKPairs output dimensions are invalid")
		}
		rows := uint32(outputElements / (uint64(attributes.K) * topKPairWidth))
		input := pointers.input(0)
		k := attributes.K
		candidates := k * chunks
		return launchGridABI(
			state, functions[kernelTopKPairsF32],
			kernel.Grid1D(int(rows)), kernel.DefaultBlock1D(),
			&input, &output, &candidates, &k, &rows,
		)
	case tensor.OpTopKPartials:
		attributes := node.Attrs.(tensor.TopKAttributes)
		if attributes.K == 0 || attributes.Chunk == 0 {
			return errors.New("invalid TopKPartials attributes")
		}
		width, rows := rowWords(node.Inputs[0].Shape)
		chunks := uint32(node.Shape.Dims[2])
		if uint64(chunks)*uint64(rows) > uint64(^uint32(0)) {
			return errors.New("TopKPartials launch count exceeds uint32")
		}
		input := pointers.input(0)
		k, chunk := attributes.K, attributes.Chunk
		return launchGridABI(
			state, functions[kernelTopKPartialsF32],
			kernel.Grid1D(int(chunks*rows)),
			kernel.WarpBlock1D(),
			&input, &output, &width, &k, &chunk, &chunks, &rows,
		)
	case tensor.OpGatherLast:
		inputNode, indicesNode := node.Inputs[0], node.Inputs[1]
		inputRows := uint32(inputNode.Shape.Dims[inputNode.Shape.Rank-1])
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
		count := shapeCount32(node.Shape)
		input, indices := pointers.input(0), pointers.input(1)
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
		attributes := runtimeAttributes.(tensor.SparseAttentionAttributes)
		keyWidth, valueWidth := uint32(node.Inputs[0].Shape.Dims[0]), uint32(node.Inputs[2].Shape.Dims[0])
		queryHeads, keyValueHeads := uint32(node.Inputs[0].Shape.Dims[1]), uint32(node.Inputs[1].Shape.Dims[1])
		queryTokens, keyValueTokens := uint32(node.Inputs[0].Shape.Dims[2]), uint32(node.Inputs[1].Shape.Dims[2])
		selected := uint32(node.Inputs[3].Shape.Dims[0])
		count := shapeCount32(node.Shape)
		query, key := pointers.input(0), pointers.input(1)
		value, indices := pointers.input(2), pointers.input(3)
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
		attributes := node.Attrs.(tensor.IndexerScoreAttributes)
		width, heads := uint32(node.Inputs[0].Shape.Dims[0]), uint32(node.Inputs[0].Shape.Dims[1])
		queryTokens, keyTokens := uint32(node.Inputs[0].Shape.Dims[2]), uint32(node.Inputs[1].Shape.Dims[2])
		count := shapeCount32(node.Shape)
		query, key, weights := pointers.input(0), pointers.input(1), pointers.input(2)
		scale, queryStart := attributes.Scale, attributes.QueryStart
		return launch1DABI(
			state, functions[kernelIndexerScoreF32], count,
			&query, &key, &weights, &output, &width, &heads, &queryTokens,
			&keyTokens, &scale, &queryStart, &count,
		)
	case tensor.OpWKV7:
		dimensions := shapeDimensions32(node.Inputs[0].Shape)
		width, heads, tokens, sequences := dimensions[0], dimensions[1], dimensions[2], dimensions[3]
		if uint64(heads)*uint64(sequences) > uint64(^uint32(0)) {
			return errors.New("WKV7 launch count exceeds uint32")
		}
		receptance := pointers.input(0)
		decay := pointers.input(1)
		key := pointers.input(2)
		value := pointers.input(3)
		a := pointers.input(4)
		bVector := pointers.input(5)
		inputState := pointers.input(6)
		return launch1DABI(
			state, functions[kernelRwkv7F32], heads*sequences,
			&receptance, &decay, &key, &value, &a, &bVector, &inputState, &output,
			&width, &heads, &tokens, &sequences,
		)
	default:
		return fmt.Errorf("unsupported CUDA operation %s", node.Op)
	}
}
