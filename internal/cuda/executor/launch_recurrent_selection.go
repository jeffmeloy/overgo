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

const (
	// sparseAttentionGroupedThreads and sparseAttentionGroupedPairs mirror the
	// grouped kernel's block width and the (head, channel) outputs each of its
	// threads accumulates.
	sparseAttentionGroupedThreads = uint32(256)
	sparseAttentionGroupedPairs   = uint32(16)
	// sparseAttentionGroupedSharedLimit is the shared memory a launch takes
	// without the dynamic opt-in.
	sparseAttentionGroupedSharedLimit = uint64(48 << 10)
	// sparseAttentionGroupedMaxTile bounds the slots a tile stages.
	sparseAttentionGroupedMaxTile = uint32(64)
	// sparseAttentionGroupedHeadWords: a head's maximum and sum.
	sparseAttentionGroupedHeadWords = uint64(2)
	// sparseAttentionGroupedWideRows: a step whose query rows across the
	// key/value heads reach this many fills the card with whole head groups;
	// a shorter step, a decode step, gives each query head a block of its own.
	sparseAttentionGroupedWideRows = uint32(256)
)

const (
	// gatedDeltaNetWarpWidth, gatedDeltaNetRowColumns and gatedDeltaNetBlockRows
	// mirror the row kernel: a warp a state row, a lane at most 8 of the
	// row's columns in registers, 4 rows a block.
	gatedDeltaNetWarpWidth  = uint32(32)
	gatedDeltaNetRowColumns = uint32(8)
	gatedDeltaNetBlockRows  = uint32(4)
)

const (
	// topKBlockThreads mirrors the block-per-row selection's block width.
	topKBlockThreads = uint32(256)
	// topKBlockKeyBytes: a candidate's 64-bit ordering key in shared memory.
	topKBlockKeyBytes = uint32(8)
	// topKBlockMaxCapacity bounds the selected keys a block sorts in shared
	// memory; a larger selection keeps the one-thread kernel.
	topKBlockMaxCapacity = uint32(4096)
)

// topKBlockCapacity reports the keys the block-per-row selection sorts for
// k selected, the power of two at or above k, or false past the bound.
func topKBlockCapacity(k uint32) (uint32, bool) {
	capacity := uint32(tensor.SingletonExtent)
	for capacity < k && capacity < topKBlockMaxCapacity {
		capacity *= tensor.PairedExtent
	}
	return capacity, k <= capacity
}

const (
	// sparseAttentionMMAWidth, sparseAttentionMMARows, sparseAttentionMMACells
	// and sparseAttentionMMAThreads mirror the tensor-core kernel: 256-channel
	// heads, at most 16 query heads a block, 32 slots a chunk, 4 warps.
	sparseAttentionMMAWidth   = uint32(256)
	sparseAttentionMMARows    = uint32(16)
	sparseAttentionMMACells   = uint32(32)
	sparseAttentionMMAThreads = uint32(128)
	// sparseAttentionMMAPad*: the row padding of its query, key, value and
	// score tiles, in floats.
	sparseAttentionMMARowPad   = uint32(4)
	sparseAttentionMMAValuePad = uint32(8)
	// sparseAttentionMMARowWords: a row's maximum, sum and rescale.
	sparseAttentionMMARowWords = uint32(3)
	// sparseAttentionMMASharedBytes: the query, key, value and score tiles,
	// the rows' words and the chunk's slot rows.
	sparseAttentionMMASharedBytes = 4 * (sparseAttentionMMARows*(sparseAttentionMMAWidth+sparseAttentionMMARowPad) +
		sparseAttentionMMACells*(sparseAttentionMMAWidth+sparseAttentionMMARowPad) +
		sparseAttentionMMACells*(sparseAttentionMMAWidth+sparseAttentionMMAValuePad) +
		sparseAttentionMMARows*(sparseAttentionMMACells+sparseAttentionMMARowPad) +
		sparseAttentionMMARowWords*sparseAttentionMMARows + sparseAttentionMMACells)
)

// mmaSparseAttention reports whether a step's selected attention runs on
// tensor cores: 256-channel keys and values, at most 16 query heads a
// key/value head, and a prompt wide enough to fill the card with blocks.
func mmaSparseAttention(keyWidth, valueWidth, queryHeads, keyValueHeads, queryTokens uint32) bool {
	return keyWidth == sparseAttentionMMAWidth && valueWidth == sparseAttentionMMAWidth && keyValueHeads > 0 &&
		queryHeads%keyValueHeads == 0 && queryHeads/keyValueHeads <= sparseAttentionMMARows &&
		queryTokens*keyValueHeads >= sparseAttentionGroupedWideRows
}

const (
	// sparseAttentionSplitThreads, sparseAttentionSplitLanes, sparseAttentionSplitHeads and
	// sparseAttentionSplitKeyLanes mirror the split kernel: 8 warps of 32 lanes, at most
	// 16 query heads a key/value head, at most 8 key channels a lane.
	sparseAttentionSplitThreads  = uint32(256)
	sparseAttentionSplitLanes    = uint32(32)
	sparseAttentionSplitHeads    = uint32(16)
	sparseAttentionSplitKeyLanes = uint32(8)
	// sparseAttentionSplitBlocks is the blocks a step's spans aim to give the
	// card, about two a multiprocessor; a span holds at least
	// sparseAttentionSplitMinSpan slots, a warp's four, and at most
	// sparseAttentionSplitMaxSpan, which bounds the block's scores.
	sparseAttentionSplitBlocks  = uint32(256)
	sparseAttentionSplitMinSpan = uint32(32)
	sparseAttentionSplitMaxSpan = uint32(256)
)

// sparseAttentionSplit sizes a split sparse attention launch: the selected
// slots a block scores, the spans they split into, and the block's shared
// memory, its query heads' queries, their scores and the span's rows.
type sparseAttentionSplit struct {
	span, splits, sharedBytes uint32
}

// splitSparseAttention reports the split kernel's geometry for a step too
// narrow for tensor cores, a decode step's: spans that give the step about
// sparseAttentionSplitBlocks blocks, each of sparseAttentionSplitMinSpan to
// sparseAttentionSplitMaxSpan slots; false where a key/value head's query
// heads or a width exceed the kernel's.
func splitSparseAttention(keyWidth, valueWidth, queryHeads, keyValueHeads, queryTokens, selected uint32) (sparseAttentionSplit, bool) {
	if keyWidth == 0 || valueWidth == 0 || keyValueHeads == 0 || queryTokens == 0 || selected == 0 || queryHeads%keyValueHeads != 0 ||
		queryHeads/keyValueHeads > sparseAttentionSplitHeads || keyWidth > sparseAttentionSplitKeyLanes*sparseAttentionSplitLanes ||
		valueWidth > sparseAttentionSplitThreads {
		return sparseAttentionSplit{}, false
	}
	rows := queryTokens * keyValueHeads
	splits := max(1, (sparseAttentionSplitBlocks+rows-1)/rows)
	span := min(max(sparseAttentionSplitMinSpan, (selected+splits-1)/splits), sparseAttentionSplitMaxSpan)
	group := queryHeads / keyValueHeads
	words := group*keyWidth + group*span + span
	return sparseAttentionSplit{span: span, splits: (selected + span - 1) / span, sharedBytes: words * uint32(f32ScalarBytes)}, true
}

// The inputs of a selected attention node: its queries, keys, values and
// selected slots.
const (
	sparseAttentionQuery = iota
	sparseAttentionKey
	sparseAttentionValue
	sparseAttentionIndices
	sparseAttentionInputs
)

// sparseAttentionTokenAxis is the queries' token axis, after their channels
// and heads.
const sparseAttentionTokenAxis = 2

// sparseSplitPartialBytes is the score staging a selected attention node's
// split launch folds its spans through, and false where the node takes one
// span or another kernel.
func sparseSplitPartialBytes(node *tensor.Tensor) (uint64, bool) {
	if node.Op != tensor.OpSparseAttention || len(node.Inputs) < sparseAttentionInputs {
		return 0, false
	}
	query, key := node.Inputs[sparseAttentionQuery].Shape, node.Inputs[sparseAttentionKey].Shape
	value, indices := node.Inputs[sparseAttentionValue].Shape, node.Inputs[sparseAttentionIndices].Shape
	keyWidth, valueWidth := uint32(query.Dims[0]), uint32(value.Dims[0])
	queryHeads, keyValueHeads := uint32(query.Dims[1]), uint32(key.Dims[1])
	queryTokens, selected := uint32(query.Dims[sparseAttentionTokenAxis]), uint32(indices.Dims[0])
	if mmaSparseAttention(keyWidth, valueWidth, queryHeads, keyValueHeads, queryTokens) {
		return 0, false
	}
	geometry, ok := splitSparseAttention(keyWidth, valueWidth, queryHeads, keyValueHeads, queryTokens, selected)
	if !ok || geometry.splits == 1 {
		return 0, false
	}
	return uint64(queryTokens) * uint64(queryHeads) * uint64(geometry.splits) * (uint64(valueWidth) + attentionDecodeMetaFloats) * f32ScalarBytes, true
}

// sparseAttentionGrouped sizes a grouped sparse attention launch: the query
// heads a block takes, the selected slots each tile stages, and the shared
// memory the block holds.
type sparseAttentionGrouped struct {
	chunkHeads, tile, sharedBytes uint32
}

// groupedSparseAttention reports the grouped kernel's geometry, the widest
// head chunk whose tile of at least one slot fits the shared memory, or false
// when one head's outputs exceed a block's accumulators or one head and one
// slot exceed the shared memory, where the per-channel kernel serves.
func groupedSparseAttention(keyWidth, valueWidth, queryHeads, keyValueHeads, queryTokens uint32) (sparseAttentionGrouped, bool) {
	if keyWidth == 0 || valueWidth == 0 || keyValueHeads == 0 || queryHeads%keyValueHeads != 0 {
		return sparseAttentionGrouped{}, false
	}
	group := queryHeads / keyValueHeads
	chunkHeads := min(group, sparseAttentionGroupedThreads*sparseAttentionGroupedPairs/valueWidth)
	if queryTokens*keyValueHeads < sparseAttentionGroupedWideRows {
		chunkHeads = min(chunkHeads, tensor.SingletonExtent)
	}
	for ; chunkHeads > 0; chunkHeads-- {
		for tile := sparseAttentionGroupedMaxTile; tile > 0; tile /= tensor.PairedExtent {
			heads, slots := uint64(chunkHeads), uint64(tile)
			words := heads*uint64(keyWidth) + slots*uint64(keyWidth+1) + slots*uint64(valueWidth) +
				heads*slots + sparseAttentionGroupedHeadWords*heads + slots
			if bytes := words * f32ScalarBytes; bytes <= sparseAttentionGroupedSharedLimit {
				return sparseAttentionGrouped{chunkHeads: chunkHeads, tile: tile, sharedBytes: uint32(bytes)}, true
			}
		}
	}
	return sparseAttentionGrouped{}, false
}

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
		if size <= gatedDeltaNetRowColumns*gatedDeltaNetWarpWidth {
			return launchGridABI(
				state, functions[kernelGatedDeltaNetRowsF32],
				driver.Dim3{X: count, Y: (size-1)/gatedDeltaNetBlockRows + 1, Z: 1},
				driver.Dim3{X: gatedDeltaNetBlockRows * gatedDeltaNetWarpWidth, Y: 1, Z: 1},
				&query, &key, &value, &gate, &beta, &stateInput, &output,
				&size, &qHeads, &kHeads, &heads, &tokens, &sequences, &gateWidth, &repeatInterleave,
			)
		}
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
		// A block selection reads its first query from the device, so a replay follows the step.
		blockSize, queryStart, outputWidth := attributes.BlockSize, pointers.attribute, uint32(node.Shape.Dims[0])
		if blockSize != 0 && queryStart == 0 {
			return errors.New("block selection first-query storage is unavailable")
		}
		if capacity, ok := topKBlockCapacity(k); ok && rows > 0 {
			return launchGridSharedABI(
				state, functions[kernelTopKBlockF32],
				driver.Dim3{X: rows, Y: 1, Z: 1}, driver.Dim3{X: topKBlockThreads, Y: 1, Z: 1}, capacity*topKBlockKeyBytes,
				&input, &output, &width, &k, &rows, &blockSize, &queryStart, &outputWidth, &capacity,
			)
		}
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
		queryStart := pointers.attribute
		if queryStart == 0 {
			return errors.New("sparse attention first-query storage is unavailable")
		}
		if mmaSparseAttention(keyWidth, valueWidth, queryHeads, keyValueHeads, queryTokens) {
			return launchGridSharedABI(
				state, functions[kernelSparseAttentionMmaF32],
				driver.Dim3{X: queryTokens, Y: keyValueHeads, Z: 1},
				driver.Dim3{X: sparseAttentionMMAThreads, Y: 1, Z: 1}, sparseAttentionMMASharedBytes,
				&query, &key, &value, &indices, &output,
				&queryHeads, &keyValueHeads, &queryTokens, &keyValueTokens, &selected,
				&scale, &causal, &queryStart,
			)
		}
		// A narrower step splits its slots across blocks, whose partials fold through the score staging where
		// it holds them: one block a query head walked 2,048 slots in 2.3 ms of each Flash-Next decode layer.
		if geometry, ok := splitSparseAttention(keyWidth, valueWidth, queryHeads, keyValueHeads, queryTokens, selected); ok {
			rows := uint64(queryTokens) * uint64(queryHeads)
			partialBytes, _ := sparseSplitPartialBytes(node)
			if geometry.splits == 1 || blas != nil && blas.scores != 0 && partialBytes <= blas.scoreBytes {
				span, splits := geometry.span, geometry.splits
				var partials driver.DevicePtr
				if splits > 1 {
					partials = blas.scores
				}
				if err := launchGridSharedABI(
					state, functions[kernelSparseAttentionSplitF32],
					driver.Dim3{X: queryTokens, Y: keyValueHeads, Z: splits},
					driver.Dim3{X: sparseAttentionSplitThreads, Y: 1, Z: 1}, geometry.sharedBytes,
					&query, &key, &value, &indices, &output, &partials, &keyWidth, &valueWidth,
					&queryHeads, &keyValueHeads, &queryTokens, &keyValueTokens, &selected,
					&scale, &causal, &queryStart, &span, &splits,
				); err != nil || splits == 1 {
					return err
				}
				rowCount := uint32(rows)
				return launchGridABI(
					state, functions[kernelAttentionDecodeCombineF32],
					driver.Dim3{X: rowCount, Y: 1, Z: 1}, driver.Dim3{X: attentionDecodeThreads, Y: 1, Z: 1},
					&partials, &output, &valueWidth, &splits, &rowCount,
				)
			}
		}
		if geometry, ok := groupedSparseAttention(keyWidth, valueWidth, queryHeads, keyValueHeads, queryTokens); ok {
			chunkHeads, tile := geometry.chunkHeads, geometry.tile
			grid := driver.Dim3{X: queryTokens, Y: keyValueHeads, Z: (queryHeads/keyValueHeads-1)/chunkHeads + 1}
			return launchGridSharedABI(
				state, functions[kernelSparseAttentionGroupedF32], grid,
				driver.Dim3{X: sparseAttentionGroupedThreads, Y: 1, Z: 1}, geometry.sharedBytes,
				&query, &key, &value, &indices, &output, &keyWidth, &valueWidth,
				&queryHeads, &keyValueHeads, &queryTokens, &keyValueTokens, &selected,
				&scale, &causal, &queryStart, &chunkHeads, &tile,
			)
		}
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
