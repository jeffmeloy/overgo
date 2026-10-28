package executor

import (
	"errors"

	"overgo/internal/cuda/device"
	"overgo/internal/cuda/driver"
	"overgo/internal/tensor"
)

// The QSA attention operation's launch: Strata's QSA layer between its projections (src/core/layer.cpp qsa_layer
// for a token, src/prefill/prefill.cpp's QSA section for a chunk), each step of it the ported kernel's launch in
// launch_strata_qsa.go. The step records are the node's runtime words, staged to the device every execution, so a
// replayed decode graph reads the next token's records. Its inputs are tensor.QSAAttention's, in order.
const (
	qsaInputQuery = iota
	qsaInputKey
	qsaInputValue
	qsaInputIndexerKey
	qsaInputIndexerQuery
	qsaInputIndexerNorm
	qsaInputIndexerTail
	qsaInputIndexerSpare
	qsaInputKeyCodes
	qsaInputValueCodes
	qsaInputKeyScales
	qsaInputValueScales
	qsaInputPageTable
	qsaInputIndexerPooled
	qsaInputBlockPosition
	qsaInputPositions
	qsaInputRopeCos
	qsaInputRopeSin
	// A streamed state's inputs follow (tensor.QSAStream): its host copy, its map, its staging pool.
	qsaInputHostKeyCodes
	qsaInputHostValueCodes
	qsaInputHostKeyScales
	qsaInputHostValueScales
	qsaInputStreamMap
	qsaInputStageKeyCodes
	qsaInputStageValueCodes
	qsaInputStageKeyScales
	qsaInputStageValueScales
	qsaInputStageTable
)

// qsaPositionBase is the indexer's position base, zero as generate.cpp passes it: a cell's positions arrive through
// the image position table, the identity for text.
const qsaPositionBase = 0

// qsaSelectionBatch is the prompt read's queries scored and selected at once (prefill.cpp's sel_batch), which bounds
// the scores' scratch.
const qsaSelectionBatch = int64(256)

// qsaSelectionCapacity is a query's selected-cell capacity, qsa_selection_width at kTopkMaxCells: the budget's whole
// blocks plus an incomplete tail.
var qsaSelectionCapacity = int64(tensor.QSASelectionWidth(strataTopKMaxCells))

// strataTopKMaxCells is qsa.hpp's kTopkMaxCells, the cells the capacity of a selection is taken at.
const strataTopKMaxCells = 32768

// qsaScratch is the operation's scratch past the kernels' own staging: the selected ids of every token, the block
// scores of a selection batch, and the decode attention's partials.
type qsaScratch struct {
	ids, scores, partials, total uint64
}

func qsaScratchLayout(node *tensor.Tensor) (qsaScratch, error) {
	attributes, ok := node.Attrs.(tensor.QSAAttentionAttributes)
	if !ok || len(attributes.Cells) == 0 {
		return qsaScratch{}, errors.New("QSA attention attributes are invalid")
	}
	tokens := int64(len(attributes.Cells))
	maxBlocks := int64(attributes.Capacity)/strataIndexerBlockCells + int64(tensor.QSAPooledSpare)
	var layout qsaScratch
	layout.ids = 0
	layout.scores = alignQSAScratch(uint64(tokens * qsaSelectionCapacity * f32Bytes))
	layout.partials = alignQSAScratch(layout.scores + uint64(min(tokens, qsaSelectionBatch)*maxBlocks*f32Bytes))
	// An in-place window's rows attend as a decode batch, each with partials of its own.
	attending := int64(1)
	if attributes.InPlace {
		attending = tokens
	}
	layout.total = alignQSAScratch(layout.partials + uint64(attending*strataAttentionScratchFloats(qsaSelectionCapacity, tensor.QSAQueryHeads)*f32Bytes))
	return layout, nil
}

// qsaStreamMap is a streamed state's residency map: its page table, and the map input's per-slot arrays then its
// control ints (tensor.QSAStream.Map), the slots the pools' pages.
func qsaStreamMap(node *tensor.Tensor, table, words driver.DevicePtr, capacity int64) strataKVStreamMap {
	pageCells := int64(tensor.QSAKeyValueHeads * tensor.QSABlockCells)
	slots := int64(node.Inputs[qsaInputKeyCodes].Shape.Dims[1]) / pageCells
	array := func(index int64) driver.DevicePtr { return words + driver.DevicePtr(index*slots*i32Bytes) }
	return strataKVStreamMap{
		table: table, slotBlock: array(qsaMapSlotBlock), slotStamp: array(qsaMapSlotStamp), slotRef: array(qsaMapSlotRef),
		missBlock: array(qsaMapMissBlock), missSlot: array(qsaMapMissSlot),
		ctl: array(tensor.QSAStreamSlotArrays), blocks: capacity / tensor.QSABlockCells, slots: slots,
	}
}

// The stream map's per-slot arrays, in tensor.QSAStream.Map's order.
const (
	qsaMapSlotBlock = iota
	qsaMapSlotStamp
	qsaMapSlotRef
	qsaMapMissBlock
	qsaMapMissSlot
)

// i32Bytes is an int32's bytes, the stream map's element.
const i32Bytes = int64(4)

// qsaScratchAlignment keeps each scratch region at a boundary the kernels' vector loads accept.
const qsaScratchAlignment = uint64(256)

func alignQSAScratch(bytes uint64) uint64 {
	return (bytes + qsaScratchAlignment - 1) / qsaScratchAlignment * qsaScratchAlignment
}

// launchQSACommit runs a QSA layer's commit (tensor.Builder.QSACommit), verify.cpp capture_commit's QSA part: the tail
// restored from the window's snapshot, then each kept row's raw key appended at its cell, a row named by a negative cell
// skipped by the kernel; then the first cell's word to the output.
func launchQSACommit(state *device.State, functions functionSet, node *tensor.Tensor, pointers launchPointerFrame) error {
	attributes, ok := node.Attrs.(tensor.QSACommitAttributes)
	if !ok || len(node.Inputs) < tensor.QSACommitOperands {
		return errors.New("QSA commit node is invalid")
	}
	tail, snapshot := pointers.input(tensor.QSACommitTail), pointers.input(tensor.QSACommitSnapshot)
	count := uint32(tensor.QSATailRows * tensor.QSAIndexerWidth)
	if err := launch1DABI(state, functions[kernelCopyF32], count, &snapshot, &tail, &count); err != nil {
		return err
	}
	indexer := strataIndexerBuffers{
		tail: tail, dead: pointers.input(tensor.QSACommitSpare),
		pooled: pointers.input(tensor.QSACommitPooled), blockPos: pointers.input(tensor.QSACommitBlockPosition),
	}
	geometry := strataIndexerGeometry{keyWidth: tensor.QSAIndexerWidth, block: tensor.QSABlockCells, rotated: tensor.QSARotated}
	raw, cells := pointers.input(tensor.QSACommitRaw), pointers.input(tensor.QSACommitCells)
	for row := range int64(node.Inputs[tensor.QSACommitCells].Shape.Dims[0]) {
		if err := launchStrataIndexerKeyAppend(state, functions, raw+driver.DevicePtr(row*tensor.QSAIndexerWidth*f32Bytes),
			cells+driver.DevicePtr(row*i32Bytes), int32(qsaPositionBase), pointers.input(tensor.QSACommitNorm), attributes.Epsilon,
			indexer, geometry, pointers.input(tensor.QSACommitRopeCos), pointers.input(tensor.QSACommitRopeSin),
			pointers.input(tensor.QSACommitPositions)); err != nil {
			return err
		}
	}
	return launchCommitWord(state, functions, cells, pointers.output())
}

// launchQSAAttention runs one QSA step: a token as Strata's decode layer, several consecutive tokens as its prompt
// read.
func launchQSAAttention(
	state *device.State, functions functionSet, blas *blasState, node *tensor.Tensor, runtimeAttributes tensor.Attributes,
	pointers launchPointerFrame,
) error {
	attributes, ok := runtimeAttributes.(tensor.QSAAttentionAttributes)
	// An in-place step's snapshot follows every other input.
	inputs := len(node.Inputs)
	if attributes.InPlace {
		inputs--
	}
	streamed := inputs == tensor.QSAAttentionInputs+tensor.QSAStreamInputs
	if !ok || len(attributes.Cells) == 0 || inputs != tensor.QSAAttentionInputs && !streamed {
		return errors.New("QSA attention node is invalid")
	}
	layout, err := qsaScratchLayout(node)
	if err != nil {
		return err
	}
	if blas == nil || blas.scores == 0 || layout.total > blas.scoreBytes {
		return errors.New("QSA attention scratch is unavailable")
	}
	ids := blas.scores + driver.DevicePtr(layout.ids)
	scores := blas.scores + driver.DevicePtr(layout.scores)
	partials := blas.scores + driver.DevicePtr(layout.partials)
	steps := pointers.attribute
	pools := strataKVPools{
		key: pointers.input(qsaInputKeyCodes), value: pointers.input(qsaInputValueCodes),
		keyScale: pointers.input(qsaInputKeyScales), valueScale: pointers.input(qsaInputValueScales),
	}
	table := pointers.input(qsaInputPageTable)
	tokens := int64(len(attributes.Cells))
	// The fixed state after the step is the output's after the attention: the step copies the tail and spare it read
	// there and appends to the copies, so the fixed state it read stays the one a resumed cache holds.
	output := pointers.output()
	tail := output + driver.DevicePtr(tokens*tensor.QSAQueryHeads*tensor.QSAHeadWidth*f32Bytes)
	dead := tail + driver.DevicePtr(tensor.QSATailRows*tensor.QSAIndexerWidth*f32Bytes)
	copies := []struct {
		source, target driver.DevicePtr
		count          uint32
	}{
		{pointers.input(qsaInputIndexerTail), tail, tensor.QSATailRows * tensor.QSAIndexerWidth},
		{pointers.input(qsaInputIndexerSpare), dead, tensor.QSAIndexerWidth},
	}
	if attributes.InPlace {
		// verify.cpp record_window: the tail's snapshot, then the step's appends to the fixed state where it sits.
		tail, dead = pointers.input(qsaInputIndexerTail), pointers.input(qsaInputIndexerSpare)
		copies = copies[:1]
		copies[0].target = pointers.input(len(node.Inputs) - 1)
	}
	for _, part := range copies {
		if err := launch1DABI(state, functions[kernelCopyF32], part.count, &part.source, &part.target, &part.count); err != nil {
			return err
		}
	}
	indexer := strataIndexerBuffers{
		tail: tail, dead: dead,
		pooled: pointers.input(qsaInputIndexerPooled), blockPos: pointers.input(qsaInputBlockPosition),
	}
	positions := pointers.input(qsaInputPositions)
	geometry := strataKVGeometry{kvHeads: tensor.QSAKeyValueHeads, headDim: tensor.QSAHeadWidth, pageSize: tensor.QSABlockCells}
	maxBlocks := int64(attributes.Capacity)/strataIndexerBlockCells + int64(tensor.QSAPooledSpare)
	keys, values := pointers.input(qsaInputKey), pointers.input(qsaInputValue)
	raw, gamma := pointers.input(qsaInputIndexerKey), pointers.input(qsaInputIndexerNorm)
	indexerQueries, queries := pointers.input(qsaInputIndexerQuery), pointers.input(qsaInputQuery)
	attention := strataAttentionPools{kv: pools, table: table}
	var host, stage strataKVPools
	var stream strataKVStreamMap
	if streamed {
		host = strataKVPools{
			key: pointers.input(qsaInputHostKeyCodes), value: pointers.input(qsaInputHostValueCodes),
			keyScale: pointers.input(qsaInputHostKeyScales), valueScale: pointers.input(qsaInputHostValueScales),
		}
		stage = strataKVPools{
			key: pointers.input(qsaInputStageKeyCodes), value: pointers.input(qsaInputStageValueCodes),
			keyScale: pointers.input(qsaInputStageKeyScales), valueScale: pointers.input(qsaInputStageValueScales),
		}
		stream = qsaStreamMap(node, table, pointers.input(qsaInputStreamMap), int64(attributes.Capacity))
	}
	positionBase := int32(qsaPositionBase)
	if tokens > 1 && attributes.InPlace {
		// verify.cpp record_window's QSA section: a window's rows through the decode kernels - its cells in one launch
		// over their steps (the host copy too when streamed), its indexer keys a row at a time, then each row's scores,
		// selection and attention over the cells up to its own, batched.
		if err := launchStrataKVAppendSteps(state, functions, pools, host, table, steps, keys, values, geometry,
			int32(tokens), tensor.QSAStepWords, tensor.QSAKeyValueHeads*tensor.QSAHeadWidth); err != nil {
			return err
		}
		stepBytes := int64(tensor.QSAStepWords) * f32Bytes
		for row := range tokens {
			if err := launchStrataIndexerKeyAppend(state, functions, raw+driver.DevicePtr(row*tensor.QSAIndexerWidth*f32Bytes),
				steps+driver.DevicePtr(row*stepBytes), positionBase, gamma, attributes.Epsilon, indexer,
				strataIndexerGeometry{keyWidth: tensor.QSAIndexerWidth, block: tensor.QSABlockCells, rotated: tensor.QSARotated},
				pointers.input(qsaInputRopeCos), pointers.input(qsaInputRopeSin), positions); err != nil {
				return err
			}
		}
		if err := launchStrataBlockScores(state, functions, indexer.pooled, indexer.dead, indexerQueries, steps, tokens, maxBlocks, scores, -1); err != nil {
			return err
		}
		if err := launchStrataBlockTopK(state, functions, scores, steps, tokens, maxBlocks, qsaSelectionCapacity, ids, -1); err != nil {
			return err
		}
		if streamed {
			if err := launchStrataKVStreamResolve(state, functions, stream, pools, host, ids, steps, tokens, qsaSelectionCapacity, geometry); err != nil {
				return err
			}
		}
		return launchStrataDecodeAttention(state, functions, queries, attention, ids, steps, qsaSelectionCapacity,
			tensor.QSAKeyValueHeads, tensor.QSABlockCells, partials, output, tokens, true)
	}
	if tokens == 1 {
		// layer.cpp qsa_layer, steps 5 and 8: the cell, the indexer's key, the block scores, the selection, the
		// attention over the selected cells read in place.
		if err := launchStrataKVAppend(state, functions, pools, host, table, steps, keys, values, geometry, true); err != nil {
			return err
		}
		if err := launchStrataIndexerKeyAppend(state, functions, raw, steps, positionBase, gamma, attributes.Epsilon, indexer,
			strataIndexerGeometry{keyWidth: tensor.QSAIndexerWidth, block: tensor.QSABlockCells, rotated: tensor.QSARotated},
			pointers.input(qsaInputRopeCos), pointers.input(qsaInputRopeSin), positions); err != nil {
			return err
		}
		if err := launchStrataBlockScores(state, functions, indexer.pooled, indexer.dead, indexerQueries, steps, 1, maxBlocks, scores, -1); err != nil {
			return err
		}
		if err := launchStrataBlockTopK(state, functions, scores, steps, 1, maxBlocks, qsaSelectionCapacity, ids, -1); err != nil {
			return err
		}
		// A streamed state makes the selected blocks resident before the attention reads them (layer.cpp qsa_kv_resolve).
		if streamed {
			if err := launchStrataKVStreamResolve(state, functions, stream, pools, host, ids, steps, 1, qsaSelectionCapacity, geometry); err != nil {
				return err
			}
		}
		return launchStrataDecodeAttention(state, functions, queries, attention, ids, steps, qsaSelectionCapacity,
			tensor.QSAKeyValueHeads, tensor.QSABlockCells, partials, output, 1, true)
	}
	// prefill.cpp's QSA section: the chunk's cells in one launch, its indexer keys in one batch, then scores and
	// selection a batch of queries at a time (the active blocks the batch's last query's), then the tensor-core
	// prompt attention over every query's selection.
	stepBytes := int64(tensor.QSAStepWords) * f32Bytes
	if streamed {
		// prefill.cpp 2727-2790: the layer's cells before the chunk staged from the host copy, then the chunk's cells
		// into the host copy, the staging pool and the resident slots in one launch; the chunk reads the staging pool.
		firstCell := int64(attributes.Cells[0])
		if err := launchStrataKVStage(state, functions, host, stage, (firstCell+strataIndexerBlockCells-1)/strataIndexerBlockCells, geometry); err != nil {
			return err
		}
		if err := launchStrataPrefillKVAppend(state, functions, keys, values, firstCell, tokens, table, pools, host, stage, geometry); err != nil {
			return err
		}
		attention = strataAttentionPools{kv: stage, table: pointers.input(qsaInputStageTable)}
	} else if err := launchStrataKVAppendSteps(state, functions, pools, strataKVPools{}, table, steps, keys, values, geometry,
		int32(tokens), tensor.QSAStepWords, tensor.QSAKeyValueHeads*tensor.QSAHeadWidth); err != nil {
		return err
	}
	if err := launchStrataNativeIndexerBatch(state, functions, raw, tokens, int64(attributes.Cells[0]), positionBase, gamma,
		attributes.Epsilon, indexer, strataRopeScaling{freqBase: strataFreqBaseDefault, factor: 1, attnFactor: 1}, positions); err != nil {
		return err
	}
	for first := int64(0); first < tokens; first += qsaSelectionBatch {
		count := min(qsaSelectionBatch, tokens-first)
		last := int64(attributes.Cells[first+count-1]) + 1
		active := last/strataIndexerBlockCells + 1
		batchSteps := steps + driver.DevicePtr(first*stepBytes)
		batchQueries := indexerQueries + driver.DevicePtr(first*tensor.QSAIndexerHeads*tensor.QSAIndexerWidth*f32Bytes)
		if err := launchStrataBlockScoresTC(state, functions, indexer.pooled, indexer.dead, batchQueries, batchSteps, count, maxBlocks, scores, active); err != nil {
			return err
		}
		if err := launchStrataBlockTopK(state, functions, scores, batchSteps, count, maxBlocks, qsaSelectionCapacity,
			ids+driver.DevicePtr(first*qsaSelectionCapacity*f32Bytes), active); err != nil {
			return err
		}
	}
	return launchStrataPromptAttention(state, functions, queries, attention, ids, steps, qsaSelectionCapacity,
		tensor.QSAKeyValueHeads, tensor.QSABlockCells, output, tokens)
}
