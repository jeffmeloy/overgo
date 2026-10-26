package tensor

import (
	"errors"
	"math"
	"slices"

	"overgo/internal/tensor/dtype"
)

// QSA attention: Strata's QSA layer between its projections (kernels/cuda/strata_qsa.cu, ported from Strata at
// 296b59d2, MIT), as one operation over a per-sequence state it appends to in place. Its inputs are the layer's normed
// and rotated query and key, its value, its raw indexer key and its normed and rotated indexer query for the step's
// tokens, the indexer key's norm weight, the indexer's fixed state before the step (its tail and its spare key, which
// Strata's conversation snapshot keeps beside the cells: src/core/conversation_snapshot.cpp), and the state's cells.
// Its output packs the attention before the gate with the fixed state after the step, so the state before the step
// stays what it was for a cache that resumes from it, as a gated-delta scan packs its outputs with its state.
// A step of one token appends the cell, appends the indexer key and pools a completed block, scores the blocks,
// selects the cells and attends over them, as Strata's decode layer does; a step of several consecutive tokens does
// the same as Strata's prompt read does a chunk. The state's geometry is the one Strata's kernels are compiled for:
// 24 query heads over 2 key/value heads of 256 values, an indexer of 4 heads of 128 values pooling blocks of 4 cells
// with 64 rotated dims, 2,048 cells of budget.

// The geometry Strata's QSA kernels are compiled for (core/geometry.hpp), which the builder holds its inputs to.
const (
	// QSAQueryHeads is the layer's query heads.
	QSAQueryHeads = 24
	// QSAKeyValueHeads is the layer's K/V heads, each read by 12 query heads.
	QSAKeyValueHeads = 2
	// QSAHeadWidth is a head's values, the query's, the key's and the value's.
	QSAHeadWidth = 256
	// QSAIndexerHeads is the indexer's query heads.
	QSAIndexerHeads = 4
	// QSAIndexerWidth is an indexer head's values and a pooled key's.
	QSAIndexerWidth = 128
	// QSABlockCells is an indexer block's cells and a page's: the indexer pools a block's keys into one.
	QSABlockCells = 4
	// QSARotated is the indexer key's rotated dims.
	QSARotated = 64
	// QSABudget is the cells a query selects past its incomplete block.
	QSABudget = 2048
	// QSAScaleGroup is a scale's values: one fp16 scale a 64-value group of a cell's head.
	QSAScaleGroup = 64
	// QSATailRows is the indexer's raw tail, a block's cells before the one that completes it.
	QSATailRows = QSABlockCells - 1
	// QSAPooledSpare is the pooled rows past the completed blocks: the spare slot and one more.
	QSAPooledSpare = 2
	// QSAPositionAxes is an image position table row's axes: a cell's temporal, height and width positions.
	QSAPositionAxes = 3
	// qsaFixedValues is the fixed state's values the output packs after the attention: the tail's rows and the spare.
	qsaFixedValues = (QSATailRows + 1) * QSAIndexerWidth
	// qsaMaxCapacity bounds a state's cells, whose indices the kernels hold in int32.
	qsaMaxCapacity = 1 << 31
)

// QSAState is one layer's cells for one sequence, at a capacity of cells, which a step appends to in place: the int8
// K/V codes and their fp16 scales in 4-cell pages `[page][kv_head][4][256]` behind a page table, the indexer's pooled
// rows and the last completed block's position (which no kernel reads; Strata's conversation file keeps it), the
// image position table, and the rotation table the pooling reads. Every row a step writes is a row of cells or blocks
// past the ones it reads, so the state holds any shorter prefix of the cells it holds.
type QSAState struct {
	KeyCodes, ValueCodes   *Tensor // I8 (256, 2 x capacity)
	KeyScales, ValueScales *Tensor // F16 (4, 2 x capacity)
	PageTable              *Tensor // I32 (capacity / 4)
	IndexerPooled          *Tensor // F32 (128, capacity / 4 + 2)
	BlockPosition          *Tensor // I32 (1)
	// Positions is Strata's image position table (src/kernels/mrope.hpp): a cell's temporal, height and width
	// positions, the cell's index three times for text, which the pooling rotates a block's key by.
	Positions        *Tensor // I32 (3, capacity)
	RopeCos, RopeSin *Tensor // F32 (32, capacity): the angles of positions 0 .. capacity - 1
}

// QSAAttentionAttributes are a QSA step's tokens: the cells they occupy (consecutive, from the state's first cell),
// the state's capacity and the indexer key norm's epsilon. Each token's step record - its cell, the cells then held,
// the blocks completed and the selection's width - is staged to the device as the node's runtime words, as Strata's
// step state is written once a token.
type QSAAttentionAttributes struct {
	Cells    []uint32
	Capacity uint32
	Epsilon  float32
}

func (QSAAttentionAttributes) validFor(op Op) bool { return sameOp(op, OpQSAAttention) }

// QSARopeBase is the indexer rotation's base the kernels are compiled for (qsa_freq_base).
const QSARopeBase = 1e7

// QSARopeTable is rope.cu's build_rope_table for the indexer's rotation, unscaled: cos and sin of position p and pair
// i at `[p][32]`, each angle p * 1e7^(-2i / 64) in float64, then rounded to float32 (QSAState.RopeCos, RopeSin).
func QSARopeTable(positions int) (cosines, sines []float32) {
	half := QSARotated / PairedExtent
	cosines, sines = make([]float32, positions*half), make([]float32, positions*half)
	for position := range positions {
		for pair := range half {
			inverse := math.Pow(QSARopeBase, -PairedExtent*float64(pair)/float64(QSARotated))
			angle := float64(position) * inverse
			cosines[position*half+pair] = float32(math.Cos(angle))
			sines[position*half+pair] = float32(math.Sin(angle))
		}
	}
	return cosines, sines
}

// QSAStepWords is a step record's words: pos, n_kv, n_bid, width (qsa.hpp's kStepCount).
const QSAStepWords = 4

// QSASelectionWidth is qsa_selection_width: the cells a selection returns once the context holds cells: every cell
// up to the budget's whole blocks plus an incomplete tail.
func QSASelectionWidth(cells uint32) uint32 {
	return min(cells, QSABudget+QSABlockCells-1)
}

// steps are the attributes' step records.
func (a QSAAttentionAttributes) steps() []uint32 {
	words := make([]uint32, 0, QSAStepWords*len(a.Cells))
	for _, cell := range a.Cells {
		held := cell + 1
		words = append(words, cell, held, held/QSABlockCells, QSASelectionWidth(held))
	}
	return words
}

// QSAAttentionInputs is the operation's input count: the six step inputs, the fixed state's two and the state's
// ten tensors, in QSAAttention's order.
const QSAAttentionInputs = 18

// QSAAttentionOutputValues is the packed output's values for tokens tokens: the attention, then the tail's rows and
// the spare after the step.
func QSAAttentionOutputValues(tokens uint64) uint64 {
	return QSAHeadWidth*QSAQueryHeads*tokens + qsaFixedValues
}

// QSAAttention builds one QSA step over the state. query is (256, 24, tokens), key and value (256, 2, tokens),
// indexerKey (128, 1, tokens), indexerQuery (128, 4, tokens), indexerNorm (128), tail (128, 3) and spare (128); the
// output packs the attention (256, 24, tokens) with the tail and spare after the step (QSAAttentionParts).
func (b *Builder) QSAAttention(
	query, key, value, indexerKey, indexerQuery, indexerNorm, tail, spare *Tensor, state QSAState,
	attributes QSAAttentionAttributes,
) *Tensor {
	if b == nil || b.err != nil {
		return nil
	}
	tokens := uint64(len(attributes.Cells))
	if err := validateQSAAttention(query, key, value, indexerKey, indexerQuery, indexerNorm, tail, spare, state, attributes); err != nil {
		b.setError(err)
		return nil
	}
	shape, err := NewShape(QSAAttentionOutputValues(tokens))
	if err != nil {
		b.setError(err)
		return nil
	}
	attributes.Cells = slices.Clone(attributes.Cells)
	return b.add("", dtype.F32, shape, OpQSAAttention, []*Tensor{
		query, key, value, indexerKey, indexerQuery, indexerNorm, tail, spare,
		state.KeyCodes, state.ValueCodes, state.KeyScales, state.ValueScales, state.PageTable,
		state.IndexerPooled, state.BlockPosition, state.Positions, state.RopeCos, state.RopeSin,
	}, attributes)
}

// QSAAttentionParts are a QSA step's packed output's parts: the attention (256, 24, tokens), the tail (128, 3) and
// the spare (128) after the step.
func (b *Builder) QSAAttentionParts(packed *Tensor, tokens uint64) (attention, tail, spare *Tensor) {
	values := QSAHeadWidth * QSAQueryHeads * tokens
	attention = b.FlatSlice(packed, FirstOffset, QSAHeadWidth, QSAQueryHeads, tokens)
	tail = b.FlatSlice(packed, values, QSAIndexerWidth, QSATailRows)
	spare = b.FlatSlice(packed, values+QSATailRows*QSAIndexerWidth, QSAIndexerWidth)
	return attention, tail, spare
}

func validateQSAAttention(
	query, key, value, indexerKey, indexerQuery, indexerNorm, tail, spare *Tensor, state QSAState,
	attributes QSAAttentionAttributes,
) error {
	tokens := uint64(len(attributes.Cells))
	capacity := uint64(attributes.Capacity)
	if tokens == 0 || capacity == 0 || capacity%QSABlockCells != 0 {
		return errors.New("QSA attention takes tokens and a capacity of whole pages")
	}
	for index, cell := range attributes.Cells {
		if index > 0 && cell != attributes.Cells[index-1]+1 {
			return errors.New("QSA attention takes consecutive cells")
		}
	}
	if uint64(attributes.Cells[tokens-1]) >= capacity || capacity > qsaMaxCapacity || !(attributes.Epsilon > 0) {
		return errors.New("QSA attention cells or epsilon are outside the state")
	}
	pages := capacity / QSABlockCells
	want := []struct {
		tensor *Tensor
		kind   dtype.Type
		shape  Shape
	}{
		{query, dtype.F32, MustShape(QSAHeadWidth, QSAQueryHeads, tokens)},
		{key, dtype.F32, MustShape(QSAHeadWidth, QSAKeyValueHeads, tokens)},
		{value, dtype.F32, MustShape(QSAHeadWidth, QSAKeyValueHeads, tokens)},
		{indexerKey, dtype.F32, MustShape(QSAIndexerWidth, SingletonExtent, tokens)},
		{indexerQuery, dtype.F32, MustShape(QSAIndexerWidth, QSAIndexerHeads, tokens)},
		{indexerNorm, dtype.F32, MustShape(QSAIndexerWidth)},
		{tail, dtype.F32, MustShape(QSAIndexerWidth, QSATailRows)},
		{spare, dtype.F32, MustShape(QSAIndexerWidth)},
		{state.KeyCodes, dtype.I8, MustShape(QSAHeadWidth, QSAKeyValueHeads*capacity)},
		{state.ValueCodes, dtype.I8, MustShape(QSAHeadWidth, QSAKeyValueHeads*capacity)},
		{state.KeyScales, dtype.F16, MustShape(QSAHeadWidth/QSAScaleGroup, QSAKeyValueHeads*capacity)},
		{state.ValueScales, dtype.F16, MustShape(QSAHeadWidth/QSAScaleGroup, QSAKeyValueHeads*capacity)},
		{state.PageTable, dtype.I32, MustShape(pages)},
		{state.IndexerPooled, dtype.F32, MustShape(QSAIndexerWidth, pages+QSAPooledSpare)},
		{state.BlockPosition, dtype.I32, MustShape(SingletonExtent)},
		{state.Positions, dtype.I32, MustShape(QSAPositionAxes, capacity)},
		{state.RopeCos, dtype.F32, MustShape(QSARotated/PairedExtent, capacity)},
		{state.RopeSin, dtype.F32, MustShape(QSARotated/PairedExtent, capacity)},
	}
	for _, input := range want {
		if input.tensor == nil || input.tensor.Type != input.kind || !input.tensor.Shape.Equal(input.shape) {
			return errors.New("QSA attention inputs differ from the compiled QSA geometry")
		}
	}
	return nil
}
