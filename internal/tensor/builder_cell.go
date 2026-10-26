package tensor

import (
	"errors"
	"slices"

	"overgo/internal/tensor/dtype"
)

// Cell attention: an attention layer over Strata's int8 K/V cells (kernels/cuda/strata_qsa.cu, ported from Strata at
// 296b59d2, MIT: kv_q8's append and qsa_decode_attn's and qsa_prompt_attn's attention, instantiated for the layer's
// geometry), as one operation over a per-sequence state it appends to in place. Its inputs are the layer's normed and
// rotated query and key and its value for the step's tokens, and the layer's cells. A step appends its tokens' keys and
// values as cells, each 64-value group of a head quantized against its fp16 scale, then each token attends over the
// cells it reads: its causal prefix, or its sliding window's last cells; a token as Strata's decode layer attends, a
// chunk as its prompt read does.

// CellPageCells is a cell page's cells: the page table maps a sequence's pages to the pools' pages, and a sliding layer
// takes back the pages its window has passed. Provisional: the load-costs step prices a page against the table it
// sizes and the cells a sliding layer holds past its window.
const CellPageCells = 64

// The geometries the cell attention's kernels are instantiated for (kernels/cuda/strata_qsa.cu STRATA_CELL_ENTRIES), a
// head width and the query heads a K/V head each: a llama layer of 2 K/V heads of 128 under 16 query heads, and
// gemma-4-12B's sliding layers (8 K/V heads of 256 under 16) and global layers (one K/V head of 512 under 16).
const (
	cellLlamaWidth, cellLlamaGroup     = 128, 8
	cellSlidingWidth, cellSlidingGroup = 256, 2
	cellGlobalWidth, cellGlobalGroup   = 512, 16
)

// CellAttentionGeometries are the geometries the cell attention's kernels are instantiated for, in the order of their
// entries.
var CellAttentionGeometries = [][PairedExtent]uint64{
	{cellLlamaWidth, cellLlamaGroup}, {cellSlidingWidth, cellSlidingGroup}, {cellGlobalWidth, cellGlobalGroup},
}

// CellTokenAxis is the token axis of a cell attention step's rows, (width, heads, tokens), of cellRowRank axes.
const (
	cellRowRank   = 3
	CellTokenAxis = cellRowRank - 1
)

// CellAttentionHolds reports whether the cell attention's kernels serve a layer of query heads over kvHeads K/V heads of
// width.
func CellAttentionHolds(width, heads, kvHeads uint64) bool {
	return kvHeads > 0 && heads%kvHeads == 0 && slices.Contains(CellAttentionGeometries, [PairedExtent]uint64{width, heads / kvHeads})
}

// CellState is one layer's cells for one sequence at a capacity of cells, which a step appends to in place: the int8
// K/V codes and their fp16 scales in pages of CellPageCells cells `[page][kv_head][CellPageCells][width]` behind the
// page table.
type CellState struct {
	KeyCodes, ValueCodes   *Tensor // I8 (width, kvHeads x capacity)
	KeyScales, ValueScales *Tensor // F16 (width / 64, kvHeads x capacity)
	PageTable              *Tensor // I32 (capacity / CellPageCells)
}

// CellAttentionAttributes are a cell attention step's tokens: the cells they occupy (consecutive), the state's capacity,
// the sliding window a token reads the last cells of (zero for every cell before it) and the scores' scale. Each
// token's step record - its cell, the cells then held, the first cell it reads and the cells it reads - is staged to
// the device as the node's runtime words.
type CellAttentionAttributes struct {
	Cells    []uint32
	Capacity uint32
	Window   uint32
	Scale    float32
}

func (CellAttentionAttributes) validFor(op Op) bool { return sameOp(op, OpCellAttention) }

// steps are the attributes' step records, QSAStepWords each.
func (a CellAttentionAttributes) steps() []uint32 {
	words := make([]uint32, 0, QSAStepWords*len(a.Cells))
	for _, cell := range a.Cells {
		held, first := cell+1, uint32(0)
		if a.Window > 0 && held > a.Window {
			first = held - a.Window
		}
		words = append(words, cell, held, first, held-first)
	}
	return words
}

// Reads is the cells a token at cell reads: its sliding window's last cells, or every cell up to its own.
func (a CellAttentionAttributes) Reads(cell uint32) (first, count uint32) {
	held := cell + 1
	if a.Window > 0 && held > a.Window {
		first = held - a.Window
	}
	return first, held - first
}

// CellAttentionInputs is the operation's inputs: the query, the key and the value, then the state's five tensors.
const CellAttentionInputs = 8

// CellAttention builds one cell attention step over the state. query is (width, heads, tokens), key and value
// (width, kvHeads, tokens); the output is the attention (width, heads, tokens).
func (b *Builder) CellAttention(query, key, value *Tensor, state CellState, attributes CellAttentionAttributes) *Tensor {
	if b == nil || b.err != nil {
		return nil
	}
	if err := validateCellAttention(query, key, value, state, attributes); err != nil {
		b.setError(err)
		return nil
	}
	attributes.Cells = slices.Clone(attributes.Cells)
	inputs := []*Tensor{query, key, value, state.KeyCodes, state.ValueCodes, state.KeyScales, state.ValueScales, state.PageTable}
	return b.add("", dtype.F32, query.Shape, OpCellAttention, inputs, attributes)
}

func validateCellAttention(query, key, value *Tensor, state CellState, attributes CellAttentionAttributes) error {
	tokens := uint64(len(attributes.Cells))
	if query == nil || key == nil || value == nil || tokens == 0 {
		return errors.New("cell attention inputs are missing")
	}
	for _, input := range []*Tensor{query, key, value} {
		if input.Type != dtype.F32 || input.Shape.Rank != cellRowRank || input.Shape.Dims[CellTokenAxis] != tokens {
			return errors.New("cell attention takes F32 (width, heads, tokens) rows, a token a cell")
		}
	}
	width, heads, kvHeads := query.Shape.Dims[0], query.Shape.Dims[1], key.Shape.Dims[1]
	if key.Shape != value.Shape || key.Shape.Dims[0] != width || !CellAttentionHolds(width, heads, kvHeads) {
		return errors.New("cell attention geometry has no kernels")
	}
	// A window wider than the state reads every cell, as a short sequence's sliding layer does.
	capacity := uint64(attributes.Capacity)
	if capacity == 0 || capacity%CellPageCells != 0 {
		return errors.New("cell attention capacity is not whole pages")
	}
	for index, cell := range attributes.Cells {
		if cell >= attributes.Capacity || index > 0 && cell != attributes.Cells[index-1]+1 {
			return errors.New("cell attention cells are not consecutive inside the state")
		}
	}
	codes := MustShape(width, kvHeads*capacity)
	scales := MustShape(width/QSAScaleGroup, kvHeads*capacity)
	for _, part := range []struct {
		tensor *Tensor
		kind   dtype.Type
		shape  Shape
	}{
		{state.KeyCodes, dtype.I8, codes}, {state.ValueCodes, dtype.I8, codes},
		{state.KeyScales, dtype.F16, scales}, {state.ValueScales, dtype.F16, scales},
		{state.PageTable, dtype.I32, MustShape(capacity / CellPageCells)},
	} {
		if part.tensor == nil || part.tensor.Type != part.kind || part.tensor.Shape != part.shape {
			return errors.New("cell attention state does not hold the layer's cells")
		}
	}
	return nil
}
