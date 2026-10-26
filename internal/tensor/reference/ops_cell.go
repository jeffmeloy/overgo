package reference

import (
	"errors"
	"math"

	"overgo/internal/tensor"
)

// cellAttention is the host form of the cell attention (tensor.CellAttention): each token's key and value appended as
// a cell by kv_q8's quantizer (quantizeCellGroup), then each token's query heads attending over the cells it reads,
// its causal prefix or its sliding window, the scores scaled by the attributes' scale, in float64. It writes the
// state's codes and scales in place, so a step's state is the next step's input, as on the device; a token reads the
// cells the step appended before it, its own among them.
func cellAttention(inputs []Value, attributes tensor.CellAttentionAttributes) (Value, error) {
	if len(inputs) != tensor.CellAttentionInputs {
		return Value{}, errors.New("cell attention inputs are invalid")
	}
	query, key, value := inputs[cellInputQuery], inputs[cellInputKey], inputs[cellInputValue]
	width, heads, kvHeads := int(query.Shape.Dims[0]), int(query.Shape.Dims[1]), int(key.Shape.Dims[1])
	state := cellHostState{
		width: width, kvHeads: kvHeads,
		keyCodes: inputs[cellInputKeyCodes].Data, valueCodes: inputs[cellInputValueCodes].Data,
		keyScales: inputs[cellInputKeyScales].Data, valueScales: inputs[cellInputValueScales].Data,
		table: inputs[cellInputPageTable].Data,
	}
	output := make([]float32, len(query.Data))
	for index, cell := range attributes.Cells {
		row := func(values []float32, rowHeads int) []float32 { return values[index*rowHeads*width:][:rowHeads*width] }
		if err := state.appendCell(int(cell), row(key.Data, kvHeads), row(value.Data, kvHeads)); err != nil {
			return Value{}, err
		}
		first, count := attributes.Reads(cell)
		state.attend(int(first), int(count), heads, float64(attributes.Scale), row(query.Data, heads), row(output, heads))
	}
	return Value{Shape: query.Shape, Data: output}, nil
}

// The cell attention's inputs, in tensor.CellAttention's order.
const (
	cellInputQuery = iota
	cellInputKey
	cellInputValue
	cellInputKeyCodes
	cellInputValueCodes
	cellInputKeyScales
	cellInputValueScales
	cellInputPageTable
)

// cellHostState is a layer's cells: codes `[page][kv_head][CellPageCells][width]` and scales `[page][kv_head]
// [CellPageCells][width / 64]` behind the page table.
type cellHostState struct {
	width, kvHeads                                      int
	keyCodes, valueCodes, keyScales, valueScales, table []float32
}

// row is a cell's head's row in the pools, through the page table.
func (s cellHostState) row(cell, head int) int {
	page := int(s.table[cell/tensor.CellPageCells])
	return (page*s.kvHeads+head)*tensor.CellPageCells + cell%tensor.CellPageCells
}

// appendCell is kv_append_q8's rule for one cell: each 64-value group of each head quantized against its fp16 scale.
func (s cellHostState) appendCell(cell int, keys, values []float32) error {
	if cell/tensor.CellPageCells >= len(s.table) || s.table[cell/tensor.CellPageCells] < 0 {
		return errors.New("cell is outside the state's pages")
	}
	groups := s.width / tensor.QSAScaleGroup
	for half, rows := range [][]float32{keys, values} {
		codes, scales := s.keyCodes, s.keyScales
		if half == 1 {
			codes, scales = s.valueCodes, s.valueScales
		}
		for head := range s.kvHeads {
			row := s.row(cell, head)
			for group := range groups {
				scales[row*groups+group] = quantizeCellGroup(rows[head*s.width+group*tensor.QSAScaleGroup:][:tensor.QSAScaleGroup],
					codes[row*s.width+group*tensor.QSAScaleGroup:][:tensor.QSAScaleGroup])
			}
		}
	}
	return nil
}

// attend is the attention over the cells [first, first + count): each query head's softmax of q . k * scale over its
// K/V head's dequantized keys, applied to the dequantized values, in float64.
func (s cellHostState) attend(first, count, heads int, scale float64, query, output []float32) {
	group, groups := heads/s.kvHeads, s.width/tensor.QSAScaleGroup
	weights := make([]float64, count)
	for head := range heads {
		kvHead := head / group
		maximum := math.Inf(-1)
		for index := range count {
			row := s.row(first+index, kvHead)
			var dot float64
			for d := range s.width {
				k := float64(s.keyCodes[row*s.width+d]) * float64(s.keyScales[row*groups+d/tensor.QSAScaleGroup])
				dot += float64(query[head*s.width+d]) * k
			}
			weights[index] = dot * scale
			maximum = max(maximum, weights[index])
		}
		var sum float64
		for index := range weights {
			weights[index] = math.Exp(weights[index] - maximum)
			sum += weights[index]
		}
		for d := range s.width {
			var accumulated float64
			for index := range count {
				row := s.row(first+index, kvHead)
				v := float64(s.valueCodes[row*s.width+d]) * float64(s.valueScales[row*groups+d/tensor.QSAScaleGroup])
				accumulated += weights[index] * v
			}
			output[head*s.width+d] = float32(accumulated / sum)
		}
	}
}
