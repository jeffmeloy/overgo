package reference

import (
	"errors"
	"math"
	"slices"

	"overgo/internal/quant"
	"overgo/internal/tensor"
)

// qsaAttention is the host form of Strata's QSA layer between its projections (tensor.QSAAttention, ported from
// Strata at 296b59d2, MIT): the step's cells appended to the state as int8 codes behind fp16 scales, the indexer's
// keys appended and a completed block pooled (Strata's table kernel for one token, its native batch for a chunk),
// every block scored, the cells selected by Strata's rule and attended over. It writes the state's cells in place -
// codes and scales as the values they hold, the page table and block position as integers - so a step's state is the
// next step's input, as on the device, and packs the attention with the tail and spare after the step, leaving the
// fixed state it read as it was. Its arithmetic is the reference's: the scores and the attention in float64, so a
// selection on a near tie may differ from the device's.
func qsaAttention(shape tensor.Shape, inputs []Value, attributes tensor.QSAAttentionAttributes) (Value, error) {
	// An in-place step's snapshot follows every other input.
	var snapshot Value
	if attributes.InPlace && len(inputs) > 0 {
		snapshot, inputs = inputs[len(inputs)-1], inputs[:len(inputs)-1]
	}
	streamed := len(inputs) == tensor.QSAAttentionInputs+tensor.QSAStreamInputs
	if len(inputs) != tensor.QSAAttentionInputs && !streamed || len(attributes.Cells) == 0 {
		return Value{}, errors.New("QSA attention inputs are invalid")
	}
	query, key, value := inputs[qsaQuery], inputs[qsaKey], inputs[qsaValue]
	indexerKey, indexerQuery, norm := inputs[qsaIndexerKey], inputs[qsaIndexerQuery], inputs[qsaIndexerNorm]
	tokens := len(attributes.Cells)
	queryWidth := tensor.QSAQueryHeads * tensor.QSAHeadWidth
	output := Value{Shape: shape, Data: make([]float32, tensor.QSAAttentionOutputValues(uint64(tokens)))}
	fixed := output.Data[queryWidth*tokens:]
	copy(fixed, inputs[qsaTail].Data)
	copy(fixed[tensor.QSATailRows*tensor.QSAIndexerWidth:], inputs[qsaSpare].Data)
	state := qsaHostState{
		tail: fixed[:tensor.QSATailRows*tensor.QSAIndexerWidth], spare: fixed[tensor.QSATailRows*tensor.QSAIndexerWidth:],
		keyCodes: inputs[qsaKeyCodes].Data, valueCodes: inputs[qsaValueCodes].Data,
		keyScales: inputs[qsaKeyScales].Data, valueScales: inputs[qsaValueScales].Data,
		table: inputs[qsaPageTable].Data, pooled: inputs[qsaPooled].Data, blockPosition: inputs[qsaBlockPosition].Data,
		positions: inputs[qsaPositions].Data, cos: inputs[qsaRopeCos].Data, sin: inputs[qsaRopeSin].Data,
	}
	if attributes.InPlace {
		// The step appends to the fixed state where it sits, its tail copied to the snapshot first; the packed output's
		// tail and spare stay unwritten.
		copy(snapshot.Data, inputs[qsaTail].Data)
		clear(fixed)
		state.tail, state.spare = inputs[qsaTail].Data, inputs[qsaSpare].Data
	}
	if streamed {
		// A streamed state's cells are its host copy's, at the identity layout; its slots and map are the device's cache
		// of them, which the reference does not hold.
		state.keyCodes, state.valueCodes = inputs[qsaHostKeyCodes].Data, inputs[qsaHostValueCodes].Data
		state.keyScales, state.valueScales = inputs[qsaHostKeyScales].Data, inputs[qsaHostValueScales].Data
		state.table = make([]float32, len(state.table))
		for page := range state.table {
			state.table[page] = float32(page)
		}
	}
	kvWidth := tensor.QSAKeyValueHeads * tensor.QSAHeadWidth
	for token, cell := range attributes.Cells {
		if err := state.appendCell(int(cell), key.Data[token*kvWidth:][:kvWidth], value.Data[token*kvWidth:][:kvWidth]); err != nil {
			return Value{}, err
		}
	}
	if tokens == 1 {
		state.appendIndexerTable(int(attributes.Cells[0]), indexerKey.Data, norm.Data, attributes.Epsilon)
	} else {
		state.appendIndexerBatch(int(attributes.Cells[0]), tokens, indexerKey.Data, norm.Data, attributes.Epsilon)
	}
	indexerWidth := tensor.QSAIndexerHeads * tensor.QSAIndexerWidth
	for token, cell := range attributes.Cells {
		selection := state.selectCells(int(cell)+1, indexerQuery.Data[token*indexerWidth:][:indexerWidth])
		state.attend(selection, query.Data[token*queryWidth:][:queryWidth], output.Data[token*queryWidth:][:queryWidth])
	}
	return output, nil
}

// qsaHostState is a state's values: codes `[page][kv_head][4][256]` and scales `[page][kv_head][4][4]` behind the
// page table, the indexer's tail (3, 128), spare (128), pooled rows (128 each), the image position table (3 a cell)
// and the rotation table (32 a position).
type qsaHostState struct {
	keyCodes, valueCodes, keyScales, valueScales []float32
	table, tail, spare, pooled, blockPosition    []float32
	positions, cos, sin                          []float32
}

// rotationPosition is mrope.hpp's mrope_pos: the position a pair of the block starting at cell rotates by, the cell's
// axis pair % 3 in the image position table.
func (s qsaHostState) rotationPosition(cell, pair int) int {
	return int(s.positions[cell*tensor.QSAPositionAxes+pair%tensor.QSAPositionAxes])
}

// row is a cell's head's row in the pools, through the page table.
func (s qsaHostState) row(cell, head int) int {
	page := int(s.table[cell/tensor.QSABlockCells])
	return (page*tensor.QSAKeyValueHeads+head)*tensor.QSABlockCells + cell%tensor.QSABlockCells
}

// appendCell is kv_append_q8_step: each 64-value group of each head quantized against its fp16-stored scale.
func (s qsaHostState) appendCell(cell int, keys, values []float32) error {
	if cell/tensor.QSABlockCells >= len(s.table) || s.table[cell/tensor.QSABlockCells] < 0 {
		return errors.New("QSA cell is outside the resident pages")
	}
	groups := tensor.QSAHeadWidth / tensor.QSAScaleGroup
	for half, rows := range [][]float32{keys, values} {
		codes, scales := s.keyCodes, s.keyScales
		if half == 1 {
			codes, scales = s.valueCodes, s.valueScales
		}
		for head := range tensor.QSAKeyValueHeads {
			row := s.row(cell, head)
			for group := range groups {
				x := rows[head*tensor.QSAHeadWidth+group*tensor.QSAScaleGroup:][:tensor.QSAScaleGroup]
				var maximum float32
				for _, value := range x {
					maximum = max(maximum, float32(math.Abs(float64(value))))
				}
				stored := quant.Float16ToFloat32(quant.Float32ToFloat16(maximum / qsaCodeRange))
				scales[row*groups+group] = stored
				for index, value := range x {
					code := float32(0)
					if stored > 0 {
						code = float32(min(max(math.RoundToEven(float64(value/stored)), -qsaCodeRange), qsaCodeRange))
					}
					codes[row*tensor.QSAHeadWidth+group*tensor.QSAScaleGroup+index] = code
				}
			}
		}
	}
	return nil
}

// qsaCodeRange is an int8 code's magnitude, the scale's divisor.
const qsaCodeRange = 127

// appendIndexerTable is qsa.cu's indexer_key_append for one cell: the raw key into the tail; on cell 0 the spare,
// rms_norm(raw) * w at angle 0; on a block's last cell the block's mean of its four raw keys normed, scaled and
// rotated at the block's first cell, in double.
func (s qsaHostState) appendIndexerTable(cell int, raw, weight []float32, epsilon float32) {
	width := tensor.QSAIndexerWidth
	slot := cell % tensor.QSABlockCells
	if slot < tensor.QSATailRows {
		copy(s.tail[slot*width:][:width], raw)
	}
	normed := func(x []float64) []float64 {
		var sum float64
		for _, value := range x {
			sum = float64(sum + float64(value*value))
		}
		inverse := 1 / math.Sqrt(sum/float64(width)+float64(epsilon))
		out := make([]float64, width)
		for d := range width {
			out[d] = float64(x[d]*inverse) * float64(weight[d])
		}
		return out
	}
	if cell == 0 {
		x := make([]float64, width)
		for d := range width {
			x[d] = float64(raw[d])
		}
		y := normed(x)
		for d := range width {
			value := float32(y[d])
			if d < tensor.QSARotated {
				value *= s.cos[d%(tensor.QSARotated/tensor.PairedExtent)]
			}
			s.spare[d], s.pooled[d] = value, value
		}
	}
	if slot != tensor.QSABlockCells-1 {
		return
	}
	mean := make([]float64, width)
	for d := range width {
		var sum float64
		for j := range tensor.QSATailRows {
			sum += float64(s.tail[j*width+d])
		}
		mean[d] = (sum + float64(raw[d])) / tensor.QSABlockCells
	}
	y := normed(mean)
	block := cell / tensor.QSABlockCells
	row := s.pooled[block*width:][:width]
	for d := range width {
		row[d] = float32(y[d])
	}
	copy(s.pooled[(block+1)*width:][:width], s.spare)
	s.blockPosition[0] = float32(block * tensor.QSABlockCells)
	s.rotate(row, block*tensor.QSABlockCells)
}

// rotate is the pooled row's in-place NEOX rotation of the block starting at cell, each pair from the table row of its
// position in the image position table.
func (s qsaHostState) rotate(row []float32, cell int) {
	half := tensor.QSARotated / tensor.PairedExtent
	for pair := range half {
		position := s.rotationPosition(cell, pair)
		c, sn := s.cos[position*half+pair], s.sin[position*half+pair]
		a, b := row[pair], row[half+pair]
		row[pair], row[half+pair] = a*c-b*sn, a*sn+b*c
	}
}

// appendIndexerBatch is native_qsa_indexer_append_batch for count consecutive cells from first: keys rounded
// through fp16, the spare on cell 0, every block the batch completes pooled, normed, scaled and rotated, then the
// tail. Its angles are cos and sin of position * base^(-2 pair / 64) computed in float32, as the native kernel's.
func (s qsaHostState) appendIndexerBatch(first, count int, raw, weight []float32, epsilon float32) {
	width := tensor.QSAIndexerWidth
	rounded := func(value float32) float32 { return quant.Float16ToFloat32(quant.Float32ToFloat16(value)) }
	thetaScale := float32(math.Pow(tensor.QSARopeBase, -tensor.PairedExtent/float64(tensor.QSARotated)))
	key := func(cell, d int) float32 {
		if cell >= first {
			return rounded(raw[(cell-first)*width+d])
		}
		return s.tail[(cell%tensor.QSABlockCells)*width+d]
	}
	pooledRow := func(cells [tensor.QSABlockCells]int, zero bool) []float32 {
		mean := make([]float32, width)
		square := float32(0)
		for d := range width {
			sum := key(cells[0], d)
			for j := 1; j < tensor.QSABlockCells; j++ {
				sum += key(cells[j], d)
			}
			mean[d] = sum / tensor.QSABlockCells
			square += mean[d] * mean[d]
		}
		scale := float32(1 / math.Sqrt(float64(square/float32(width)+epsilon)))
		values := make([]float32, width)
		for d := range width {
			values[d] = scale * mean[d] * weight[d]
		}
		out := slices.Clone(values)
		half := tensor.QSARotated / tensor.PairedExtent
		for pair := range half {
			angle := float32(0)
			if !zero {
				angle = float32(s.rotationPosition(cells[0], pair)) * float32(math.Pow(float64(thetaScale), float64(pair)))
			}
			c, sn := float32(math.Cos(float64(angle))), float32(math.Sin(float64(angle)))
			a, z := values[pair], values[pair+half]
			out[pair], out[pair+half] = a*c-z*sn, a*sn+z*c
		}
		return out
	}
	if first == 0 {
		spare := pooledRow([tensor.QSABlockCells]int{0, 0, 0, 0}, true)
		copy(s.pooled[:width], spare)
		copy(s.spare, spare)
	}
	last := first + count - 1
	for block := first / tensor.QSABlockCells; block*tensor.QSABlockCells+tensor.QSABlockCells-1 <= last; block++ {
		end := block*tensor.QSABlockCells + tensor.QSABlockCells - 1
		if end < first {
			continue
		}
		cells := [tensor.QSABlockCells]int{}
		for j := range cells {
			cells[j] = block*tensor.QSABlockCells + j
		}
		position := block * tensor.QSABlockCells
		copy(s.pooled[block*width:][:width], pooledRow(cells, false))
		if (block+1)*tensor.QSABlockCells+tensor.QSABlockCells-1 > last {
			copy(s.pooled[(block+1)*width:][:width], s.spare)
			s.blockPosition[0] = float32(position)
		}
	}
	for slot := range tensor.QSATailRows {
		cell := last - ((last%tensor.QSABlockCells)-slot+tensor.QSABlockCells)%tensor.QSABlockCells
		if cell >= first {
			for d := range width {
				s.tail[slot*width+d] = rounded(raw[(cell-first)*width+d])
			}
		}
	}
}

// qsaTailScore lifts the incomplete block's score above every complete block's, so a selection always holds it
// (qsa_select.cu's tail bonus).
const qsaTailScore = 1e9

// The QSA operation's inputs, in tensor.QSAAttention's order.
const (
	qsaQuery = iota
	qsaKey
	qsaValue
	qsaIndexerKey
	qsaIndexerQuery
	qsaIndexerNorm
	qsaTail
	qsaSpare
	qsaKeyCodes
	qsaValueCodes
	qsaKeyScales
	qsaValueScales
	qsaPageTable
	qsaPooled
	qsaBlockPosition
	qsaPositions
	qsaRopeCos
	qsaRopeSin
	qsaHostKeyCodes
	qsaHostValueCodes
	qsaHostKeyScales
	qsaHostValueScales
)

// selectCells is qsa_block_scores and qsa_block_topk for a query that sees held cells: each block's score the sum
// over the indexer heads of relu(q . k), the tail block scoring the spare key with +1e9 when it holds cells; then the
// selection width's cells by score, ties to the lowest index, ascending - the identity while every cell fits.
func (s qsaHostState) selectCells(held int, query []float32) []int {
	width := int(tensor.QSASelectionWidth(uint32(held)))
	if held <= width {
		cells := make([]int, held)
		for index := range cells {
			cells[index] = index
		}
		return cells
	}
	blocks := held / tensor.QSABlockCells
	type scored struct {
		block int
		score float64
	}
	scores := make([]scored, blocks+1)
	for block := range blocks + 1 {
		keys := s.pooled[block*tensor.QSAIndexerWidth:][:tensor.QSAIndexerWidth]
		if block == blocks {
			keys = s.spare
		}
		var score float64
		for head := range tensor.QSAIndexerHeads {
			var dot float64
			for d, k := range keys {
				dot += float64(k) * float64(query[head*tensor.QSAIndexerWidth+d])
			}
			score += max(dot, 0)
		}
		if block == blocks && held%tensor.QSABlockCells != 0 {
			score += qsaTailScore
		}
		scores[block] = scored{block, float64(float32(score))}
	}
	slices.SortStableFunc(scores, func(a, b scored) int {
		switch {
		case a.score > b.score:
			return -1
		case a.score < b.score:
			return 1
		}
		return a.block - b.block
	})
	cells := make([]int, 0, width)
	for _, entry := range scores {
		count := tensor.QSABlockCells
		if entry.block == blocks {
			count = held - blocks*tensor.QSABlockCells
		}
		for cell := range count {
			if len(cells) == width {
				break
			}
			cells = append(cells, entry.block*tensor.QSABlockCells+cell)
		}
	}
	slices.Sort(cells)
	return cells
}

// attend is the attention over the selected cells: each query head's softmax of q . k / 16 over its K/V head's
// dequantized keys, applied to the dequantized values, in float64.
func (s qsaHostState) attend(cells []int, query, output []float32) {
	group := tensor.QSAQueryHeads / tensor.QSAKeyValueHeads
	groups := tensor.QSAHeadWidth / tensor.QSAScaleGroup
	scale := 1 / math.Sqrt(tensor.QSAHeadWidth)
	weights := make([]float64, len(cells))
	for head := range tensor.QSAQueryHeads {
		kvHead := head / group
		maximum := math.Inf(-1)
		for index, cell := range cells {
			row := s.row(cell, kvHead)
			var dot float64
			for d := range tensor.QSAHeadWidth {
				k := float64(s.keyCodes[row*tensor.QSAHeadWidth+d]) * float64(s.keyScales[row*groups+d/tensor.QSAScaleGroup])
				dot += float64(query[head*tensor.QSAHeadWidth+d]) * k
			}
			weights[index] = dot * scale
			maximum = max(maximum, weights[index])
		}
		var sum float64
		for index := range weights {
			weights[index] = math.Exp(weights[index] - maximum)
			sum += weights[index]
		}
		for d := range tensor.QSAHeadWidth {
			var accumulated float64
			for index, cell := range cells {
				row := s.row(cell, kvHead)
				v := float64(s.valueCodes[row*tensor.QSAHeadWidth+d]) * float64(s.valueScales[row*groups+d/tensor.QSAScaleGroup])
				accumulated += weights[index] * v
			}
			output[head*tensor.QSAHeadWidth+d] = float32(accumulated / sum)
		}
	}
}

// qsaCommit is capture_commit's QSA part: the tail restored from the snapshot, then each kept row's raw key appended
// again at its cell by qsa.cu's indexer_key_append, in place on the state.
func qsaCommit(node *tensor.Tensor, inputs []Value) (Value, error) {
	attributes, ok := node.Attrs.(tensor.QSACommitAttributes)
	if !ok || len(inputs) < tensor.QSACommitOperands {
		return Value{}, errors.New("QSA commit inputs are invalid")
	}
	state := qsaHostState{
		tail: inputs[tensor.QSACommitTail].Data, spare: inputs[tensor.QSACommitSpare].Data,
		pooled: inputs[tensor.QSACommitPooled].Data, blockPosition: inputs[tensor.QSACommitBlockPosition].Data,
		positions: inputs[tensor.QSACommitPositions].Data, cos: inputs[tensor.QSACommitRopeCos].Data,
		sin: inputs[tensor.QSACommitRopeSin].Data,
	}
	copy(state.tail, inputs[tensor.QSACommitSnapshot].Data)
	raw, cells, norm := inputs[tensor.QSACommitRaw].Data, inputs[tensor.QSACommitCells].Data, inputs[tensor.QSACommitNorm].Data
	width := tensor.QSAIndexerWidth
	for row, cell := range cells {
		if cell >= 0 {
			state.appendIndexerTable(int(cell), raw[row*width:(row+1)*width], norm, attributes.Epsilon)
		}
	}
	return Value{Shape: node.Shape, Data: []float32{math.Float32frombits(uint32(int32(cells[0])))}}, nil
}
