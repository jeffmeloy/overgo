// Forward for the layered-attention seq2seq capability: bidirectional RoPE
// encoder, full-recompute teacher-forced decode, and the shared per-layer
// pieces the incremental session reuses. All math is per-row identical
// between the full and incremental paths, so both decode forms agree
// bit-for-bit (asserted by the parity tests).
package seq2seq

import (
	"fmt"

	"overgo/internal/hostmath"
	"overgo/internal/tensor/dtype"
)

// embedRows: dst[rows,d] = embed[token]*sqrt(d) — the reference's scaled
// word embedding.
func (m *Model) embedRows(dst []float32, tokens []int) error {
	d := m.Dims.DModel
	for i, token := range tokens {
		if token < 0 || token >= m.Dims.Vocab {
			return fmt.Errorf("seq2seq: token %d at position %d outside vocab %d", token, i, m.Dims.Vocab)
		}
		row, source := dst[i*d:(i+1)*d], m.embed[token*d:(token+1)*d]
		for j, value := range source {
			row[j] = dtype.BF16ToFloat32(value) * m.embedScale
		}
	}
	return nil
}

// projectQ: normed hidden -> per-head-normed, roped (posBase.. for self;
// rope skipped when roped=false for cross), score-scale-folded queries.
// Score scale folds into q because the hostmath cores run scale 1; with a
// power-of-two head dim the fold is bit-exact against scaling the scores.
func (m *Model) projectQ(dst, normed []float32, rows int, block *attnBlock, posBase int, roped bool) {
	m.projectQTrace(dst, nil, normed, rows, block, posBase, roped)
}

func (m *Model) projectQTrace(dst, raw, normed []float32, rows int, block *attnBlock, posBase int, roped bool) {
	heads, hd := m.Dims.Heads, m.Dims.HeadDim
	projected := dst
	if raw != nil {
		projected = raw
	}
	hostmath.LinearBF16(projected, normed, block.q, rows, m.Dims.DModel, heads*hd)
	hostmath.RMSNormInto(dst, projected, block.qNorm, rows*heads, hd, m.Dims.RMSEps)
	if roped {
		m.ropeRows(dst, rows, heads, posBase)
	}
	for i := range dst {
		dst[i] *= m.scoreScale
	}
}

// projectKV: normed source rows -> per-head-normed roped keys and raw values.
func (m *Model) projectKV(dstK, dstV, source []float32, rows int, block *attnBlock, posBase int, roped bool) {
	m.projectKVTrace(dstK, dstV, nil, source, rows, block, posBase, roped)
}

func (m *Model) projectKVTrace(dstK, dstV, rawK, source []float32, rows int, block *attnBlock, posBase int, roped bool) {
	kv, hd := m.Dims.KVHeads, m.Dims.HeadDim
	projectedK := dstK
	if rawK != nil {
		projectedK = rawK
	}
	hostmath.LinearBF16(projectedK, source, block.k, rows, m.Dims.DModel, kv*hd)
	hostmath.RMSNormInto(dstK, projectedK, block.kNorm, rows*kv, hd, m.Dims.RMSEps)
	if roped {
		m.ropeRows(dstK, rows, kv, posBase)
	}
	hostmath.LinearBF16(dstV, source, block.v, rows, m.Dims.DModel, kv*hd)
}

// ropeRows: rotate-half every head of rows whose absolute positions start
// at posBase.
func (m *Model) ropeRows(x []float32, rows, heads, posBase int) {
	hd := m.Dims.HeadDim
	for p := 0; p < rows; p++ {
		for h := 0; h < heads; h++ {
			hostmath.ApplyRotaryHalf(x[(p*heads+h)*hd:(p*heads+h+1)*hd], m.invFreq, posBase+p)
		}
	}
}

// gatedResidualOut: hidden += gate * (attn @ o_proj) per row.
func (m *Model) gatedResidualOut(hidden, attn []float32, oProj []uint16, rows int, gate float32) {
	d := m.Dims.DModel
	projected := make([]float32, rows*d)
	hostmath.LinearBF16(projected, attn, oProj, rows, m.Dims.Heads*m.Dims.HeadDim, d)
	addGatedResidual(hidden, projected, gate)
}

func addGatedResidual(hidden, projected []float32, gate float32) {
	for i, value := range projected {
		hidden[i] += gate * value
	}
}

// Encode runs the bidirectional encoder over source token ids and returns
// the final-normed memory [len(src), d].
func (m *Model) Encode(src []int) ([]float32, error) {
	if len(src) == 0 {
		return nil, fmt.Errorf("seq2seq: empty source")
	}
	dims := m.Dims
	rows, d := len(src), dims.DModel
	hidden := make([]float32, rows*d)
	if err := m.embedRows(hidden, src); err != nil {
		return nil, err
	}
	normed := make([]float32, rows*d)
	q := make([]float32, rows*dims.Heads*dims.HeadDim)
	k := make([]float32, rows*dims.KVHeads*dims.HeadDim)
	v := make([]float32, rows*dims.KVHeads*dims.HeadDim)
	attn := make([]float32, rows*dims.Heads*dims.HeadDim)
	for layer := range m.encoder {
		block := &m.encoder[layer]
		hostmath.RMSNormInto(normed, hidden, block.inNorm, rows, d, dims.RMSEps)
		m.projectQ(q, normed, rows, block, 0, true)
		m.projectKV(k, v, normed, rows, block, 0, true)
		hostmath.MaskedBidirectionalAttention(attn, q, k, v, rows, rows, dims.Heads, dims.KVHeads, dims.HeadDim, nil)
		m.gatedResidualOut(hidden, attn, block.o, rows, block.gate)
	}
	hostmath.RMSNormInto(hidden, hidden, m.encFinalNorm, rows, d, dims.RMSEps)
	return hidden, nil
}

// crossMemory: per-decoder-layer static cross-attention K/V, projected once
// from the encoder memory (keys per-head normed; no RoPE on cross rows).
type crossMemory struct {
	k, v [][]float32
	rows int
}

// projectCrossMemory computes every cross layer's static K/V from memory.
// Shared by full and incremental decode so both read identical values.
func (m *Model) projectCrossMemory(memory []float32, memRows int) (*crossMemory, error) {
	return m.projectCrossMemoryTrace(memory, memRows, nil)
}

func (m *Model) projectCrossMemoryTrace(memory []float32, memRows int, trace *decoderTrainingTrace) (*crossMemory, error) {
	if memRows <= 0 || len(memory) != memRows*m.Dims.DModel {
		return nil, fmt.Errorf("seq2seq: memory %d values for %d rows of width %d", len(memory), memRows, m.Dims.DModel)
	}
	cross := &crossMemory{rows: memRows}
	width := memRows * m.Dims.KVHeads * m.Dims.HeadDim
	for layer := range m.decoderCross {
		block := &m.decoderCross[layer]
		k, v := make([]float32, width), make([]float32, width)
		if trace != nil && layer == len(m.decoderCross)-1 {
			trace.finalCrossMemory = memory
			trace.finalCrossKRaw = make([]float32, width)
			m.projectKVTrace(k, v, trace.finalCrossKRaw, memory, memRows, block, 0, false)
		} else {
			m.projectKV(k, v, memory, memRows, block, 0, false)
		}
		cross.k = append(cross.k, k)
		cross.v = append(cross.v, v)
	}
	return cross, nil
}

// projectLogits: logits[vocab] = rmsnorm(hiddenRow) @ embed^T (tied head).
func (m *Model) projectLogits(logits, hiddenRow, normedScratch []float32) {
	d := m.Dims.DModel
	hostmath.RMSNormInto(normedScratch[:d], hiddenRow, m.decFinalNorm, 1, d, m.Dims.RMSEps)
	hostmath.LinearBF16(logits[:m.Dims.Vocab], normedScratch[:d], m.embed, 1, d, m.Dims.Vocab)
}

// DecodeFull teacher-forces tgt through the decoder in one full-sequence
// pass and returns logits [len(tgt), vocab] — the recompute reference the
// incremental session is checked against.
func (m *Model) DecodeFull(memory []float32, memRows int, tgt []int) ([]float32, error) {
	hidden, err := m.decodeHiddenFull(memory, memRows, tgt)
	if err != nil {
		return nil, err
	}
	rows, d := len(tgt), m.Dims.DModel
	logits := make([]float32, rows*m.Dims.Vocab)
	scratch := make([]float32, d)
	for row := 0; row < rows; row++ {
		m.projectLogits(logits[row*m.Dims.Vocab:(row+1)*m.Dims.Vocab], hidden[row*d:(row+1)*d], scratch)
	}
	return logits, nil
}

func (m *Model) decodeHiddenFull(memory []float32, memRows int, tgt []int) ([]float32, error) {
	return m.decodeHiddenFullTrace(memory, memRows, tgt, nil)
}

type decoderTrainingTrace struct {
	finalCrossProjected []float32
	finalCrossAttention []float32
	finalCrossInput     []float32
	finalCrossNormed    []float32
	finalCrossMemory    []float32
	finalCrossQRaw      []float32
	finalCrossKRaw      []float32
	finalCrossQ         []float32
	finalCrossK         []float32
	finalCrossV         []float32
}

func (m *Model) decodeHiddenFullTrace(memory []float32, memRows int, tgt []int, trace *decoderTrainingTrace) ([]float32, error) {
	if len(tgt) == 0 {
		return nil, fmt.Errorf("seq2seq: empty target")
	}
	cross, err := m.projectCrossMemoryTrace(memory, memRows, trace)
	if err != nil {
		return nil, err
	}
	dims := m.Dims
	rows, d := len(tgt), dims.DModel
	hidden := make([]float32, rows*d)
	if err := m.embedRows(hidden, tgt); err != nil {
		return nil, err
	}
	normed := make([]float32, rows*d)
	q := make([]float32, rows*dims.Heads*dims.HeadDim)
	k := make([]float32, rows*dims.KVHeads*dims.HeadDim)
	v := make([]float32, rows*dims.KVHeads*dims.HeadDim)
	attn := make([]float32, rows*dims.Heads*dims.HeadDim)
	for layer := range m.decoderSelf {
		self := &m.decoderSelf[layer]
		hostmath.RMSNormInto(normed, hidden, self.inNorm, rows, d, dims.RMSEps)
		m.projectQ(q, normed, rows, self, 0, true)
		m.projectKV(k, v, normed, rows, self, 0, true)
		hostmath.CausalAttention(attn, q, k, v, rows, dims.Heads, dims.KVHeads, dims.HeadDim)
		m.gatedResidualOut(hidden, attn, self.o, rows, self.gate)

		crossBlock := &m.decoderCross[layer]
		if trace != nil && layer == len(m.decoderCross)-1 {
			trace.finalCrossInput = append(trace.finalCrossInput[:0], hidden...)
		}
		hostmath.RMSNormInto(normed, hidden, crossBlock.inNorm, rows, d, dims.RMSEps)
		if trace != nil && layer == len(m.decoderCross)-1 {
			trace.finalCrossNormed = append(trace.finalCrossNormed[:0], normed...)
			trace.finalCrossQRaw = make([]float32, len(q))
			m.projectQTrace(q, trace.finalCrossQRaw, normed, rows, crossBlock, 0, false)
		} else {
			m.projectQ(q, normed, rows, crossBlock, 0, false)
		}
		hostmath.MaskedBidirectionalAttention(attn, q, cross.k[layer], cross.v[layer], rows, cross.rows, dims.Heads, dims.KVHeads, dims.HeadDim, nil)
		if trace != nil && layer == len(m.decoderCross)-1 {
			trace.finalCrossAttention = append(trace.finalCrossAttention[:0], attn...)
			trace.finalCrossQ = append(trace.finalCrossQ[:0], q...)
			trace.finalCrossK = append(trace.finalCrossK[:0], cross.k[layer]...)
			trace.finalCrossV = append(trace.finalCrossV[:0], cross.v[layer]...)
			trace.finalCrossProjected = make([]float32, rows*d)
			hostmath.LinearBF16(trace.finalCrossProjected, attn, crossBlock.o, rows, dims.Heads*dims.HeadDim, d)
			addGatedResidual(hidden, trace.finalCrossProjected, crossBlock.gate)
		} else {
			m.gatedResidualOut(hidden, attn, crossBlock.o, rows, crossBlock.gate)
		}
	}
	return hidden, nil
}
