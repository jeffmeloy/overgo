// Backbone forward: incremental causal decode over caller-owned KV state.
// The full-sequence pass IS the incremental pass over a fresh state
// (hostmath.CausalAttentionStep is bit-identical to the full-sequence core
// row), so streamed and batched computations cannot drift apart.
package speechsynth

import (
	"fmt"

	"overgo/internal/hostmath"
)

// DecodeState: per-layer KV caches; rows counts positions appended so far.
type DecodeState struct {
	k, v [][]float32 // per layer, [rows*d] flat, [row][heads][headDim]
	rows int
}

// NewDecodeState pre-sizes the caches for capacity positions.
func (m *Model) NewDecodeState(capacity int) *DecodeState {
	st := &DecodeState{k: make([][]float32, m.Dims.Layers), v: make([][]float32, m.Dims.Layers)}
	for i := range st.k {
		st.k[i] = make([]float32, 0, capacity*m.Dims.DModel)
		st.v[i] = make([]float32, 0, capacity*m.Dims.DModel)
	}
	return st
}

// AppendForward advances the decode by T rows: x is [T][d] time-major and is
// overwritten with the transformer outputs (pre-out_norm residual stream).
// T > 1 runs layer-major with rows=T kernels (prefill utilization); each
// per-row op is the same accumulation as the T=1 path, and
// CausalAttentionSteps is bit-identical per row to CausalAttentionStep, so
// the batched pass cannot drift from the incremental one.
func (m *Model) AppendForward(st *DecodeState, x []float32, T int) {
	d, h, hd, ff := m.Dims.DModel, m.Dims.Heads, m.Dims.HeadDim, m.Dims.FF
	rows0 := st.rows
	qkv := make([]float32, T*3*d)
	q := make([]float32, T*d)
	xn := make([]float32, T*d)
	attn := make([]float32, T*d)
	proj := make([]float32, T*d)
	h1 := make([]float32, T*ff)
	for li := range m.layers {
		l := &m.layers[li]
		hostmath.LayerNormInto(xn, x, l.norm1W, l.norm1B, T, d, transformerLayerNormEps)
		hostmath.Linear(qkv, xn, l.inProj, T, d, 3*d)
		for t := 0; t < T; t++ {
			qr, kr, vr := qkv[t*3*d:t*3*d+d], qkv[t*3*d+d:t*3*d+2*d], qkv[t*3*d+2*d:(t+1)*3*d]
			for head := 0; head < h; head++ {
				hostmath.ApplyRotaryInterleaved(qr[head*hd:(head+1)*hd], m.invFreq, rows0+t)
				hostmath.ApplyRotaryInterleaved(kr[head*hd:(head+1)*hd], m.invFreq, rows0+t)
			}
			st.k[li] = append(st.k[li], kr...)
			st.v[li] = append(st.v[li], vr...)
			qt := q[t*d : (t+1)*d]
			for i, v := range qr {
				qt[i] = v * m.scoreScale
			}
		}
		hostmath.CausalAttentionSteps(attn, q, st.k[li], st.v[li], rows0, T, h, h, hd)
		hostmath.Linear(proj, attn, l.outProj, T, d, d)
		for i := range x {
			x[i] += proj[i]
		}
		hostmath.LayerNormInto(xn, x, l.norm2W, l.norm2B, T, d, transformerLayerNormEps)
		hostmath.Linear(h1, xn, l.lin1, T, d, ff)
		hostmath.GELUErfInPlace(h1)
		hostmath.Linear(proj, h1, l.lin2, T, ff, d)
		for i := range x {
			x[i] += proj[i]
		}
	}
	st.rows = rows0 + T
}

// TextEmbedInto copies conditioner lookup rows for ids into out [len(ids)][d].
func (m *Model) TextEmbedInto(out []float32, ids []int) error {
	d := m.Dims.DModel
	for i, id := range ids {
		if id < 0 || id >= m.Dims.TextVocab {
			return fmt.Errorf("speechsynth: text id %d outside conditioner table %d", id, m.Dims.TextVocab)
		}
		copy(out[i*d:(i+1)*d], m.condEmbed[id*d:(id+1)*d])
	}
	return nil
}

// LatentInputInto projects one frame latent into the transformer width.
func (m *Model) LatentInputInto(dst, latent []float32) {
	hostmath.Linear(dst, latent, m.inputLinear, 1, m.Dims.LatentDim, m.Dims.DModel)
}

// OutNormInto applies the flow-LM output LayerNorm over rows of width d.
func (m *Model) OutNormInto(dst, x []float32, rows int) {
	hostmath.LayerNormInto(dst, x, m.outNormW, m.outNormB, rows, m.Dims.DModel, transformerLayerNormEps)
}

// EOSLogit: the scalar end-of-speech head over one out_norm row.
func (m *Model) EOSLogit(cond []float32) float64 {
	logit := float64(m.outEosB)
	for i, w := range m.outEosW {
		logit += float64(w) * float64(cond[i])
	}
	return logit
}
