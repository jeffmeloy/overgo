// Decoder wiring for the patched time-series capability: post-norm
// full-attention layers over hostmath primitives. Ordering ported from the
// reference (post-norm family): RoPE is applied BEFORE the per-head q/k RMS
// norms; the query is then scaled per-dimension by log2(e)/sqrt(headDim) *
// softplus(per_dim_scale) (softmax score scale is 1 — the compiled query
// scale carries the whole normalization); attention is causal over the
// patch sequence; both branches are post-normed before their plain residual
// adds; the feed-forward is dense sequential (ff1·SiLU(ff0·x)). Every
// dimension derives from tensor shapes; no constant here asserts a model
// fact.
package seriesforecast

import (
	"fmt"
	"math"

	"overgo/internal/hostmath"
)

// layer wraps one stacked_xf layer's tensors under their canonical names.
type layer struct {
	preAttnLN, postAttnLN, preFFLN, postFFLN []float32 // [hidden]
	q, k, v, o                               []float32 // [hidden*hidden] each
	queryLN, keyLN, perDimScale              []float32 // [headDim]
	ff0, ff1                                 []float32 // [hidden*hidden]
}

func (m *Model) layerWeights(index int) (layer, error) {
	prefix := fmt.Sprintf("stacked_xf.%d", index)
	attn := prefix + ".attn"
	hidden := m.Dims.Hidden
	matrix := hidden * hidden
	qkv := m.Weights[attn+".qkv_proj.weight"]
	if len(qkv) != 3*matrix {
		return layer{}, fmt.Errorf("seriesforecast: layer %d qkv=%d, want %d", index, len(qkv), 3*matrix)
	}
	l := layer{
		preAttnLN: m.Weights[prefix+".pre_attn_ln.scale"], postAttnLN: m.Weights[prefix+".post_attn_ln.scale"],
		preFFLN: m.Weights[prefix+".pre_ff_ln.scale"], postFFLN: m.Weights[prefix+".post_ff_ln.scale"],
		q: qkv[:matrix], k: qkv[matrix : 2*matrix], v: qkv[2*matrix:],
		o:       m.Weights[attn+".out.weight"],
		queryLN: m.Weights[attn+".query_ln.scale"], keyLN: m.Weights[attn+".key_ln.scale"],
		perDimScale: m.Weights[attn+".per_dim_scale.per_dim_scale"],
		ff0:         m.Weights[prefix+".ff0.weight"], ff1: m.Weights[prefix+".ff1.weight"],
	}
	for name, tensor := range map[string][]float32{
		"pre_attn_ln": l.preAttnLN, "post_attn_ln": l.postAttnLN, "pre_ff_ln": l.preFFLN, "post_ff_ln": l.postFFLN,
		"out": l.o, "query_ln": l.queryLN, "key_ln": l.keyLN, "per_dim_scale": l.perDimScale,
		"ff0": l.ff0, "ff1": l.ff1,
	} {
		if len(tensor) == 0 {
			return layer{}, fmt.Errorf("seriesforecast: layer %d tensor %s missing", index, name)
		}
	}
	return l, nil
}

// compiledQueryScale: log2(e)/sqrt(headDim) * softplus(per_dim_scale). The
// learned softplus scale replaces the fixed 1/sqrt(d) softmax scale (score
// scale is 1); log2(e)*softplus(0) ~= 1, so the product is ~1/sqrt(d) at the
// parameter's zero init.
func compiledQueryScale(perDimScale []float32, headDim int) []float32 {
	factor := math.Log2E / math.Sqrt(float64(headDim))
	out := make([]float32, headDim)
	for i := 0; i < headDim; i++ {
		out[i] = float32(factor * hostmath.Softplus(float64(perDimScale[i])))
	}
	return out
}

// layerForward runs one post-norm layer over the full patch sequence,
// writing the residual stream in place.
func (m *Model) layerForward(hidden []float32, l layer, invFreq []float64, seq int) {
	d, heads, hd := m.Dims.Hidden, m.Dims.Heads, m.Dims.HeadDim
	eps := m.Dims.RMSEps
	inNorm := make([]float32, seq*d)
	hostmath.RMSNormInto(inNorm, hidden, l.preAttnLN, seq, d, eps)

	width := heads * hd
	q := make([]float32, seq*width)
	k := make([]float32, seq*width)
	v := make([]float32, seq*width)
	hostmath.Linear(q, inNorm, l.q, seq, d, width)
	hostmath.Linear(k, inNorm, l.k, seq, d, width)
	hostmath.Linear(v, inNorm, l.v, seq, d, width)

	// RoPE before the per-head q/k norms (the post-norm family's ordering).
	for p := 0; p < seq; p++ {
		for h := 0; h < heads; h++ {
			hostmath.ApplyRotaryHalf(q[(p*heads+h)*hd:(p*heads+h+1)*hd], invFreq, p)
			hostmath.ApplyRotaryHalf(k[(p*heads+h)*hd:(p*heads+h+1)*hd], invFreq, p)
		}
	}
	hostmath.RMSNormInto(q, q, l.queryLN, seq*heads, hd, eps)
	hostmath.RMSNormInto(k, k, l.keyLN, seq*heads, hd, eps)
	scale := compiledQueryScale(l.perDimScale, hd)
	for row := 0; row < seq*heads; row++ {
		qRow := q[row*hd : (row+1)*hd]
		for dim := range qRow {
			qRow[dim] *= scale[dim]
		}
	}

	attn := make([]float32, seq*width)
	hostmath.CausalAttention(attn, q, k, v, seq, heads, hd)
	oProj := make([]float32, seq*d)
	hostmath.Linear(oProj, attn, l.o, seq, width, d)

	// Post-norm attention branch, plain residual.
	proj := make([]float32, seq*d)
	hostmath.RMSNormInto(proj, oProj, l.postAttnLN, seq, d, eps)
	for i := range hidden {
		hidden[i] += proj[i]
	}

	// Dense sequential feed-forward, post-normed, plain residual.
	ffIn := make([]float32, seq*d)
	hostmath.RMSNormInto(ffIn, hidden, l.preFFLN, seq, d, eps)
	inter := make([]float32, seq*d)
	hostmath.Linear(inter, ffIn, l.ff0, seq, d, d)
	hostmath.SiLUInPlace(inter)
	mlp := make([]float32, seq*d)
	hostmath.Linear(mlp, inter, l.ff1, seq, d, d)
	hostmath.RMSNormInto(mlp, mlp, l.postFFLN, seq, d, eps)
	for i := range hidden {
		hidden[i] += mlp[i]
	}
}
