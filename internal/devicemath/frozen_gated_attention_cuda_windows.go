//go:build windows

package devicemath

import (
	"errors"
	"math"

	"overgo/internal/cuda/driver"
	"overgo/internal/hostmath"
	"overgo/internal/tensor/dtype"
)

// sigmoidGate is activated_gate_f32's activation code for a gate taken through the sigmoid.
const sigmoidGate = 0

// negatedLogits is scale_f32's factor that turns the gate's logits to their negation, whose sigmoid the derivative of
// the gate's takes.
const negatedLogits = -1.0

// rotaryHalves are the halves of a head's turned values, whose pairs, one from each, rope_half_f32 turns together.
const rotaryHalves = 2

// FrozenGatedAttentionLayer is one Qwen3.5 gated full-attention decoder layer on the card with every weight frozen, as
// hostmath.HybridDecoderLayerForward computes an attention layer with an output gate: out = h + MLP(RMSNorm(h,
// PostNorm)), h = x + Wo·(Attention(xn) ⊙ sigmoid(Gate·xn)), xn = RMSNorm(x, InputNorm). The attention is causal over
// grouped query heads, its q and k normalized a head and turned on the first RopeWidth values of each head by the
// token's position, q scaled by 1/sqrt(HeadDim). Its weights are put on the card once in F32; Forward keeps on the card
// what Backward reads, and Backward gives the gradient with respect to the layer's input alone.
type FrozenGatedAttentionLayer struct {
	frozenLayer
	weights frozenGatedAttentionWeights
	cache   frozenGatedAttentionCache
	work    frozenGatedAttentionWork
	rotary  int
}

// frozenGatedAttentionWeights are the mix's weights on the card, the projections row-major [out, in], with the rotary
// inverse frequencies.
type frozenGatedAttentionWeights struct {
	q, k, v, o, gate, qNorm, kNorm, frequencies driver.DevicePtr
}

// frozenGatedAttentionCache is what the forward keeps for the backward: the q and k projections before their norms;
// the gate's logits; the queries (normalized, turned and scaled), keys (normalized and turned) and values head-major,
// [heads, tokens, head dim]; the attention's probabilities, [heads, tokens, tokens]; and its output before the gate.
type frozenGatedAttentionCache struct {
	qProjected, kProjected, gate, queries, keys, values, probabilities, attention driver.DevicePtr
}

// frozenGatedAttentionWork is the passes' scratch: query rows token-major (two) and head-major (two), key rows
// token-major (two) and head-major, a matrix of scores a query head, and the turned prefixes of every query head.
type frozenGatedAttentionWork struct {
	rows, rows2, heads, heads2, keyRows, keyRows2, keyHeads, scores, prefixes driver.DevicePtr
}

// OpenFrozenGatedAttentionLayer puts the gated full-attention decoder layer w on the card for sequences of d.Tokens
// tokens, every weight frozen, with the cache its forward keeps and the scratch its backward takes, in one allocation
// the layer releases on Close; it runs on this session's module and cuBLAS handle. The layer carries its output gate,
// as Qwen3.5's attention layers do, its query heads are a whole multiple of its key heads, and its heads and rotary
// width are even. Its memory grows with the tokens' rows of the layer's widths and with their square a query head: the
// probabilities the forward keeps and their gradient.
func (s *ResidentOpsSession) OpenFrozenGatedAttentionLayer(w hostmath.HybridLayerWeights, d hostmath.HybridLayerDims) (*FrozenGatedAttentionLayer, error) {
	a := d.Attn
	tokens, hidden, inter, heads, kvHeads, width := d.Tokens, d.Hidden, d.Inter, a.Heads, a.KVHeads, a.HeadDim
	rotary := hostmath.RopeWidth(a.RopeDim, width)
	if s == nil || s.closed || w.IsLinear || w.Attn.Gate == nil || min(tokens, hidden, inter, heads, kvHeads, width) <= 0 || heads%kvHeads != 0 ||
		width%rotaryHalves != 0 || rotary%rotaryHalves != 0 || !(a.RopeTheta > 0) || a.Tokens != tokens || a.Hidden != hidden {
		return nil, errors.New("frozen gated-attention layer: needs an open session and a gated attention layer of positive geometry, its query heads a multiple of its key heads, and its heads and rotary width even")
	}
	queryDim, keyDim, scores := heads*width, kvHeads*width, heads*tokens*tokens
	l := &FrozenGatedAttentionLayer{rotary: rotary}
	lw, c, k := &l.weights, &l.cache, &l.work
	weights := []frozenWeight{
		{frozenBuffer{&lw.q, queryDim * hidden}, w.Attn.Wq}, {frozenBuffer{&lw.k, keyDim * hidden}, w.Attn.Wk},
		{frozenBuffer{&lw.v, keyDim * hidden}, w.Attn.Wv}, {frozenBuffer{&lw.o, hidden * queryDim}, w.Attn.Wo},
		{frozenBuffer{&lw.gate, queryDim * hidden}, w.Attn.Gate},
		{frozenBuffer{&lw.qNorm, width}, w.Attn.QNorm}, {frozenBuffer{&lw.kNorm, width}, w.Attn.KNorm},
		{frozenBuffer{&lw.frequencies, rotary / rotaryHalves}, dtype.Float64SliceToFloat32(hostmath.RopeInvFreq(a.RopeTheta, rotary))},
	}
	buffers := []frozenBuffer{
		{&c.qProjected, tokens * queryDim}, {&c.kProjected, tokens * keyDim}, {&c.gate, tokens * queryDim},
		{&c.queries, tokens * queryDim}, {&c.keys, tokens * keyDim}, {&c.values, tokens * keyDim},
		{&c.probabilities, scores}, {&c.attention, tokens * queryDim},
		{&k.rows, tokens * queryDim}, {&k.rows2, tokens * queryDim}, {&k.heads, tokens * queryDim}, {&k.heads2, tokens * queryDim},
		{&k.keyRows, tokens * keyDim}, {&k.keyRows2, tokens * keyDim}, {&k.keyHeads, tokens * keyDim},
		{&k.scores, scores}, {&k.prefixes, tokens * heads * rotary},
	}
	if err := l.open(s, "frozen gated-attention layer", l, w, d, weights, buffers); err != nil {
		return nil, err
	}
	return l, nil
}

// forwardMix queues hostmath's gated attention mix over the normed rows.
func (l *FrozenGatedAttentionLayer) forwardMix(r *frozenRun) {
	d, at, w, c, k, a := l.d, l.d.Attn, &l.weights, &l.cache, &l.work, &l.around
	tokens, hidden, heads, kvHeads, width := d.Tokens, d.Hidden, at.Heads, at.KVHeads, at.HeadDim
	queries, keys := tokens*heads*width, tokens*kvHeads*width
	r.project(a.normed, w.q, c.qProjected, tokens, hidden, heads*width)
	r.project(a.normed, w.k, c.kProjected, tokens, hidden, kvHeads*width)
	r.project(a.normed, w.v, k.keyRows, tokens, hidden, kvHeads*width)
	r.project(a.normed, w.gate, c.gate, tokens, hidden, heads*width)
	r.each(string(headMajorPack), keys, k.keyRows, c.values, tokens, kvHeads, width)
	// q and k normalized a head and turned, q scaled by 1/sqrt(head dim), then each laid out head-major.
	r.rows("weighted_rms_norm_f32", tokens*heads, c.qProjected, w.qNorm, k.rows, width, tokens*heads, 1, at.Eps)
	l.rotate(r, k.rows, heads, ropeForward)
	r.each("scale_f32", queries, k.rows, k.rows, 1/math.Sqrt(float64(width)), queries)
	r.each(string(headMajorPack), queries, k.rows, c.queries, tokens, heads, width)
	r.rows("weighted_rms_norm_f32", tokens*kvHeads, c.kProjected, w.kNorm, k.keyRows, width, tokens*kvHeads, 1, at.Eps)
	l.rotate(r, k.keyRows, kvHeads, ropeForward)
	r.each(string(headMajorPack), keys, k.keyRows, c.keys, tokens, kvHeads, width)
	// Each query head's scores against its key head, softmaxed over the causal prefix in place, then its values.
	l.grouped(r, true, width, tokens, c.queries, c.keys, c.probabilities)
	r.each("causal_softmax_batched_f32", heads*tokens, c.probabilities, c.probabilities, heads, tokens)
	l.grouped(r, false, tokens, width, c.probabilities, c.values, k.heads)
	r.each(string(headMajorUnpack), queries, k.heads, c.attention, tokens, heads, width)
	// gated = attention ⊙ sigmoid(gate), then Wo.
	r.each("activated_gate_f32", queries, c.gate, c.attention, k.rows, sigmoidGate, queries)
	r.project(k.rows, w.o, a.mixed, tokens, heads*width, hidden)
}

// backwardMix queues the VJP of forwardMix with respect to the normed rows alone, as hostmath's attention mix backward
// gives its input gradient, from the cache the last forward kept.
func (l *FrozenGatedAttentionLayer) backwardMix(r *frozenRun) {
	d, at, w, c, k, a := l.d, l.d.Attn, &l.weights, &l.cache, &l.work, &l.around
	tokens, hidden, heads, kvHeads, width := d.Tokens, d.Hidden, at.Heads, at.KVHeads, at.HeadDim
	queries := tokens * heads * width
	// gated = attention ⊙ σ(gate): dAttention = dGated ⊙ σ(gate), and the gate's logits take dGated ⊙ attention ⊙
	// σ(gate) ⊙ σ(-gate), the sigmoid's derivative.
	r.back(a.mixed, w.o, k.rows, tokens, hidden, heads*width, false)
	r.each("activated_gate_f32", queries, c.gate, k.rows, k.rows2, sigmoidGate, queries)
	r.each("multiply_f32", queries, k.rows2, c.attention, k.rows, queries)
	r.each("scale_f32", queries, c.gate, k.heads2, negatedLogits, queries)
	r.each("activated_gate_f32", queries, k.heads2, k.rows, k.rows, sigmoidGate, queries)
	r.back(k.rows, w.gate, a.normed, tokens, heads*width, hidden, false)
	// The attention's VJP head-major: dP = dAttention·Vᵀ, dS its softmax VJP in place, dV = Pᵀ·dAttention, dK = dSᵀ·Q
	// and dQ = dS·K, a key head's gradients summed over the query heads that share it.
	r.each(string(headMajorPack), queries, k.rows2, k.heads, tokens, heads, width)
	l.grouped(r, true, width, tokens, k.heads, c.values, k.scores)
	r.each("softmax_backward_f32", heads*tokens, c.probabilities, k.scores, k.scores, heads*tokens, tokens)
	r.batched(true, false, tokens, tokens, width, heads, c.probabilities, tokens*tokens, k.heads, tokens*width, k.heads2, tokens*width)
	l.sharedHeads(r, k.heads2, k.keyRows)
	r.back(k.keyRows, w.v, a.normed, tokens, kvHeads*width, hidden, true)
	r.batched(true, false, tokens, tokens, width, heads, k.scores, tokens*tokens, c.queries, tokens*width, k.heads2, tokens*width)
	l.sharedHeads(r, k.heads2, k.keyRows)
	l.grouped(r, false, tokens, width, k.scores, c.keys, k.heads)
	r.each(string(headMajorUnpack), queries, k.heads, k.rows, tokens, heads, width)
	// Back through q's scale, the turns and the norms to the projections.
	r.each("scale_f32", queries, k.rows, k.rows, 1/math.Sqrt(float64(width)), queries)
	l.rotate(r, k.rows, heads, ropeGradient)
	l.rotate(r, k.keyRows, kvHeads, ropeGradient)
	r.rows("rms_norm_input_backward_f32", tokens*heads, k.rows, c.qProjected, w.qNorm, k.rows2, width, tokens*heads, at.Eps)
	r.back(k.rows2, w.q, a.normed, tokens, heads*width, hidden, true)
	r.rows("rms_norm_input_backward_f32", tokens*kvHeads, k.keyRows, c.kProjected, w.kNorm, k.keyRows2, width, tokens*kvHeads, at.Eps)
	r.back(k.keyRows2, w.k, a.normed, tokens, kvHeads*width, hidden, true)
}

// rotate turns the first rotary values of each of rows' heads, [tokens, heads, head dim], by rope_half_f32's angles at
// the token's position, or by their transpose with direction ropeGradient, through the work's prefixes when the turn
// leaves a tail.
func (l *FrozenGatedAttentionLayer) rotate(r *frozenRun, rows driver.DevicePtr, heads int, direction ropeDirection) {
	tokens, width, rotary, prefixes := l.d.Tokens, l.d.Attn.HeadDim, l.rotary, rows
	partial := rotary < width
	if partial {
		prefixes = l.work.prefixes
		r.each("strided_row_copy_f32", tokens*heads*rotary, rows, prefixes, tokens*heads, rotary, width, 0, rotary, 0)
	}
	r.each(string(direction), tokens*heads*rotary/rotaryHalves, prefixes, l.weights.frequencies, tokens, heads, rotary)
	if partial {
		r.each("strided_row_copy_f32", tokens*heads*rotary, prefixes, rows, tokens*heads, rotary, rotary, 0, width, 0)
	}
}

// grouped queues, for each key head, the products of its query heads' matrices in perHead, [heads, tokens, inner],
// with its own matrix in shared, [key heads, tokens, head dim], transposed as asked, into output, [heads, tokens,
// columns].
func (l *FrozenGatedAttentionLayer) grouped(r *frozenRun, transposeShared bool, inner, columns int, perHead, shared, output driver.DevicePtr) {
	at, tokens := l.d.Attn, l.d.Tokens
	group := at.Heads / at.KVHeads
	for head := range at.KVHeads {
		first := head * group
		r.batched(false, transposeShared, tokens, inner, columns, group, ptrAt(perHead, first*tokens*inner), tokens*inner,
			ptrAt(shared, head*tokens*at.HeadDim), 0, ptrAt(output, first*tokens*columns), tokens*columns)
	}
}

// sharedHeads sums expanded, a gradient a query head head-major, over the query heads that share each key head, and
// lays the sums out token-major in output, [tokens, key heads, head dim].
func (l *FrozenGatedAttentionLayer) sharedHeads(r *frozenRun, expanded, output driver.DevicePtr) {
	at, tokens := l.d.Attn, l.d.Tokens
	keys := tokens * at.KVHeads * at.HeadDim
	r.each("reduce_gqa_heads_f32", keys, expanded, l.work.keyHeads, at.Heads, at.KVHeads, tokens*at.HeadDim, keys)
	r.each(string(headMajorUnpack), keys, l.work.keyHeads, output, tokens, at.KVHeads, at.HeadDim)
}
