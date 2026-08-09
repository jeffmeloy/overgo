// Backward for the dense causal-LM capability, composed from hostmath VJPs
// in checkpoint posture: only the per-layer residual-stream inputs are
// retained; every trace is recomputed from them. Gradients are keyed by the
// checkpoint tensor names an optimizer consumes. Tied embeddings: the one
// embedding gradient slot accumulates BOTH the lm-head contribution and the
// input-lookup scatter.
package densecausal

import (
	"fmt"
	"math"

	"overgo/internal/hostmath"
)

// Grads accumulates named weight gradients, allocated on first touch.
type Grads map[string][]float32

func (g Grads) slot(name string, n int) []float32 {
	if buf, ok := g[name]; ok {
		return buf
	}
	buf := make([]float32, n)
	g[name] = buf
	return buf
}

// LossAndGrads runs forward + full backward on one token batch: causal-LM
// mean CE (positions 0..n-2 predict tokens 1..n-1), the full logits, and
// every parameter gradient.
func (m *Model) LossAndGrads(tokens []int) (float64, []float32, Grads, error) {
	if len(tokens) < 2 {
		return 0, nil, nil, fmt.Errorf("densecausal: need at least 2 tokens, got %d", len(tokens))
	}
	d := m.Dims
	seq := len(tokens)
	states, err := m.forwardStates(tokens)
	if err != nil {
		return 0, nil, nil, err
	}
	final := states[d.Layers]
	normed := make([]float32, seq*d.Hidden)
	hostmath.RMSNormInto(normed, final, m.Weights["model.norm.weight"], seq, d.Hidden, d.RMSEps)
	embed := m.Weights["model.embed_tokens.weight"]
	logits := make([]float32, seq*d.Vocab)
	hostmath.Linear(logits, normed, embed, seq, d.Hidden, d.Vocab)

	g := Grads{}
	// Last position predicts nothing: its logits row carries zero gradient.
	dLogits := make([]float32, seq*d.Vocab)
	loss := hostmath.SoftmaxCrossEntropy(dLogits[:(seq-1)*d.Vocab], logits[:(seq-1)*d.Vocab], tokens[1:], seq-1, d.Vocab)

	gradEmbed := g.slot("model.embed_tokens.weight", len(embed))
	dNormed := make([]float32, seq*d.Hidden)
	hostmath.LinearBackward(dNormed, gradEmbed, nil, normed, embed, dLogits, seq, d.Hidden, d.Vocab, false)
	dx := make([]float32, seq*d.Hidden)
	hostmath.RMSNormBackward(dx, g.slot("model.norm.weight", d.Hidden), final, m.Weights["model.norm.weight"], dNormed, seq, d.Hidden, d.RMSEps, false)

	invFreq := hostmath.RopeInvFreq(d.RopeTheta, d.HeadDim)
	for index := d.Layers - 1; index >= 0; index-- {
		dx, err = m.layerBackward(index, states[index], dx, invFreq, seq, g)
		if err != nil {
			return 0, nil, nil, err
		}
	}
	// Input-embedding scatter (second tied contribution).
	for t, id := range tokens {
		row := gradEmbed[id*d.Hidden : (id+1)*d.Hidden]
		dRow := dx[t*d.Hidden : (t+1)*d.Hidden]
		for i := range row {
			row[i] += dRow[i]
		}
	}
	return loss, logits, g, nil
}

// layerBackward: VJP of layerForward. x is the layer input residual stream
// (retained); dOut arrives at the layer output; dx returns at the input.
func (m *Model) layerBackward(index int, x, dOut []float32, invFreq []float64, seq int, g Grads) ([]float32, error) {
	d := m.Dims
	l, err := m.layerWeights(index)
	if err != nil {
		return nil, err
	}
	prefix := fmt.Sprintf("model.layers.%d.", index)
	width := d.Heads * d.HeadDim
	kvWidth := d.KVHeads * d.HeadDim

	// Recompute the trace through the attention residual.
	xn := make([]float32, seq*d.Hidden)
	hostmath.RMSNormInto(xn, x, l.inLN, seq, d.Hidden, d.RMSEps)
	tr := m.attnSubForward(l, xn, invFreq, seq)
	attnOut := make([]float32, seq*d.Hidden)
	hostmath.Linear(attnOut, tr.attnCore, l.o, seq, width, d.Hidden)
	h2 := make([]float32, seq*d.Hidden)
	for i := range h2 {
		h2[i] = x[i] + attnOut[i]
	}
	hn := make([]float32, seq*d.Hidden)
	hostmath.RMSNormInto(hn, h2, l.postLN, seq, d.Hidden, d.RMSEps)
	gate := make([]float32, seq*d.Intermediate)
	up := make([]float32, seq*d.Intermediate)
	hostmath.Linear(gate, hn, l.gate, seq, d.Hidden, d.Intermediate)
	hostmath.Linear(up, hn, l.up, seq, d.Hidden, d.Intermediate)
	h := make([]float32, seq*d.Intermediate)
	hostmath.SiLUGate(h, gate, up)

	// MLP branch backward; dh2 carries the residual plus the norm path.
	dh2 := append([]float32(nil), dOut...)
	dH := make([]float32, seq*d.Intermediate)
	hostmath.LinearBackward(dH, g.slot(prefix+"mlp.down_proj.weight", d.Hidden*d.Intermediate), nil,
		h, l.down, dOut, seq, d.Intermediate, d.Hidden, false)
	dGate := make([]float32, seq*d.Intermediate)
	dUp := make([]float32, seq*d.Intermediate)
	hostmath.SiLUGateBackward(dGate, dUp, gate, up, dH)
	dHn := make([]float32, seq*d.Hidden)
	hostmath.LinearBackward(dHn, g.slot(prefix+"mlp.gate_proj.weight", d.Intermediate*d.Hidden), nil,
		hn, l.gate, dGate, seq, d.Hidden, d.Intermediate, false)
	hostmath.LinearBackward(dHn, g.slot(prefix+"mlp.up_proj.weight", d.Intermediate*d.Hidden), nil,
		hn, l.up, dUp, seq, d.Hidden, d.Intermediate, true)
	hostmath.RMSNormBackward(dh2, g.slot(prefix+"post_attention_layernorm.weight", d.Hidden), h2, l.postLN, dHn, seq, d.Hidden, d.RMSEps, true)

	// Attention branch backward.
	dAttnCore := make([]float32, seq*width)
	hostmath.LinearBackward(dAttnCore, g.slot(prefix+"self_attn.o_proj.weight", d.Hidden*width), nil,
		tr.attnCore, l.o, dh2, seq, width, d.Hidden, false)
	dq := make([]float32, seq*width)
	dk := make([]float32, seq*kvWidth)
	dv := make([]float32, seq*kvWidth)
	hostmath.CausalAttentionBackward(dq, dk, dv, tr.qScaled, tr.kRoped, tr.v, dAttnCore, seq, d.Heads, d.KVHeads, d.HeadDim)
	scale := float32(1 / math.Sqrt(float64(d.HeadDim)))
	for i := range dq {
		dq[i] *= scale
	}
	for p := 0; p < seq; p++ {
		for h := 0; h < d.Heads; h++ {
			hostmath.RotaryHalfBackward(dq[(p*d.Heads+h)*d.HeadDim:(p*d.Heads+h+1)*d.HeadDim], invFreq, p)
		}
		for h := 0; h < d.KVHeads; h++ {
			hostmath.RotaryHalfBackward(dk[(p*d.KVHeads+h)*d.HeadDim:(p*d.KVHeads+h+1)*d.HeadDim], invFreq, p)
		}
	}
	// Optional q/k/v bias grads (qwen2): dB sums dy rows; dx path unchanged.
	var dbQ, dbK, dbV []float32
	if l.qb != nil {
		dbQ = g.slot(prefix+"self_attn.q_proj.bias", width)
		dbK = g.slot(prefix+"self_attn.k_proj.bias", kvWidth)
		dbV = g.slot(prefix+"self_attn.v_proj.bias", kvWidth)
	}
	dXn := make([]float32, seq*d.Hidden)
	hostmath.LinearBackward(dXn, g.slot(prefix+"self_attn.q_proj.weight", width*d.Hidden), dbQ,
		xn, l.q, dq, seq, d.Hidden, width, false)
	hostmath.LinearBackward(dXn, g.slot(prefix+"self_attn.k_proj.weight", kvWidth*d.Hidden), dbK,
		xn, l.k, dk, seq, d.Hidden, kvWidth, true)
	hostmath.LinearBackward(dXn, g.slot(prefix+"self_attn.v_proj.weight", kvWidth*d.Hidden), dbV,
		xn, l.v, dv, seq, d.Hidden, kvWidth, true)
	dx := dh2
	hostmath.RMSNormBackward(dx, g.slot(prefix+"input_layernorm.weight", d.Hidden), x, l.inLN, dXn, seq, d.Hidden, d.RMSEps, true)
	return dx, nil
}
