// Backward for the dense causal-LM capability, composed from hostmath VJPs
// in checkpoint posture: only the per-layer residual-stream inputs are
// retained; every trace is recomputed from them. Gradients are keyed by the
// checkpoint tensor names an optimizer consumes. Head grads key by HeadName:
// tied, the one embedding slot accumulates BOTH the lm-head contribution and
// the input-lookup scatter; untied, lm_head.weight gets its own slot.
package densecausal

import (
	"fmt"
	"math"

	"overgo/internal/hostmath"
)

// Grads accumulates named weight gradients, allocated on first touch.
type Grads map[string][]float32

// LossAndGrads runs forward + full backward on one token batch: causal-LM
// mean CE (positions 0..n-2 predict tokens 1..n-1), the full logits, and
// every parameter gradient.
func (m *Model) LossAndGrads(tokens []int) (float64, []float32, Grads, error) {
	if len(tokens) < 2 {
		return 0, nil, nil, fmt.Errorf("densecausal: need at least 2 tokens, got %d", len(tokens))
	}
	states, err := m.forwardStates(tokens)
	if err != nil {
		return 0, nil, nil, err
	}
	invFreq := hostmath.RopeInvFreq(m.Dims.RopeTheta, m.Dims.HeadDim)
	return m.lossAndGradsFromStates(tokens, states, func(
		index int, input, outputGradient []float32, sequence int, gradients Grads,
	) ([]float32, error) {
		return m.layerBackward(index, input, outputGradient, invFreq, sequence, gradients)
	})
}

type layerGradient func(
	index int,
	input, outputGradient []float32,
	sequence int,
	gradients Grads,
) ([]float32, error)

func (m *Model) lossAndGradsFromStates(
	tokens []int,
	states [][]float32,
	backward layerGradient,
) (float64, []float32, Grads, error) {
	d := m.Dims
	seq := len(tokens)
	final := states[d.Layers]
	normed := make([]float32, seq*d.Hidden)
	hostmath.RMSNormInto(normed, final, m.Weights["model.norm.weight"], seq, d.Hidden, d.RMSEps)
	head := m.head()
	logits := make([]float32, seq*d.Vocab)
	hostmath.Linear(logits, normed, head, seq, d.Hidden, d.Vocab)

	g := Grads{}
	// Last position predicts nothing: its logits row carries zero gradient.
	dLogits := make([]float32, seq*d.Vocab)
	loss := hostmath.SoftmaxCrossEntropy(dLogits[:(seq-1)*d.Vocab], logits[:(seq-1)*d.Vocab], tokens[1:], seq-1, d.Vocab)

	gradHead := hostmath.GradientSlot(g, m.headName(), len(head))
	dNormed := make([]float32, seq*d.Hidden)
	hostmath.LinearBackward(dNormed, gradHead, nil, normed, head, dLogits, seq, d.Hidden, d.Vocab, false)
	dx := make([]float32, seq*d.Hidden)
	hostmath.RMSNormBackward(dx, hostmath.GradientSlot(g, "model.norm.weight", d.Hidden), final, m.Weights["model.norm.weight"], dNormed, seq, d.Hidden, d.RMSEps, false)

	for index := d.Layers - 1; index >= 0; index-- {
		var err error
		dx, err = backward(index, states[index], dx, seq, g)
		if err != nil {
			return 0, nil, nil, err
		}
	}
	// Input-embedding scatter (the second contribution when tied; the only
	// embedding contribution when untied).
	embed := m.Weights["model.embed_tokens.weight"]
	gradEmbed := hostmath.GradientSlot(g, "model.embed_tokens.weight", len(embed))
	scatterEmbeddingGradient(gradEmbed, dx, tokens, d.Hidden)
	return loss, logits, g, nil
}

func scatterEmbeddingGradient(embedding, gradient []float32, tokens []int, hidden int) {
	for token, id := range tokens {
		row := embedding[id*hidden : (id+1)*hidden]
		addInPlace(row, gradient[token*hidden:(token+1)*hidden])
	}
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
	hostmath.LinearBackward(dH, hostmath.GradientSlot(g, prefix+"mlp.down_proj.weight", d.Hidden*d.Intermediate), nil,
		h, l.down, dOut, seq, d.Intermediate, d.Hidden, false)
	dGate := make([]float32, seq*d.Intermediate)
	dUp := make([]float32, seq*d.Intermediate)
	hostmath.SiLUGateBackward(dGate, dUp, gate, up, dH)
	dHn := make([]float32, seq*d.Hidden)
	hostmath.LinearBackward(dHn, hostmath.GradientSlot(g, prefix+"mlp.gate_proj.weight", d.Intermediate*d.Hidden), nil,
		hn, l.gate, dGate, seq, d.Hidden, d.Intermediate, false)
	hostmath.LinearBackward(dHn, hostmath.GradientSlot(g, prefix+"mlp.up_proj.weight", d.Intermediate*d.Hidden), nil,
		hn, l.up, dUp, seq, d.Hidden, d.Intermediate, true)
	hostmath.RMSNormBackward(dh2, hostmath.GradientSlot(g, prefix+"post_attention_layernorm.weight", d.Hidden), h2, l.postLN, dHn, seq, d.Hidden, d.RMSEps, true)

	// Attention branch backward.
	dAttnCore := make([]float32, seq*width)
	hostmath.LinearBackward(dAttnCore, hostmath.GradientSlot(g, prefix+"self_attn.o_proj.weight", d.Hidden*width), nil,
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
		dbQ = hostmath.GradientSlot(g, prefix+"self_attn.q_proj.bias", width)
		dbK = hostmath.GradientSlot(g, prefix+"self_attn.k_proj.bias", kvWidth)
		dbV = hostmath.GradientSlot(g, prefix+"self_attn.v_proj.bias", kvWidth)
	}
	dXn := make([]float32, seq*d.Hidden)
	hostmath.LinearBackward(dXn, hostmath.GradientSlot(g, prefix+"self_attn.q_proj.weight", width*d.Hidden), dbQ,
		xn, l.q, dq, seq, d.Hidden, width, false)
	hostmath.LinearBackward(dXn, hostmath.GradientSlot(g, prefix+"self_attn.k_proj.weight", kvWidth*d.Hidden), dbK,
		xn, l.k, dk, seq, d.Hidden, kvWidth, true)
	hostmath.LinearBackward(dXn, hostmath.GradientSlot(g, prefix+"self_attn.v_proj.weight", kvWidth*d.Hidden), dbV,
		xn, l.v, dv, seq, d.Hidden, kvWidth, true)
	dx := dh2
	hostmath.RMSNormBackward(dx, hostmath.GradientSlot(g, prefix+"input_layernorm.weight", d.Hidden), x, l.inLN, dXn, seq, d.Hidden, d.RMSEps, true)
	return dx, nil
}
