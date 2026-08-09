// Forward for the dense causal-LM capability: embedding lookup, pre-norm
// decoder layers (split-half rope, GQA causal attention, SiLU-gated MLP),
// final norm, tied lm head. Full-sequence prefill — training posture, no
// cache.
package densecausal

import (
	"fmt"
	"math"

	"overgo/internal/hostmath"
)

// layer: weight views for one decoder layer. qb/kb/vb are the optional
// q/k/v projection biases (qwen2); nil = absent (llama), one code path.
type layer struct {
	inLN, q, k, v, o []float32
	qb, kb, vb       []float32
	postLN           []float32
	gate, up, down   []float32
}

func (m *Model) layerWeights(index int) (layer, error) {
	prefix := fmt.Sprintf("model.layers.%d.", index)
	l := layer{
		inLN:   m.Weights[prefix+"input_layernorm.weight"],
		q:      m.Weights[prefix+"self_attn.q_proj.weight"],
		k:      m.Weights[prefix+"self_attn.k_proj.weight"],
		v:      m.Weights[prefix+"self_attn.v_proj.weight"],
		o:      m.Weights[prefix+"self_attn.o_proj.weight"],
		qb:     m.Weights[prefix+"self_attn.q_proj.bias"],
		kb:     m.Weights[prefix+"self_attn.k_proj.bias"],
		vb:     m.Weights[prefix+"self_attn.v_proj.bias"],
		postLN: m.Weights[prefix+"post_attention_layernorm.weight"],
		gate:   m.Weights[prefix+"mlp.gate_proj.weight"],
		up:     m.Weights[prefix+"mlp.up_proj.weight"],
		down:   m.Weights[prefix+"mlp.down_proj.weight"],
	}
	for _, w := range [][]float32{l.inLN, l.q, l.k, l.v, l.o, l.postLN, l.gate, l.up, l.down} {
		if w == nil {
			return layer{}, fmt.Errorf("densecausal: layer %d tensor missing", index)
		}
	}
	if m.Dims.AttnBias && (l.qb == nil || l.kb == nil || l.vb == nil) {
		return layer{}, fmt.Errorf("densecausal: layer %d attention bias missing", index)
	}
	return l, nil
}

// attnTrace: recomputable attention sub-block intermediates. qScaled folds
// the 1/sqrt(headDim) score scale (hostmath attention runs scale 1).
type attnTrace struct {
	kRoped, v    []float32
	qPost, kPost []float32
	qScaled      []float32
	attnCore     []float32
}

// attnSubForward: projections from the normed stream, split-half rope on
// q (per query head) and k (per kv head), scale fold, causal GQA attention.
func (m *Model) attnSubForward(l layer, xn []float32, invFreq []float64, seq int) attnTrace {
	d := m.Dims
	width := d.Heads * d.HeadDim
	kvWidth := d.KVHeads * d.HeadDim
	tr := attnTrace{
		qPost: make([]float32, seq*width),
		kPost: make([]float32, seq*kvWidth),
		v:     make([]float32, seq*kvWidth),
	}
	hostmath.Linear(tr.qPost, xn, l.q, seq, d.Hidden, width)
	hostmath.Linear(tr.kPost, xn, l.k, seq, d.Hidden, kvWidth)
	hostmath.Linear(tr.v, xn, l.v, seq, d.Hidden, kvWidth)
	if l.qb != nil {
		// Bias lands before rope, matching the HF projection layout.
		for p := 0; p < seq; p++ {
			hostmath.AddBias(tr.qPost[p*width:(p+1)*width], l.qb)
			hostmath.AddBias(tr.kPost[p*kvWidth:(p+1)*kvWidth], l.kb)
			hostmath.AddBias(tr.v[p*kvWidth:(p+1)*kvWidth], l.vb)
		}
	}
	tr.qScaled = append([]float32(nil), tr.qPost...)
	tr.kRoped = append([]float32(nil), tr.kPost...)
	for p := 0; p < seq; p++ {
		for h := 0; h < d.Heads; h++ {
			hostmath.ApplyRotaryHalf(tr.qScaled[(p*d.Heads+h)*d.HeadDim:(p*d.Heads+h+1)*d.HeadDim], invFreq, p)
		}
		for h := 0; h < d.KVHeads; h++ {
			hostmath.ApplyRotaryHalf(tr.kRoped[(p*d.KVHeads+h)*d.HeadDim:(p*d.KVHeads+h+1)*d.HeadDim], invFreq, p)
		}
	}
	scale := float32(1 / math.Sqrt(float64(d.HeadDim)))
	for i := range tr.qScaled {
		tr.qScaled[i] *= scale
	}
	tr.attnCore = make([]float32, seq*width)
	hostmath.CausalAttention(tr.attnCore, tr.qScaled, tr.kRoped, tr.v, seq, d.Heads, d.KVHeads, d.HeadDim)
	return tr
}

// layerForward advances the residual stream in place through one pre-norm
// layer: x += o(attn(norm(x))); x += down(silu(gate(norm(x)))*up(norm(x))).
func (m *Model) layerForward(x []float32, l layer, invFreq []float64, seq int) {
	d := m.Dims
	width := d.Heads * d.HeadDim
	xn := make([]float32, seq*d.Hidden)
	hostmath.RMSNormInto(xn, x, l.inLN, seq, d.Hidden, d.RMSEps)
	tr := m.attnSubForward(l, xn, invFreq, seq)
	attnOut := make([]float32, seq*d.Hidden)
	hostmath.Linear(attnOut, tr.attnCore, l.o, seq, width, d.Hidden)
	for i := range x {
		x[i] += attnOut[i]
	}
	hn := make([]float32, seq*d.Hidden)
	hostmath.RMSNormInto(hn, x, l.postLN, seq, d.Hidden, d.RMSEps)
	gate := make([]float32, seq*d.Intermediate)
	up := make([]float32, seq*d.Intermediate)
	hostmath.Linear(gate, hn, l.gate, seq, d.Hidden, d.Intermediate)
	hostmath.Linear(up, hn, l.up, seq, d.Hidden, d.Intermediate)
	hostmath.SiLUGate(gate, gate, up)
	mlp := make([]float32, seq*d.Hidden)
	hostmath.Linear(mlp, gate, l.down, seq, d.Intermediate, d.Hidden)
	for i := range x {
		x[i] += mlp[i]
	}
}

// forwardStates runs the stack retaining the residual stream at each layer
// input (checkpoint posture — backward recomputes everything else).
// states[l] is layer l's input; states[Layers] is the final pre-norm stream.
func (m *Model) forwardStates(tokens []int) ([][]float32, error) {
	d := m.Dims
	seq := len(tokens)
	if seq == 0 {
		return nil, fmt.Errorf("densecausal: empty token batch")
	}
	embed := m.Weights["model.embed_tokens.weight"]
	x := make([]float32, seq*d.Hidden)
	for t, id := range tokens {
		if id < 0 || id >= d.Vocab {
			return nil, fmt.Errorf("densecausal: token %d out of vocab %d", id, d.Vocab)
		}
		copy(x[t*d.Hidden:(t+1)*d.Hidden], embed[id*d.Hidden:(id+1)*d.Hidden])
	}
	invFreq := hostmath.RopeInvFreq(d.RopeTheta, d.HeadDim)
	states := make([][]float32, d.Layers+1)
	for index := 0; index < d.Layers; index++ {
		states[index] = append([]float32(nil), x...)
		l, err := m.layerWeights(index)
		if err != nil {
			return nil, err
		}
		m.layerForward(x, l, invFreq, seq)
	}
	states[d.Layers] = x
	return states, nil
}

// Logits: final norm then the tied lm head over every position;
// returns [seq*vocab] flat.
func (m *Model) logits(final []float32, seq int) []float32 {
	d := m.Dims
	normed := make([]float32, seq*d.Hidden)
	hostmath.RMSNormInto(normed, final, m.Weights["model.norm.weight"], seq, d.Hidden, d.RMSEps)
	out := make([]float32, seq*d.Vocab)
	hostmath.Linear(out, normed, m.Weights["model.embed_tokens.weight"], seq, d.Hidden, d.Vocab)
	return out
}

// Loss: forward-only causal-LM loss (mean CE, positions 0..n-2 predicting
// tokens 1..n-1) plus the full logits. The gradient path is LossAndGrads.
func (m *Model) Loss(tokens []int) (float64, []float32, error) {
	if len(tokens) < 2 {
		return 0, nil, fmt.Errorf("densecausal: need at least 2 tokens, got %d", len(tokens))
	}
	states, err := m.forwardStates(tokens)
	if err != nil {
		return 0, nil, err
	}
	seq := len(tokens)
	logits := m.logits(states[m.Dims.Layers], seq)
	dLogits := make([]float32, (seq-1)*m.Dims.Vocab)
	loss := hostmath.SoftmaxCrossEntropy(dLogits, logits[:(seq-1)*m.Dims.Vocab], tokens[1:], seq-1, m.Dims.Vocab)
	return loss, logits, nil
}
