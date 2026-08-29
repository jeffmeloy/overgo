// Forward for the dense causal-LM capability: embedding lookup, pre-norm
// decoder layers (split-half rope, GQA causal attention, SiLU-gated MLP),
// final norm, lm head (tied embedding or untied lm_head.weight). Full-sequence
// prefill — training posture, no cache.
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
	moe              *moeWeights
	names            layerTensorNames
}

type layerTensorNames struct {
	inLN, q, k, v, o string
	qb, kb, vb       string
	postLN           string
	gate, up, down   string
}

func compileLayer(weights map[string][]float32, index int, attentionBias bool, policy MoERouterPolicy, hidden int) (layer, error) {
	names := denseLayerTensorNames(index)
	l := layer{
		inLN: weights[names.inLN], postLN: weights[names.postLN],
		q: weights[names.q], k: weights[names.k], v: weights[names.v], o: weights[names.o],
		qb: weights[names.qb], kb: weights[names.kb], vb: weights[names.vb],
		gate: weights[names.gate], up: weights[names.up], down: weights[names.down],
		names: names,
	}
	for _, w := range [][]float32{l.inLN, l.q, l.k, l.v, l.o, l.postLN} {
		if w == nil {
			return layer{}, fmt.Errorf("densecausal: layer %d tensor missing", index)
		}
	}
	if attentionBias && (l.qb == nil || l.kb == nil || l.vb == nil) {
		return layer{}, fmt.Errorf("densecausal: layer %d attention bias missing", index)
	}
	// FFN kind derives from the layer's own tensors: a dense gate projection
	// makes a dense layer; a router plus expert set makes a routed layer.
	if l.gate != nil {
		if l.up == nil || l.down == nil {
			return layer{}, fmt.Errorf("densecausal: layer %d dense mlp incomplete", index)
		}
		return l, nil
	}
	if policy.TopK <= 0 {
		return layer{}, fmt.Errorf("densecausal: layer %d has no dense mlp and no declared mixture policy", index)
	}
	moe, err := compileMoELayer(weights, index, policy, hidden)
	if err != nil {
		return layer{}, err
	}
	l.moe = moe
	return l, nil
}

// compileMoELayer binds one routed layer's mixture tensors; the expert count
// derives from the tensors present, and the shared expert is optional.
func compileMoELayer(weights map[string][]float32, index int, policy MoERouterPolicy, hidden int) (*moeWeights, error) {
	prefix := fmt.Sprintf("model.layers.%d.mlp.", index)
	names := moeTensorNames{router: prefix + "gate.weight"}
	router := weights[names.router]
	if router == nil {
		return nil, fmt.Errorf("densecausal: layer %d router tensor %s missing", index, names.router)
	}
	w := &moeWeights{router: router}
	for expert := 0; ; expert++ {
		en := moeExpertNames{
			gate: fmt.Sprintf("%sexperts.%d.gate_proj.weight", prefix, expert),
			up:   fmt.Sprintf("%sexperts.%d.up_proj.weight", prefix, expert),
			down: fmt.Sprintf("%sexperts.%d.down_proj.weight", prefix, expert),
		}
		gate := weights[en.gate]
		if gate == nil {
			break
		}
		if weights[en.up] == nil || weights[en.down] == nil {
			return nil, fmt.Errorf("densecausal: layer %d expert %d incomplete", index, expert)
		}
		w.experts = append(w.experts, moeExpert{gate: gate, up: weights[en.up], down: weights[en.down]})
		names.experts = append(names.experts, en)
	}
	if len(w.experts) == 0 || policy.TopK > len(w.experts) {
		return nil, fmt.Errorf("densecausal: layer %d has %d experts for top_k %d", index, len(w.experts), policy.TopK)
	}
	shared := moeExpertNames{
		gate: prefix + "shared_experts.gate_proj.weight",
		up:   prefix + "shared_experts.up_proj.weight",
		down: prefix + "shared_experts.down_proj.weight",
	}
	for e, expert := range w.experts {
		if len(expert.gate) != policy.ExpertInter*hidden || len(expert.up) != policy.ExpertInter*hidden || len(expert.down) != hidden*policy.ExpertInter {
			return nil, fmt.Errorf("densecausal: layer %d expert %d geometry differs from declared moe intermediate %d", index, e, policy.ExpertInter)
		}
	}
	if sg := weights[shared.gate]; sg != nil {
		if weights[shared.up] == nil || weights[shared.down] == nil {
			return nil, fmt.Errorf("densecausal: layer %d shared expert incomplete", index)
		}
		if hidden <= 0 || len(sg)%hidden != 0 {
			return nil, fmt.Errorf("densecausal: layer %d shared expert gate len=%d not divisible by hidden %d", index, len(sg), hidden)
		}
		w.shared = &moeExpert{gate: sg, up: weights[shared.up], down: weights[shared.down]}
		w.sharedInter = len(sg) / hidden
		names.shared = &shared
	}
	w.names = names
	return w, nil
}

func denseLayerTensorNames(index int) layerTensorNames {
	prefix := fmt.Sprintf("model.layers.%d.", index)
	return layerTensorNames{
		inLN: prefix + "input_layernorm.weight", postLN: prefix + "post_attention_layernorm.weight",
		q: prefix + "self_attn.q_proj.weight", k: prefix + "self_attn.k_proj.weight", v: prefix + "self_attn.v_proj.weight", o: prefix + "self_attn.o_proj.weight",
		qb: prefix + "self_attn.q_proj.bias", kb: prefix + "self_attn.k_proj.bias", vb: prefix + "self_attn.v_proj.bias",
		gate: prefix + "mlp.gate_proj.weight", up: prefix + "mlp.up_proj.weight", down: prefix + "mlp.down_proj.weight",
	}
}

func addInPlace(destination, values []float32) {
	for index := range destination {
		destination[index] += values[index]
	}
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
		for p := range seq {
			hostmath.AddBias(tr.qPost[p*width:(p+1)*width], l.qb)
			hostmath.AddBias(tr.kPost[p*kvWidth:(p+1)*kvWidth], l.kb)
			hostmath.AddBias(tr.v[p*kvWidth:(p+1)*kvWidth], l.vb)
		}
	}
	tr.qScaled = append([]float32(nil), tr.qPost...)
	tr.kRoped = append([]float32(nil), tr.kPost...)
	for p := range seq {
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
func (m *Model) layerForward(x []float32, l layer, invFreq []float64, seq int) error {
	d := m.Dims
	width := d.Heads * d.HeadDim
	xn := make([]float32, seq*d.Hidden)
	hostmath.RMSNormInto(xn, x, l.inLN, seq, d.Hidden, d.RMSEps)
	tr := m.attnSubForward(l, xn, invFreq, seq)
	attnOut := make([]float32, seq*d.Hidden)
	hostmath.Linear(attnOut, tr.attnCore, l.o, seq, width, d.Hidden)
	addInPlace(x, attnOut)
	hn := make([]float32, seq*d.Hidden)
	hostmath.RMSNormInto(hn, x, l.postLN, seq, d.Hidden, d.RMSEps)
	if l.moe != nil {
		mixture, _, err := moeForward(hn, *l.moe, seq, d.Hidden, d.MoE)
		if err != nil {
			return err
		}
		addInPlace(x, mixture)
		return nil
	}
	gate := make([]float32, seq*d.Intermediate)
	up := make([]float32, seq*d.Intermediate)
	hostmath.Linear(gate, hn, l.gate, seq, d.Hidden, d.Intermediate)
	hostmath.Linear(up, hn, l.up, seq, d.Hidden, d.Intermediate)
	hostmath.SiLUGate(gate, gate, up)
	mlp := make([]float32, seq*d.Hidden)
	hostmath.Linear(mlp, gate, l.down, seq, d.Intermediate, d.Hidden)
	addInPlace(x, mlp)
	return nil
}

// forwardStates runs the stack retaining the residual stream at each layer
// input (checkpoint posture — backward recomputes everything else).
// states[l] is layer l's input; states[Layers] is the final pre-norm stream.
func (m *Model) forwardStates(tokens []int) ([][]float32, error) {
	d := m.Dims
	invFreq := hostmath.RopeInvFreq(d.RopeTheta, d.HeadDim)
	return m.retainedForwardStates(tokens, func(x []float32, index, seq int) error {
		return m.layerForward(x, m.layers[index], invFreq, seq)
	})
}

func (m *Model) retainedForwardStates(
	tokens []int,
	forward func(x []float32, index, sequence int) error,
) ([][]float32, error) {
	d := m.Dims
	seq := len(tokens)
	if seq == 0 {
		return nil, fmt.Errorf("densecausal: empty token batch")
	}
	embed := m.tensors.embedding.values
	x := make([]float32, seq*d.Hidden)
	for token, id := range tokens {
		if id < 0 || id >= d.Vocab {
			return nil, fmt.Errorf("densecausal: token %d out of vocab %d", id, d.Vocab)
		}
		copy(x[token*d.Hidden:(token+1)*d.Hidden], embed[id*d.Hidden:(id+1)*d.Hidden])
	}
	states := make([][]float32, d.Layers+1)
	for index := range d.Layers {
		states[index] = append([]float32(nil), x...)
		if err := forward(x, index, seq); err != nil {
			return nil, err
		}
	}
	states[d.Layers] = x
	return states, nil
}

// logits applies the compiled final norm and head bindings.
func (m *Model) logits(final []float32, seq int) []float32 {
	d := m.Dims
	normed := make([]float32, seq*d.Hidden)
	hostmath.RMSNormInto(normed, final, m.tensors.finalNorm.values, seq, d.Hidden, d.RMSEps)
	out := make([]float32, seq*d.Vocab)
	hostmath.Linear(out, normed, m.tensors.head.values, seq, d.Hidden, d.Vocab)
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
