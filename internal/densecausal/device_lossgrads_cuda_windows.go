//go:build windows

package densecausal

import (
	"fmt"

	"overgo/internal/cuda/device"
	"overgo/internal/devicemath"
	"overgo/internal/hostmath"
	"overgo/internal/tensor/dtype"
)

// deviceLossAndGrads is the device counterpart to LossAndGrads. The WHOLE layer
// stack runs forward then backward inside ONE resident session
// (devicemath.StackForwardBackwardResident): each layer's weights upload once and
// serve both its forward and its backward, and each layer's forward activation
// cache stays device-resident and is consumed by that layer's backward with no
// host download/reupload in between. The only host round-trips per step are the
// embedding lookup in, the final pre-norm stream out (for the host head / final
// RMSNorm / softmax-CE tail), and the tail's output-gradient back in. Loss, logits
// and every parameter gradient match LossAndGrads within fp32 tolerance.
// Attention bias is not yet supported (AttnBias must be false).
func (m *Model) deviceLossAndGrads(worker *device.Worker, tokens []int) (float64, []float32, Grads, error) {
	if len(tokens) < 2 {
		return 0, nil, nil, fmt.Errorf("densecausal: need at least 2 tokens, got %d", len(tokens))
	}
	d := m.Dims
	if ok, reason := DeviceTrainingSupported(d); !ok {
		return 0, nil, nil, fmt.Errorf("deviceLossAndGrads: %s", reason)
	}
	seq := len(tokens)
	width := d.Heads * d.HeadDim
	kvWidth := d.KVHeads * d.HeadDim

	// Embedding lookup (host): the stack's input residual, states[0].
	embed := m.Weights["model.embed_tokens.weight"]
	embeds := make([]float32, seq*d.Hidden)
	for token, id := range tokens {
		if id < 0 || id >= d.Vocab {
			return 0, nil, nil, fmt.Errorf("densecausal: token %d out of vocab %d", id, d.Vocab)
		}
		copy(embeds[token*d.Hidden:(token+1)*d.Hidden], embed[id*d.Hidden:(id+1)*d.Hidden])
	}

	// Gather per-layer weights (host slices; the session uploads each once).
	layers := make([]devicemath.LayerForwardWeights, d.Layers)
	for i := 0; i < d.Layers; i++ {
		l, err := m.layerWeights(i)
		if err != nil {
			return 0, nil, nil, err
		}
		layers[i] = devicemath.LayerForwardWeights{
			InLN: l.inLN, PostLN: l.postLN, Q: l.q, K: l.k, V: l.v, O: l.o,
			Gate: l.gate, Up: l.up, Down: l.down,
		}
	}
	invF32 := dtype.Float64SliceToFloat32(hostmath.RopeInvFreq(d.RopeTheta, d.HeadDim))

	// Host head / final RMSNorm / softmax-CE tail: consumes the final pre-norm
	// stream, writes the head + norm grads, and returns the last layer's output
	// gradient. Same math as lossAndGradsFromStates' head portion.
	g := Grads{}
	var loss float64
	var logits []float32
	tail := func(final []float32) ([]float32, error) {
		normed := make([]float32, seq*d.Hidden)
		hostmath.RMSNormInto(normed, final, m.Weights["model.norm.weight"], seq, d.Hidden, d.RMSEps)
		head := m.head()
		logits = make([]float32, seq*d.Vocab)
		hostmath.Linear(logits, normed, head, seq, d.Hidden, d.Vocab)
		dLogits := make([]float32, seq*d.Vocab)
		loss = hostmath.SoftmaxCrossEntropy(dLogits[:(seq-1)*d.Vocab], logits[:(seq-1)*d.Vocab], tokens[1:], seq-1, d.Vocab)
		gradHead := g.slot(m.headName(), len(head))
		dNormed := make([]float32, seq*d.Hidden)
		hostmath.LinearBackward(dNormed, gradHead, nil, normed, head, dLogits, seq, d.Hidden, d.Vocab, false)
		dx := make([]float32, seq*d.Hidden)
		hostmath.RMSNormBackward(dx, g.slot("model.norm.weight", d.Hidden), final, m.Weights["model.norm.weight"], dNormed, seq, d.Hidden, d.RMSEps, false)
		return dx, nil
	}

	dxEmbed, grads, err := devicemath.StackForwardBackwardResident(
		worker, embeds, layers, invF32, seq, d.Hidden, d.Heads, d.KVHeads, d.HeadDim, d.Intermediate, d.RMSEps, tail,
	)
	if err != nil {
		return 0, nil, nil, err
	}

	// Scatter each layer's weight grads into the named slots (same slots as
	// deviceLayerBackward / layerBackward).
	for i := 0; i < d.Layers; i++ {
		prefix := fmt.Sprintf("model.layers.%d.", i)
		r := grads[i]
		copy(g.slot(prefix+"mlp.gate_proj.weight", d.Intermediate*d.Hidden), r.DWGate)
		copy(g.slot(prefix+"mlp.up_proj.weight", d.Intermediate*d.Hidden), r.DWUp)
		copy(g.slot(prefix+"mlp.down_proj.weight", d.Hidden*d.Intermediate), r.DWDown)
		copy(g.slot(prefix+"post_attention_layernorm.weight", d.Hidden), r.DWPostLN)
		copy(g.slot(prefix+"self_attn.o_proj.weight", d.Hidden*width), r.DWO)
		copy(g.slot(prefix+"self_attn.q_proj.weight", width*d.Hidden), r.DWQ)
		copy(g.slot(prefix+"self_attn.k_proj.weight", kvWidth*d.Hidden), r.DWK)
		copy(g.slot(prefix+"self_attn.v_proj.weight", kvWidth*d.Hidden), r.DWV)
		copy(g.slot(prefix+"input_layernorm.weight", d.Hidden), r.DWInLN)
	}

	// Input-embedding scatter (tied: accumulates onto the head contribution
	// already in the slot; untied: the sole embedding contribution).
	gradEmbed := g.slot("model.embed_tokens.weight", len(embed))
	scatterEmbeddingGradient(gradEmbed, dxEmbed, tokens, d.Hidden)
	return loss, logits, g, nil
}
