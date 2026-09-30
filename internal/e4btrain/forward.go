package e4btrain

import (
	"math"

	"overgo/internal/hostmath"
)

// layerTrace holds what one layer's VJP reads back from its forward. Rows
// are tokens. keys and values are what the layer's attention read: its own
// normalized, rotated projections, or its source layer's.
type layerTrace struct {
	input, normed              []float32
	query, rotatedQuery        []float32 // projection before its norm; normalized, rotated and scaled
	key, value                 []float32 // own projections before their norms; nil when the layer reads its source's
	keys, values               []float32
	attention, projected       []float32 // attention output; its output projection before the post norm
	residual, feedForwardInput []float32
	gate, up, activated, down  []float32
	adapterInput, adapted      []float32 // adapter input; layer output before the output scale
	adapter                    hostmath.PerLayerAdapterTrace
}

// trace is one forward over the model's window.
type trace struct {
	embedded  []float32   // the window's scaled input rows
	projected []float32   // per-layer projection, scaled, before its norm: [tokens*layers, perLayer]
	sides     [][]float32 // each layer's per-layer input: [tokens, perLayer]
	layers    []layerTrace
	final     []float32 // last layer output
	logits    []float32 // [tokens, vocab], before the soft cap
	capped    []float32
}

func linear(x, w []float32, rows, inDim, outDim int) []float32 {
	dst := make([]float32, rows*outDim)
	hostmath.Linear(dst, x, w, rows, inDim, outDim)
	return dst
}

func rmsNorm(x, weight []float32, rows, width int, eps float64) []float32 {
	dst := make([]float32, len(x))
	hostmath.RMSNormInto(dst, x, weight, rows, width, eps)
	return dst
}

// rotate applies each head row's split-half rotation in place over the
// rotary width: position p turns pair i by p*invFreq[i].
func rotate(x []float32, invFreq []float64, tokens, heads, headDim int, apply func([]float32, []float64, int)) {
	width := len(invFreq) * pairWidth
	for position := range tokens {
		for head := range heads {
			start := (position*heads + head) * headDim
			apply(x[start:start+width], invFreq, position)
		}
	}
}

func scale(x []float32, factor float64) {
	if factor == 1 {
		return
	}
	for index := range x {
		x[index] = float32(float64(x[index]) * factor)
	}
}

// embedded is the window's input rows: the tied table's token rows at the
// declared input scale.
func (m *Model) embedded() []float32 {
	hidden := m.shape.hidden
	x := make([]float32, len(m.inputs)*hidden)
	for row, token := range m.inputs {
		copy(x[row*hidden:(row+1)*hidden], m.head[token*hidden:(token+1)*hidden])
	}
	scale(x, m.shape.embeddingScale)
	return x
}

// perLayerInputs is every layer's per-layer input: the input rows projected
// to every layer's width at the inverse square root of the hidden width and
// normalized, plus the window's per-layer embedding rows at the square root
// of the per-layer width, the sum weighted by the per-layer input scale.
func (m *Model) perLayerInputs(x []float32, tr *trace) {
	g := m.shape
	if g.perLayer == 0 {
		tr.sides = make([][]float32, len(g.layers))
		return
	}
	tokens, width := len(m.inputs), len(g.layers)*g.perLayer
	tr.projected = linear(x, m.global.perLayerProjection, tokens, g.hidden, width)
	scale(tr.projected, float64(hostmath.InvSqrt32(uint64(g.hidden))))
	normed := rmsNorm(tr.projected, m.global.perLayerNorm, tokens*len(g.layers), g.perLayer, g.eps)
	selected := float64(hostmath.Sqrt32(uint64(g.perLayer)))
	tr.sides = make([][]float32, len(g.layers))
	for layer := range g.layers {
		side := make([]float32, tokens*g.perLayer)
		for row := range tokens {
			for column := range g.perLayer {
				at := row*width + layer*g.perLayer + column
				sum := float64(normed[at]) + float64(m.perLayerRows[at])*selected
				side[row*g.perLayer+column] = float32(sum * g.perLayerScale)
			}
		}
		tr.sides[layer] = side
	}
}

// forward runs the decoder over the window and keeps what the VJP reads.
func (m *Model) forward() (*trace, error) {
	g := m.shape
	tr := &trace{layers: make([]layerTrace, len(g.layers))}
	x := m.embedded()
	tr.embedded = x
	m.perLayerInputs(x, tr)
	for index := range g.layers {
		output, err := m.layerForward(index, x, tr)
		if err != nil {
			return nil, err
		}
		x = output
	}
	tokens := len(m.inputs)
	tr.final = x
	normed := rmsNorm(x, m.global.outputNorm, tokens, g.hidden, g.eps)
	tr.logits = linear(normed, m.head, tokens, g.hidden, g.vocab)
	scale(tr.logits, g.logitScale)
	tr.capped = tr.logits
	if g.softcap > 0 {
		tr.capped = make([]float32, len(tr.logits))
		for index, value := range tr.logits {
			tr.capped[index] = float32(g.softcap * math.Tanh(float64(value)/g.softcap))
		}
	}
	return tr, nil
}

// layerForward runs one decoder layer: normalized attention with
// normalized, rotated queries and keys and unit-normalized values, a post
// norm and the residual; the gated GELU feed-forward block with its post
// norm and residual; the per-layer input adapter; the output scale.
func (m *Model) layerForward(index int, x []float32, tr *trace) ([]float32, error) {
	g, layer, w := m.shape, m.shape.layers[index], m.layers[index]
	tokens := len(m.inputs)
	queryWidth, keyWidth := layer.heads*layer.headDim, layer.kvHeads*layer.headDim
	t := &tr.layers[index]
	t.input = x
	t.normed = rmsNorm(x, w.attentionNorm, tokens, g.hidden, g.eps)
	t.query = linear(t.normed, w.query, tokens, g.hidden, queryWidth)
	t.rotatedQuery = rmsNorm(t.query, w.queryNorm, tokens*layer.heads, layer.headDim, g.eps)
	rotate(t.rotatedQuery, layer.invFreq, tokens, layer.heads, layer.headDim, hostmath.ApplyRotaryHalf)
	scale(t.rotatedQuery, g.attentionScale)
	if layer.source == index {
		t.key = linear(t.normed, w.key, tokens, g.hidden, keyWidth)
		t.keys = rmsNorm(t.key, w.keyNorm, tokens*layer.kvHeads, layer.headDim, g.eps)
		rotate(t.keys, layer.invFreq, tokens, layer.kvHeads, layer.headDim, hostmath.ApplyRotaryHalf)
		t.value = linear(t.normed, w.value, tokens, g.hidden, keyWidth)
		t.values = rmsNorm(t.value, nil, tokens*layer.kvHeads, layer.headDim, g.eps)
	} else {
		t.keys, t.values = tr.layers[layer.source].keys, tr.layers[layer.source].values
	}
	t.attention = make([]float32, tokens*queryWidth)
	hostmath.WindowedCausalAttention(t.attention, t.rotatedQuery, t.keys, t.values, tokens, layer.heads, layer.kvHeads, layer.headDim, layer.window)
	t.projected = linear(t.attention, w.attentionOutput, tokens, queryWidth, g.hidden)
	attended := rmsNorm(t.projected, w.postAttentionNorm, tokens, g.hidden, g.eps)
	t.residual = make([]float32, len(x))
	for position := range x {
		t.residual[position] = x[position] + attended[position]
	}

	t.feedForwardInput = rmsNorm(t.residual, w.feedForwardNorm, tokens, g.hidden, g.eps)
	t.gate = linear(t.feedForwardInput, w.gate, tokens, g.hidden, layer.inter)
	t.up = linear(t.feedForwardInput, w.up, tokens, g.hidden, layer.inter)
	t.activated = make([]float32, len(t.gate))
	for position, value := range t.gate {
		t.activated[position] = float32(hostmath.GELUTanh(float64(value))) * t.up[position]
	}
	t.down = linear(t.activated, w.down, tokens, layer.inter, g.hidden)
	fed := rmsNorm(t.down, w.postFeedForward, tokens, g.hidden, g.eps)
	t.adapterInput = make([]float32, len(x))
	for position := range x {
		t.adapterInput[position] = t.residual[position] + fed[position]
	}

	t.adapted = t.adapterInput
	if g.perLayer > 0 {
		var err error
		t.adapted, t.adapter, err = hostmath.PerLayerAdapterForward(t.adapterInput, tr.sides[index],
			w.adapterGate, w.adapterProjection, w.adapterNorm, tokens, g.hidden, g.perLayer, g.eps)
		if err != nil {
			return nil, err
		}
	}
	if !layer.scaled {
		return t.adapted, nil
	}
	output := make([]float32, len(x))
	for position, value := range t.adapted {
		output[position] = value * w.outputScale[0]
	}
	return output, nil
}

// Loss is the mean next-token cross-entropy of the window under the
// current weights.
func (m *Model) Loss() (float64, error) {
	tr, err := m.forward()
	if err != nil {
		return 0, err
	}
	scratch := make([]float32, len(tr.capped))
	return hostmath.SoftmaxCrossEntropy(scratch, tr.capped, m.targets, len(m.inputs), m.shape.vocab), nil
}
