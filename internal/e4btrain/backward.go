package e4btrain

import (
	"slices"

	"overgo/internal/hostmath"
)

// attentionVJP writes dq, dk and dv of windowed causal attention at score
// scale 1 for the output gradient dOut; window 0 is the whole causal
// prefix. q is [seq, heads*headDim]; k and v are [seq, kvHeads*headDim].
type attentionVJP func(dq, dk, dv, q, k, v, dOut []float32, seq, heads, kvHeads, headDim, window int) error

// hostAttention is the host kit's windowed attention VJP.
func hostAttention(dq, dk, dv, q, k, v, dOut []float32, seq, heads, kvHeads, headDim, window int) error {
	hostmath.WindowedCausalAttentionBackward(dq, dk, dv, q, k, v, dOut, seq, heads, kvHeads, headDim, window)
	return nil
}

// linearBackward is the VJP of dst = x·wᵀ for a row-major [out, in] w: dW
// accumulates dyᵀ·x, and dx is set, or accumulated when add.
func linearBackward(dx, dW, x, w, dy []float32, rows, inDim, outDim int, add bool) {
	hostmath.LinearWeightGradient(dW, x, dy, rows, inDim, outDim)
	if !add {
		hostmath.LinearBackwardInput(dx, dy, w, rows, inDim, outDim)
		return
	}
	partial := make([]float32, len(dx))
	hostmath.LinearBackwardInput(partial, dy, w, rows, inDim, outDim)
	for index, value := range partial {
		dx[index] += value
	}
}

// accumulate adds gradient into the buffer target holds, allocating it on
// first use.
func accumulate(target *[]float32, gradient []float32) {
	if *target == nil {
		*target = make([]float32, len(gradient))
	}
	for index, value := range gradient {
		(*target)[index] += value
	}
}

// differentiate runs the window forward and back, fills the gradient slab and
// returns the loss; attention is the windowed attention VJP every layer
// runs.
func (m *Model) differentiate(attention attentionVJP) (float64, error) {
	clear(m.gradients)
	tr, err := m.forward()
	if err != nil {
		return 0, err
	}
	g, tokens := m.shape, len(m.inputs)
	dLogits := make([]float32, len(tr.capped))
	loss := hostmath.SoftmaxCrossEntropy(dLogits, tr.capped, m.targets, tokens, g.vocab)
	if g.softcap > 0 {
		hostmath.SoftcapBackward(dLogits, dLogits, tr.logits, g.softcap)
	}
	scale(dLogits, g.logitScale)
	dNormed := make([]float32, tokens*g.hidden)
	hostmath.LinearBackwardInput(dNormed, dLogits, m.head, tokens, g.hidden, g.vocab)
	dx := make([]float32, tokens*g.hidden)
	hostmath.RMSNormBackward(dx, m.globalGradients.outputNorm, tr.final, m.global.outputNorm, dNormed, tokens, g.hidden, g.eps, false)
	// Keys and values a later layer attended reach their source layer's
	// projections; the source is always the earlier layer, so its backward
	// runs after every reader's.
	keyGradients := make([][]float32, len(g.layers))
	valueGradients := make([][]float32, len(g.layers))
	sides := make([][]float32, len(g.layers))
	for index := len(g.layers) - 1; index >= 0; index-- {
		if dx, err = m.layerBackward(index, tr, dx, attention, keyGradients, valueGradients, sides); err != nil {
			return 0, err
		}
	}
	m.perLayerBackward(tr, sides)
	return loss, nil
}

// layerBackward is the VJP of layerForward: it fills the layer's gradient
// views and returns the gradient at the layer input. The gradients of the
// keys and values the layer attended accumulate at their source layer.
func (m *Model) layerBackward(index int, tr *trace, dOutput []float32, attention attentionVJP, keyGradients, valueGradients, sides [][]float32) ([]float32, error) {
	g, layer, w, d := m.shape, m.shape.layers[index], m.layers[index], m.layerGradients[index]
	t := &tr.layers[index]
	tokens := len(m.inputs)
	queryWidth, keyWidth := layer.heads*layer.headDim, layer.kvHeads*layer.headDim

	dAdapted := dOutput
	if layer.scaled {
		var dScale float64
		dAdapted = make([]float32, len(dOutput))
		for position, value := range dOutput {
			dScale += float64(value) * float64(t.adapted[position])
			dAdapted[position] = value * w.outputScale[0]
		}
		d.outputScale[0] += float32(dScale)
	}
	dAdapterInput := dAdapted
	if g.perLayer > 0 {
		var err error
		dAdapterInput, sides[index], err = hostmath.PerLayerAdapterBackward(t.adapterInput, tr.sides[index],
			w.adapterGate, w.adapterProjection, w.adapterNorm, dAdapted,
			d.adapterGate, d.adapterProjection, d.adapterNorm, tokens, g.hidden, g.perLayer, g.eps, t.adapter)
		if err != nil {
			return nil, err
		}
	}

	dResidual := slices.Clone(dAdapterInput)
	dDown := make([]float32, tokens*g.hidden)
	hostmath.RMSNormBackward(dDown, d.postFeedForward, t.down, w.postFeedForward, dAdapterInput, tokens, g.hidden, g.eps, false)
	dActivated := make([]float32, tokens*layer.inter)
	linearBackward(dActivated, d.down, t.activated, w.down, dDown, tokens, layer.inter, g.hidden, false)
	dGate := make([]float32, len(dActivated))
	dUp := make([]float32, len(dActivated))
	for position, value := range t.gate {
		dUp[position] = dActivated[position] * float32(hostmath.GELUTanh(float64(value)))
		dGate[position] = dActivated[position] * t.up[position]
	}
	hostmath.GELUTanhBackward(dGate, t.gate, dGate)
	dFeedForward := make([]float32, tokens*g.hidden)
	linearBackward(dFeedForward, d.gate, t.feedForwardInput, w.gate, dGate, tokens, g.hidden, layer.inter, false)
	linearBackward(dFeedForward, d.up, t.feedForwardInput, w.up, dUp, tokens, g.hidden, layer.inter, true)
	hostmath.RMSNormBackward(dResidual, d.feedForwardNorm, t.residual, w.feedForwardNorm, dFeedForward, tokens, g.hidden, g.eps, true)

	dInput := slices.Clone(dResidual)
	dProjected := make([]float32, tokens*g.hidden)
	hostmath.RMSNormBackward(dProjected, d.postAttentionNorm, t.projected, w.postAttentionNorm, dResidual, tokens, g.hidden, g.eps, false)
	dAttention := make([]float32, tokens*queryWidth)
	linearBackward(dAttention, d.attentionOutput, t.attention, w.attentionOutput, dProjected, tokens, queryWidth, g.hidden, false)
	dQuery := make([]float32, tokens*queryWidth)
	dKeys := make([]float32, tokens*keyWidth)
	dValues := make([]float32, tokens*keyWidth)
	if err := attention(dQuery, dKeys, dValues, t.rotatedQuery, t.keys, t.values, dAttention,
		tokens, layer.heads, layer.kvHeads, layer.headDim, layer.window); err != nil {
		return nil, err
	}

	scale(dQuery, g.attentionScale)
	rotate(dQuery, layer.invFreq, tokens, layer.heads, layer.headDim, hostmath.RotaryHalfBackward)
	dQueryProjection := make([]float32, len(dQuery))
	hostmath.RMSNormBackward(dQueryProjection, d.queryNorm, t.query, w.queryNorm, dQuery, tokens*layer.heads, layer.headDim, g.eps, false)
	dNormed := make([]float32, tokens*g.hidden)
	linearBackward(dNormed, d.query, t.normed, w.query, dQueryProjection, tokens, g.hidden, queryWidth, false)
	if layer.source != index {
		accumulate(&keyGradients[layer.source], dKeys)
		accumulate(&valueGradients[layer.source], dValues)
	} else {
		accumulate(&keyGradients[index], dKeys)
		accumulate(&valueGradients[index], dValues)
		dKeys, dValues = keyGradients[index], valueGradients[index]
		rotate(dKeys, layer.invFreq, tokens, layer.kvHeads, layer.headDim, hostmath.RotaryHalfBackward)
		dKeyProjection := make([]float32, len(dKeys))
		hostmath.RMSNormBackward(dKeyProjection, d.keyNorm, t.key, w.keyNorm, dKeys, tokens*layer.kvHeads, layer.headDim, g.eps, false)
		linearBackward(dNormed, d.key, t.normed, w.key, dKeyProjection, tokens, g.hidden, keyWidth, true)
		dValueProjection := make([]float32, len(dValues))
		hostmath.RMSNormBackward(dValueProjection, nil, t.value, nil, dValues, tokens*layer.kvHeads, layer.headDim, g.eps, false)
		linearBackward(dNormed, d.value, t.normed, w.value, dValueProjection, tokens, g.hidden, keyWidth, true)
	}
	hostmath.RMSNormBackward(dInput, d.attentionNorm, t.input, w.attentionNorm, dNormed, tokens, g.hidden, g.eps, true)
	return dInput, nil
}

// perLayerBackward carries every layer's per-layer input gradient back
// through the per-layer norm and projection; the per-layer embedding rows
// are lexical and frozen, and so are the input rows the projection reads.
func (m *Model) perLayerBackward(tr *trace, sides [][]float32) {
	g := m.shape
	if g.perLayer == 0 {
		return
	}
	tokens, width := len(m.inputs), len(g.layers)*g.perLayer
	dNormed := make([]float32, tokens*width)
	for layer, dSide := range sides {
		for row := range tokens {
			for column := range g.perLayer {
				dNormed[row*width+layer*g.perLayer+column] = float32(float64(dSide[row*g.perLayer+column]) * g.perLayerScale)
			}
		}
	}
	dProjected := make([]float32, len(dNormed))
	hostmath.RMSNormBackward(dProjected, m.globalGradients.perLayerNorm, tr.projected, m.global.perLayerNorm, dNormed, tokens*len(g.layers), g.perLayer, g.eps, false)
	scale(dProjected, float64(hostmath.InvSqrt32(uint64(g.hidden))))
	hostmath.LinearWeightGradient(m.globalGradients.perLayerProjection, tr.embedded, dProjected, tokens, g.hidden, width)
}
