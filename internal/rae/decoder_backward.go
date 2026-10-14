package rae

import (
	"errors"
	"fmt"
	"math"

	"overgo/internal/checked"
	"overgo/internal/hostmath"
)

// layerTrace is what one decoder block computed on the way forward that its
// backward pass reads: the stream entering the block and after its attention,
// both norms' outputs, the attention's projections and mix, and the
// feed-forward's values before and after its activation.
type layerTrace struct {
	input, normed1, q, k, v, attended []float32
	middle, normed2, inner, activated []float32
}

// decoderTrace is a forward pass kept for its backward pass.
type decoderTrace struct {
	rows   []float32
	layers []layerTrace
	final  []float32
	normed []float32
}

// decoderGradients are the gradients of a loss with respect to a decoder's
// weights, each the shape of the weight it belongs to.
type decoderGradients struct {
	embedWeight, embedBias []float32
	class                  []float32
	normWeight, normBias   []float32
	predWeight, predBias   []float32
	layers                 []layer
}

// decodeTraced is decodeTokens keeping what its backward pass reads.
func (d *vitDecoder) decodeTraced(rows []float32) ([]float32, decoderTrace, error) {
	tokens := d.grid * d.grid
	if !checked.Equal(len(rows), d.latent*tokens) {
		return nil, decoderTrace{}, fmt.Errorf("rae: %d token values, want %d", len(rows), d.latent*tokens)
	}
	sequence := tokens + 1
	trace := decoderTrace{rows: rows}
	hidden := make([]float32, sequence*d.hidden)
	copy(hidden, d.class)
	linear(hidden[d.hidden:], rows, d.embedWeight, d.embedBias, tokens, d.latent, d.hidden)
	for index, value := range d.position {
		hidden[index] += value
	}
	for _, l := range d.layers {
		step := layerTrace{input: append([]float32(nil), hidden...)}
		step.normed1 = make([]float32, len(hidden))
		hostmath.LayerNormInto(step.normed1, hidden, l.norm1Weight, l.norm1Bias, sequence, d.hidden, d.epsilon)
		step.q, step.k, step.v = make([]float32, len(hidden)), make([]float32, len(hidden)), make([]float32, len(hidden))
		linear(step.q, step.normed1, l.query, l.queryBias, sequence, d.hidden, d.hidden)
		linear(step.k, step.normed1, l.key, l.keyBias, sequence, d.hidden, d.hidden)
		linear(step.v, step.normed1, l.value, l.valueBias, sequence, d.hidden, d.hidden)
		step.attended = hostmath.BatchedAttentionF32(step.q, step.k, step.v, 1, sequence, sequence, d.hidden, d.heads)
		projected := make([]float32, len(hidden))
		linear(projected, step.attended, l.output, l.outputBias, sequence, d.hidden, d.hidden)
		add(hidden, projected)
		step.middle = append([]float32(nil), hidden...)
		step.normed2 = make([]float32, len(hidden))
		hostmath.LayerNormInto(step.normed2, hidden, l.norm2Weight, l.norm2Bias, sequence, d.hidden, d.epsilon)
		step.inner = make([]float32, sequence*d.intermediate)
		linear(step.inner, step.normed2, l.up, l.upBias, sequence, d.hidden, d.intermediate)
		step.activated = append([]float32(nil), step.inner...)
		hostmath.GELUErfInPlace(step.activated)
		linear(projected, step.activated, l.down, l.downBias, sequence, d.intermediate, d.hidden)
		add(hidden, projected)
		trace.layers = append(trace.layers, step)
	}
	trace.final = append([]float32(nil), hidden...)
	trace.normed = make([]float32, len(hidden))
	hostmath.LayerNormInto(trace.normed, hidden, d.normWeight, d.normBias, sequence, d.hidden, d.epsilon)
	channels := len(d.pixelMean)
	width := d.patch * channels
	predicted := make([]float32, tokens*d.patch*width)
	linear(predicted, trace.normed[d.hidden:], d.predWeight, d.predBias, tokens, d.hidden, d.patch*width)
	side := d.grid * d.patch
	pixels := make([]float32, channels*side*side)
	for row := range d.grid {
		for column := range d.grid {
			patch := predicted[(row*d.grid+column)*d.patch*width:]
			for y := range d.patch {
				for x := range d.patch {
					for channel := range channels {
						value := patch[(y*d.patch+x)*channels+channel]
						pixels[(channel*side+row*d.patch+y)*side+column*d.patch+x] = value*d.pixelStd[channel] + d.pixelMean[channel]
					}
				}
			}
		}
	}
	for _, value := range pixels {
		if !checked.Finite32(value) {
			return nil, decoderTrace{}, errors.New("rae: the decoder produced a non-finite value")
		}
	}
	return pixels, trace, nil
}

// backward carries the gradient of a loss with respect to the decoded
// pixels back through a traced forward pass. It returns the gradient with
// respect to the tokens the decoder read, and, when gradients is not nil,
// adds the gradient with respect to every weight into it.
func (d *vitDecoder) backward(trace decoderTrace, dPixels []float32, gradients *decoderGradients) ([]float32, error) {
	tokens := d.grid * d.grid
	sequence := tokens + 1
	channels := len(d.pixelMean)
	side := d.grid * d.patch
	width := d.patch * channels
	if !checked.Equal(len(dPixels), channels*side*side) || !checked.Equal(len(trace.layers), len(d.layers)) {
		return nil, errors.New("rae: the pixel gradient or the trace does not fit the decoder")
	}
	// A weight's gradient is taken only when the caller keeps them.
	var zero decoderGradients
	keep := gradients != nil
	if !keep {
		gradients = &zero
		gradients.layers = make([]layer, len(d.layers))
	}
	dPredicted := make([]float32, tokens*d.patch*width)
	for row := range d.grid {
		for column := range d.grid {
			patch := dPredicted[(row*d.grid+column)*d.patch*width:]
			for y := range d.patch {
				for x := range d.patch {
					for channel := range channels {
						patch[(y*d.patch+x)*channels+channel] = dPixels[(channel*side+row*d.patch+y)*side+column*d.patch+x] * d.pixelStd[channel]
					}
				}
			}
		}
	}
	dNormed := make([]float32, sequence*d.hidden)
	hostmath.LinearBackward(dNormed[d.hidden:], gradients.predWeight, gradients.predBias, trace.normed[d.hidden:], d.predWeight, dPredicted, tokens, d.hidden, d.patch*width, false)
	dHidden := make([]float32, sequence*d.hidden)
	hostmath.LayerNormBackward(dHidden, gradients.normWeight, gradients.normBias, trace.final, d.normWeight, dNormed, sequence, d.hidden, d.epsilon, false)
	scale := 1 / math.Sqrt(float64(d.hidden/d.heads))
	dActivated := make([]float32, sequence*d.intermediate)
	dBranch := make([]float32, sequence*d.hidden)
	dAttended := make([]float32, sequence*d.hidden)
	dq, dk, dv := make([]float32, sequence*d.hidden), make([]float32, sequence*d.hidden), make([]float32, sequence*d.hidden)
	for index := len(d.layers) - 1; index >= 0; index-- {
		l, step, g := d.layers[index], trace.layers[index], gradients.layers[index]
		// The feed-forward branch: the stream's gradient reaches it whole,
		// and what it sends back through its norm joins the stream's.
		hostmath.LinearBackward(dActivated, g.down, g.downBias, step.activated, l.down, dHidden, sequence, d.intermediate, d.hidden, false)
		hostmath.GELUErfBackward(dActivated, step.inner, dActivated)
		hostmath.LinearBackward(dBranch, g.up, g.upBias, step.normed2, l.up, dActivated, sequence, d.hidden, d.intermediate, false)
		hostmath.LayerNormBackward(dHidden, g.norm2Weight, g.norm2Bias, step.middle, l.norm2Weight, dBranch, sequence, d.hidden, d.epsilon, true)
		// The attention branch, the same way.
		hostmath.LinearBackward(dAttended, g.output, g.outputBias, step.attended, l.output, dHidden, sequence, d.hidden, d.hidden, false)
		hostmath.ScaledMaskedBidirectionalAttentionBackward(dq, dk, dv, step.q, step.k, step.v, dAttended, sequence, sequence, d.heads, d.heads, d.hidden/d.heads, scale, nil)
		hostmath.LinearBackward(dBranch, g.query, g.queryBias, step.normed1, l.query, dq, sequence, d.hidden, d.hidden, false)
		hostmath.LinearBackward(dBranch, g.key, g.keyBias, step.normed1, l.key, dk, sequence, d.hidden, d.hidden, true)
		hostmath.LinearBackward(dBranch, g.value, g.valueBias, step.normed1, l.value, dv, sequence, d.hidden, d.hidden, true)
		hostmath.LayerNormBackward(dHidden, g.norm1Weight, g.norm1Bias, step.input, l.norm1Weight, dBranch, sequence, d.hidden, d.epsilon, true)
	}
	if keep {
		add(gradients.class, dHidden[:d.hidden])
	}
	dRows := make([]float32, tokens*d.latent)
	hostmath.LinearBackward(dRows, gradients.embedWeight, gradients.embedBias, trace.rows, d.embedWeight, dHidden[d.hidden:], tokens, d.latent, d.hidden, false)
	return dRows, nil
}
