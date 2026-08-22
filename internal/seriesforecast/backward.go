// Backward wiring for the patched time-series capability, composed from
// hostmath VJPs in the same order the grad goldens are factored: residual
// block, RevIN, running patch stats, patch embed, attention sub-block, full
// post-norm layer. Traces are recomputed from each stage's input
// (checkpoint posture — nothing retained beyond the residual stream).
package seriesforecast

import (
	"fmt"
	"math"

	"overgo/internal/extent"
	"overgo/internal/hostmath"
)

// Grads accumulates named weight gradients, allocated on first touch with
// the canonical tensor names an optimizer will consume.
type Grads map[string][]float32

// gradFor returns the accumulation slot only when the forward weight exists
// (heads ship without biases; absent weight means no gradient to hold).
func (m *Model) gradFor(g Grads, name string) []float32 {
	w := m.Weights[name]
	if w == nil {
		return nil
	}
	return hostmath.GradientSlot(g, name, len(w))
}

// residualBlockBackward: VJP of residualBlock; returns dx and accumulates
// the block's weight/bias gradients.
func (m *Model) residualBlockBackward(prefix string, x, dOut []float32, g Grads) ([]float32, error) {
	hiddenShape, ok := m.Shapes[prefix+".hidden_layer.weight"]
	if !ok || len(hiddenShape) != 2 {
		return nil, fmt.Errorf("seriesforecast: residual block %q hidden layer missing", prefix)
	}
	outputShape, ok := m.Shapes[prefix+".output_layer.weight"]
	if !ok || len(outputShape) != 2 {
		return nil, fmt.Errorf("seriesforecast: residual block %q output layer missing", prefix)
	}
	hiddenDim, inputDim, outputDim := hiddenShape[0], hiddenShape[1], outputShape[0]
	if len(x) != inputDim || len(dOut) != outputDim {
		return nil, fmt.Errorf("seriesforecast: residual block %q backward x=%d dOut=%d", prefix, len(x), len(dOut))
	}
	hidden := make([]float32, hiddenDim)
	hostmath.Linear(hidden, x, m.Weights[prefix+".hidden_layer.weight"], 1, inputDim, hiddenDim)
	hostmath.AddBias(hidden, m.Weights[prefix+".hidden_layer.bias"])
	activated := make([]float32, hiddenDim)
	copy(activated, hidden)
	hostmath.SiLUInPlace(activated)

	dActivated := make([]float32, hiddenDim)
	hostmath.LinearBackward(dActivated,
		hostmath.GradientSlot(g, prefix+".output_layer.weight", outputDim*hiddenDim),
		m.gradFor(g, prefix+".output_layer.bias"),
		activated, m.Weights[prefix+".output_layer.weight"], dOut, 1, hiddenDim, outputDim, false)
	dHidden := make([]float32, hiddenDim)
	hostmath.SiLUBackward(dHidden, hidden, dActivated)
	dx := make([]float32, inputDim)
	hostmath.LinearBackward(dx,
		hostmath.GradientSlot(g, prefix+".hidden_layer.weight", hiddenDim*inputDim),
		m.gradFor(g, prefix+".hidden_layer.bias"),
		x, m.Weights[prefix+".hidden_layer.weight"], dHidden, 1, inputDim, hiddenDim, false)
	hostmath.LinearBackward(dx,
		hostmath.GradientSlot(g, prefix+".residual_layer.weight", outputDim*inputDim),
		m.gradFor(g, prefix+".residual_layer.bias"),
		x, m.Weights[prefix+".residual_layer.weight"], dOut, 1, inputDim, outputDim, true)
	return dx, nil
}

// revinBackward: VJP of one patch's RevIN normalization with mu and sigma
// treated as independent inputs. When sigma sits below the tolerance the
// forward divisor was one and carries no sigma dependence.
func revinBackward(dNormed, x []float32, mu, sigma float64) (dx []float32, dMu, dSigma float64) {
	denom := sigma
	clamped := denom < revinTolerance
	if clamped {
		denom = 1
	}
	dx = make([]float32, len(x))
	for i := range x {
		g := float64(dNormed[i])
		dx[i] = float32(g / denom)
		dMu -= g / denom
		if !clamped {
			dSigma -= g * (float64(x[i]) - mu) / (denom * denom)
		}
	}
	return dx, dMu, dSigma
}

// patchStatsBackward adds the running-statistics path into dSeries: element
// k (legitimate, in patch p) contributes to mu[i] and sigma[i] for every
// i >= p through dmu_i/dx_k = 1/n_i and
// dsigma_i/dx_k = (x_k - mu_i)/(n_i*sigma_i).
func patchStatsBackward(dSeries, series, masks []float32, patchLen int, mu, sigma, dMu, dSigma []float64) {
	counts := make([]float64, len(mu))
	var n float64
	for i := range mu {
		for j := 0; j < patchLen; j++ {
			if masks[i*patchLen+j] == 0 {
				n++
			}
		}
		counts[i] = n
	}
	for k := range series {
		if masks[k] != 0 {
			continue
		}
		patch := k / patchLen
		var grad float64
		for i := patch; i < len(mu); i++ {
			if counts[i] == 0 {
				continue
			}
			grad += dMu[i] / counts[i]
			if sigma[i] > 0 {
				grad += dSigma[i] * (float64(series[k]) - mu[i]) / (counts[i] * sigma[i])
			}
		}
		dSeries[k] += float32(grad)
	}
}

// patchEmbedBackward: VJP of patchEmbed — tokenizer block, RevIN, and the
// running-stats coupling; returns dSeries and accumulates tokenizer grads.
func (m *Model) patchEmbedBackward(series, masks []float32, mu, sigma []float64, dOut []float32, g Grads) ([]float32, error) {
	p := m.Dims.PatchLen
	dSeries := make([]float32, len(series))
	dMu := make([]float64, len(mu))
	dSigma := make([]float64, len(mu))
	input := make([]float32, extent.PairedExtent*p)
	for i := range mu {
		denom := sigma[i]
		if denom < revinTolerance {
			denom = 1
		}
		for j := 0; j < p; j++ {
			normed := (float64(series[i*p+j]) - mu[i]) / denom
			if masks[i*p+j] != 0 {
				normed = 0
			}
			input[j] = float32(normed)
			input[p+j] = masks[i*p+j]
		}
		dInput, err := m.residualBlockBackward("tokenizer", input, dOut[i*m.Dims.Hidden:(i+1)*m.Dims.Hidden], g)
		if err != nil {
			return nil, err
		}
		dNormed := dInput[:p]
		for j := 0; j < p; j++ {
			if masks[i*p+j] != 0 {
				dNormed[j] = 0
			}
		}
		patchX := series[i*p : (i+1)*p]
		dx, dm, ds := revinBackward(dNormed, patchX, mu[i], sigma[i])
		for j := 0; j < p; j++ {
			dSeries[i*p+j] += dx[j]
		}
		dMu[i] += dm
		dSigma[i] += ds
	}
	patchStatsBackward(dSeries, series, masks, p, mu, sigma, dMu, dSigma)
	return dSeries, nil
}

// attnSubForward recomputes the attention sub-block trace from its input
// (the normed hidden): projections, roped copies, normed q/k, scaled q,
// and the attention core output.
type attnTrace struct {
	qPost, kPost, v          []float32
	qRoped, kRoped           []float32
	qNormed                  []float32
	qFinal, kFinal, attnCore []float32
}

func (m *Model) attnSubForward(l layer, x []float32, invFreq []float64, seq int) attnTrace {
	d, heads, hd := m.Dims.Hidden, m.Dims.Heads, m.Dims.HeadDim
	width := heads * hd
	tr := attnTrace{
		qPost: make([]float32, seq*width), kPost: make([]float32, seq*width), v: make([]float32, seq*width),
	}
	hostmath.Linear(tr.qPost, x, l.q, seq, d, width)
	hostmath.Linear(tr.kPost, x, l.k, seq, d, width)
	hostmath.Linear(tr.v, x, l.v, seq, d, width)
	tr.qRoped = append([]float32(nil), tr.qPost...)
	tr.kRoped = append([]float32(nil), tr.kPost...)
	for p := 0; p < seq; p++ {
		for h := 0; h < heads; h++ {
			hostmath.ApplyRotaryHalf(tr.qRoped[(p*heads+h)*hd:(p*heads+h+1)*hd], invFreq, p)
			hostmath.ApplyRotaryHalf(tr.kRoped[(p*heads+h)*hd:(p*heads+h+1)*hd], invFreq, p)
		}
	}
	tr.qNormed = make([]float32, seq*width)
	tr.kFinal = make([]float32, seq*width)
	hostmath.RMSNormInto(tr.qNormed, tr.qRoped, l.queryLN, seq*heads, hd, m.Dims.RMSEps)
	hostmath.RMSNormInto(tr.kFinal, tr.kRoped, l.keyLN, seq*heads, hd, m.Dims.RMSEps)
	scale := compiledQueryScale(l.perDimScale, hd)
	tr.qFinal = append([]float32(nil), tr.qNormed...)
	for row := 0; row < seq*heads; row++ {
		qRow := tr.qFinal[row*hd : (row+1)*hd]
		for dim := range qRow {
			qRow[dim] *= scale[dim]
		}
	}
	tr.attnCore = make([]float32, seq*width)
	hostmath.CausalAttention(tr.attnCore, tr.qFinal, tr.kFinal, tr.v, seq, heads, heads, hd)
	return tr
}

// attnSubBackward: VJP of the attention sub-block (projections through the
// output projection) for layer index; dOut arrives at the o-projection
// output, dx returns at the sub-block input.
func (m *Model) attnSubBackward(index int, l layer, x, dOut []float32, invFreq []float64, seq int, g Grads) []float32 {
	d, heads, hd := m.Dims.Hidden, m.Dims.Heads, m.Dims.HeadDim
	eps := m.Dims.RMSEps
	width := heads * hd
	prefix := fmt.Sprintf("stacked_xf.%d", index)
	attn := prefix + ".attn"
	tr := m.attnSubForward(l, x, invFreq, seq)

	dAttnCore := make([]float32, seq*width)
	hostmath.LinearBackward(dAttnCore, hostmath.GradientSlot(g, attn+".out.weight", d*width), nil,
		tr.attnCore, l.o, dOut, seq, width, d, false)
	dq := make([]float32, seq*width)
	dk := make([]float32, seq*width)
	dv := make([]float32, seq*width)
	hostmath.CausalAttentionBackward(dq, dk, dv, tr.qFinal, tr.kFinal, tr.v, dAttnCore, seq, heads, heads, hd)

	// Per-dim softplus query scale: parameter gradient reads the normed
	// (pre-scale) query; the incoming dq then scales down to the norm output.
	factor := math.Log2E / math.Sqrt(float64(hd))
	gradPds := hostmath.GradientSlot(g, attn+".per_dim_scale.per_dim_scale", hd)
	scale := compiledQueryScale(l.perDimScale, hd)
	for row := 0; row < seq*heads; row++ {
		for dim := 0; dim < hd; dim++ {
			i := row*hd + dim
			e := math.Exp(float64(l.perDimScale[dim]))
			gradPds[dim] += float32(float64(dq[i]) * float64(tr.qNormed[i]) * factor * e / (1 + e))
			dq[i] = float32(float64(dq[i]) * float64(scale[dim]))
		}
	}

	// RoPE-before-norm: norm backward at roped coordinates, then rotation
	// backward.
	dqNorm := make([]float32, seq*width)
	dkNorm := make([]float32, seq*width)
	hostmath.RMSNormBackward(dqNorm, hostmath.GradientSlot(g, attn+".query_ln.scale", hd), tr.qRoped, l.queryLN, dq, seq*heads, hd, eps, false)
	hostmath.RMSNormBackward(dkNorm, hostmath.GradientSlot(g, attn+".key_ln.scale", hd), tr.kRoped, l.keyLN, dk, seq*heads, hd, eps, false)
	for p := 0; p < seq; p++ {
		for h := 0; h < heads; h++ {
			hostmath.RotaryHalfBackward(dqNorm[(p*heads+h)*hd:(p*heads+h+1)*hd], invFreq, p)
			hostmath.RotaryHalfBackward(dkNorm[(p*heads+h)*hd:(p*heads+h+1)*hd], invFreq, p)
		}
	}

	matrix := d * d
	gradQKV := hostmath.GradientSlot(g, attn+".qkv_proj.weight", 3*matrix)
	dx := make([]float32, seq*d)
	hostmath.LinearBackward(dx, gradQKV[:matrix], nil, x, l.q, dqNorm, seq, d, width, false)
	hostmath.LinearBackward(dx, gradQKV[matrix:2*matrix], nil, x, l.k, dkNorm, seq, d, width, true)
	hostmath.LinearBackward(dx, gradQKV[2*matrix:], nil, x, l.v, dv, seq, d, width, true)
	return dx
}

// layerBackward: VJP of layerForward — the full post-norm layer. x is the
// layer INPUT residual stream (retained by the caller); dOut arrives at the
// layer output; dx returns at the input.
func (m *Model) layerBackward(index int, x, dOut []float32, invFreq []float64, seq int, g Grads) ([]float32, error) {
	l, err := m.layerWeights(index)
	if err != nil {
		return nil, err
	}
	d := m.Dims.Hidden
	eps := m.Dims.RMSEps
	prefix := fmt.Sprintf("stacked_xf.%d", index)

	// Recompute the residual-stream trace.
	inNorm := make([]float32, seq*d)
	hostmath.RMSNormInto(inNorm, x, l.preAttnLN, seq, d, eps)
	tr := m.attnSubForward(l, inNorm, invFreq, seq)
	oProj := make([]float32, seq*d)
	hostmath.Linear(oProj, tr.attnCore, l.o, seq, m.Dims.Heads*m.Dims.HeadDim, d)
	proj := make([]float32, seq*d)
	hostmath.RMSNormInto(proj, oProj, l.postAttnLN, seq, d, eps)
	hsum := make([]float32, seq*d)
	for i := range hsum {
		hsum[i] = x[i] + proj[i]
	}
	ffIn := make([]float32, seq*d)
	hostmath.RMSNormInto(ffIn, hsum, l.preFFLN, seq, d, eps)
	pre := make([]float32, seq*d)
	hostmath.Linear(pre, ffIn, l.ff0, seq, d, d)
	activated := append([]float32(nil), pre...)
	hostmath.SiLUInPlace(activated)
	mlp := make([]float32, seq*d)
	hostmath.Linear(mlp, activated, l.ff1, seq, d, d)

	// Feed-forward branch backward.
	dhsum := append([]float32(nil), dOut...)
	dMlp := make([]float32, seq*d)
	hostmath.RMSNormBackward(dMlp, hostmath.GradientSlot(g, prefix+".post_ff_ln.scale", d), mlp, l.postFFLN, dOut, seq, d, eps, false)
	dActivated := make([]float32, seq*d)
	hostmath.LinearBackward(dActivated, hostmath.GradientSlot(g, prefix+".ff1.weight", d*d), nil, activated, l.ff1, dMlp, seq, d, d, false)
	dPre := make([]float32, seq*d)
	hostmath.SiLUBackward(dPre, pre, dActivated)
	dFFIn := make([]float32, seq*d)
	hostmath.LinearBackward(dFFIn, hostmath.GradientSlot(g, prefix+".ff0.weight", d*d), nil, ffIn, l.ff0, dPre, seq, d, d, false)
	hostmath.RMSNormBackward(dhsum, hostmath.GradientSlot(g, prefix+".pre_ff_ln.scale", d), hsum, l.preFFLN, dFFIn, seq, d, eps, true)

	// Attention branch backward.
	dOProj := make([]float32, seq*d)
	hostmath.RMSNormBackward(dOProj, hostmath.GradientSlot(g, prefix+".post_attn_ln.scale", d), oProj, l.postAttnLN, dhsum, seq, d, eps, false)
	dInNorm := m.attnSubBackward(index, l, inNorm, dOProj, invFreq, seq, g)
	hostmath.RMSNormBackward(dhsum, hostmath.GradientSlot(g, prefix+".pre_attn_ln.scale", d), x, l.preAttnLN, dInNorm, seq, d, eps, true)
	return dhsum, nil
}
