// Test-only host VJPs for alternate-state finite-difference parity.
// Every op works on the reference.Value layout used by the forward (Data index
// = token*width + feature, width = Shape.Dims[0], tokens = Shape.Dims[1]) and
// is composed from the FD-verified hostmath VJPs. Each function is checkpoint
// style: it recomputes whatever forward intermediates it needs from its inputs.
// GELU is differentiated as the smooth tanh-GELU (GELUTanhPrime); the forward's
// fp16 round-trip in roundedGELUTanh is a serving-only quantization, not part of
// the training-leg derivative. FD gates in gemma3n_backward_test.go are the
// oracle (no adaptive E4B grad goldens exist).
package inference

import (
	"math"

	"overgo/internal/hostmath"
	"overgo/internal/model"
	"overgo/internal/tensor"
	"overgo/internal/tensor/reference"
)

func g3nRows(v reference.Value) int  { return int(v.Shape.Dims[1]) }
func g3nWidth(v reference.Value) int { return int(v.Shape.Dims[0]) }

func g3nZeros(shape tensor.Shape) reference.Value {
	return reference.Value{Shape: shape, Data: make([]float32, shapeElems(shape))}
}

func shapeElems(s tensor.Shape) int {
	n := 1
	for i := 0; i < int(s.Rank); i++ {
		n *= int(s.Dims[i])
	}
	return n
}

// g3nMatMulBackward: VJP of alternateLinear(weight, input). dInput and dWeight are
// ACCUMULATED into (caller pre-clears). weight is [inner,out]-shaped but stored
// [out,inner] row-major, exactly hostmath's Linear weight layout.
func g3nMatMulBackward(dInput, dWeight []float32, input, weight, dOut reference.Value) {
	inner := int(weight.Shape.Dims[0])
	out := int(weight.Shape.Dims[1])
	tokens := int(input.Shape.Dims[1])
	hostmath.LinearBackward(dInput, dWeight, nil, input.Data, weight.Data, dOut.Data, tokens, inner, out, true)
}

// g3nMatMulSliceBackward: VJP of alternateLinearSlice over slice s of a rank-3
// [inner,out,count] weight. dInput accumulates; dWeight accumulates into slice s.
func g3nMatMulSliceBackward(dInput, dWeight3d []float32, input, weight3d reference.Value, slice int, dOut reference.Value) {
	inner := int(weight3d.Shape.Dims[0])
	out := int(weight3d.Shape.Dims[1])
	size := inner * out
	start := slice * size
	sliceW := reference.Value{Shape: tensor.MustShape(weight3d.Shape.Dims[0], weight3d.Shape.Dims[1]), Data: weight3d.Data[start : start+size]}
	g3nMatMulBackward(dInput, dWeight3d[start:start+size], input, sliceW, dOut)
}

// g3nWeightedRMSBackward: VJP of alternateRMSNorm. dInput/dWeight accumulate.
func g3nWeightedRMSBackward(dInput, dWeight []float32, input, weight, dOut reference.Value, eps float32) {
	rows := int(input.Shape.Dims[1])
	d := int(input.Shape.Dims[0])
	hostmath.RMSNormBackward(dInput, dWeight, input.Data, weight.Data, dOut.Data, rows, d, float64(eps), true)
}

// g3nModalitiesBackward: VJP of alternateStateModalities. Given dOut at the tanh'd
// router output, accumulates dInput and the router/router-norm weight grads.
// Recomputes the RMS-normed/scaled input and the pre-tanh router logits.
func g3nModalitiesBackward(dInput, dRouter, dRouterNorm []float32, input reference.Value, layer model.HostLayer, spec model.Spec, dOut reference.Value) {
	eps := spec.RMSNormEpsilon
	// Recompute forward trace.
	normalized, _ := alternateRMSNorm(input, *layer.AltUpRouterNorm, eps)
	scaled := normalized.Clone()
	invEmb := float32(1) / float32(spec.EmbeddingLength)
	for i := range scaled.Data {
		scaled.Data[i] *= invEmb
	}
	preTanh, _ := alternateLinear(*layer.AltUpRouter, scaled)
	// tanh backward.
	dPre := make([]float32, len(dOut.Data))
	hostmath.TanhBackward(dPre, preTanh.Data, dOut.Data)
	dPreV := reference.Value{Shape: preTanh.Shape, Data: dPre}
	// router matmul backward -> dScaled, dRouter.
	dScaled := make([]float32, len(scaled.Data))
	g3nMatMulBackward(dScaled, dRouter, scaled, *layer.AltUpRouter, dPreV)
	// undo the /EmbeddingLength scaling.
	for i := range dScaled {
		dScaled[i] *= invEmb
	}
	dNorm := reference.Value{Shape: normalized.Shape, Data: dScaled}
	// weighted-RMS backward -> dInput, dRouterNorm.
	g3nWeightedRMSBackward(dInput, dRouterNorm, input, *layer.AltUpRouterNorm, dNorm, eps)
}

// g3nInitializeAltUpBackward: VJP of initializeAlternateStates. dInput and the
// projection (rank-3) grad accumulate.
func g3nInitializeAltUpBackward(dInput, dProjection []float32, input, projection reference.Value, dStates []reference.Value) {
	rows := g3nRows(input)
	width := g3nWidth(input)
	// states[0] = input clone.
	for i := range dInput {
		dInput[i] += dStates[0].Data[i]
	}
	for index := 1; index < len(dStates); index++ {
		projected, _ := alternateLinearSlice(projection, index-1, input)
		// matchMagnitude(projected, input): arg input=projected, target=input.
		dProjected := make([]float32, len(projected.Data))
		dInputFromMatch := make([]float32, len(input.Data))
		hostmath.MatchMagnitudeBackward(dProjected, dInputFromMatch, projected.Data, input.Data, dStates[index].Data, rows, width)
		for i := range dInput {
			dInput[i] += dInputFromMatch[i]
		}
		dProjV := reference.Value{Shape: projected.Shape, Data: dProjected}
		g3nMatMulSliceBackward(dInput, dProjection, input, projection, index-1, dProjV)
	}
}

// g3nMergeAltUpBackward: VJP of mergeAlternateStates. dStates (len N) and the
// unembedding (rank-3) grad accumulate. active = spec.AltUpActive.
func g3nMergeAltUpBackward(dStates []reference.Value, dUnembedding []float32, states []reference.Value, unembedding reference.Value, active int, dOutput reference.Value) {
	n := len(states)
	rows := g3nRows(states[active])
	width := g3nWidth(states[active])
	invN := float32(1) / float32(n)
	g := make([]float32, len(dOutput.Data))
	for i := range g {
		g[i] = dOutput.Data[i] * invN
	}
	// base: result += states[active].
	for i := range dStates[active].Data {
		dStates[active].Data[i] += g[i]
	}
	gV := reference.Value{Shape: dOutput.Shape, Data: g}
	for index := 1; index < n; index++ {
		projected, _ := alternateLinearSlice(unembedding, index-1, states[index])
		// matchMagnitude(projected, states[active]).
		dProjected := make([]float32, len(projected.Data))
		dActiveFromMatch := make([]float32, len(states[active].Data))
		hostmath.MatchMagnitudeBackward(dProjected, dActiveFromMatch, projected.Data, states[active].Data, gV.Data, rows, width)
		for i := range dStates[active].Data {
			dStates[active].Data[i] += dActiveFromMatch[i]
		}
		dProjV := reference.Value{Shape: projected.Shape, Data: dProjected}
		g3nMatMulSliceBackward(dStates[index].Data, dUnembedding, states[index], unembedding, index-1, dProjV)
	}
}

// g3nPredictBackward: VJP of predictAlternateStates. dStates (len count) accumulate;
// the predict-coefficient, router and router-norm weight grads accumulate.
func g3nPredictBackward(dStates []reference.Value, dPredictCoeff, dRouter, dRouterNorm []float32, states []reference.Value, layer model.HostLayer, spec model.Spec, dResult []reference.Value) {
	count := len(states)
	active := int(spec.AltUpActive)
	tokens := int(states[0].Shape.Dims[1])
	width := int(states[0].Shape.Dims[0])
	// Recompute modalities + coefficients.
	modalities, _ := alternateStateModalities(
		states[active], layer, spec.EmbeddingLength, spec.RMSNormEpsilon,
	)
	coefficients, _ := alternateLinear(*layer.AltUpPredictCoefficient, modalities)
	dCoeff := reference.Value{Shape: coefficients.Shape, Data: make([]float32, len(coefficients.Data))}
	// result[o][t,f] = states[o][t,f] + sum_s coeff[t,o*count+s]*states[s][t,f].
	for o := 0; o < count; o++ {
		for i := range dStates[o].Data {
			dStates[o].Data[i] += dResult[o].Data[i]
		}
	}
	for t := 0; t < tokens; t++ {
		for o := 0; o < count; o++ {
			for s := 0; s < count; s++ {
				c := coefficients.Data[t*count*count+o*count+s]
				var dc float64
				for f := 0; f < width; f++ {
					pos := t*width + f
					dr := dResult[o].Data[pos]
					dc += float64(dr) * float64(states[s].Data[pos])
					dStates[s].Data[pos] += dr * c
				}
				dCoeff.Data[t*count*count+o*count+s] += float32(dc)
			}
		}
	}
	// coefficients matmul backward -> dModalities, dPredictCoeff.
	dModalities := make([]float32, len(modalities.Data))
	g3nMatMulBackward(dModalities, dPredictCoeff, modalities, *layer.AltUpPredictCoefficient, dCoeff)
	dModV := reference.Value{Shape: modalities.Shape, Data: dModalities}
	// modalities backward -> dStates[active], dRouter, dRouterNorm.
	g3nModalitiesBackward(dStates[active].Data, dRouter, dRouterNorm, states[active], layer, spec, dModV)
}

// g3nCorrectAndInjectBackward: VJP of correctAndInjectAlternateStates. Accumulates grads
// for predictions (len count), activated, perLayer, and every weight touched:
// correct-coefficient, router, router-norm, correct-scale, per-layer input-gate,
// per-layer projection, per-layer post-norm.
type g3nCorrectGrads struct {
	dCorrectCoeff []float32
	dRouter       []float32
	dRouterNorm   []float32
	dCorrectScale []float32
	dInputGate    []float32
	dProjection   []float32
	dPostNorm     []float32
}

func g3nCorrectAndInjectBackward(dPredictions []reference.Value, dActivated, dPerLayer []float32, wg *g3nCorrectGrads,
	predictions []reference.Value, activated, perLayer reference.Value, layer model.HostLayer, spec model.Spec, dResult []reference.Value) {
	count := len(predictions)
	active := int(spec.AltUpActive)
	tokens := int(activated.Shape.Dims[1])
	width := int(activated.Shape.Dims[0])
	eps := spec.RMSNormEpsilon

	// --- Recompute forward trace (correction mix + PLE path). ---
	modalities, _ := alternateStateModalities(
		activated, layer, spec.EmbeddingLength, spec.RMSNormEpsilon,
	)
	coefficients, _ := alternateLinear(*layer.AltUpCorrectCoefficient, modalities)
	// result[active] before PLE.
	resultActive := predictions[active].Clone()
	for t := 0; t < tokens; t++ {
		coeff := coefficients.Data[t*count+active] + 1
		for f := 0; f < width; f++ {
			pos := t*width + f
			innovation := activated.Data[pos] - predictions[active].Data[pos]
			resultActive.Data[pos] += innovation * coeff
		}
	}
	scaleWidth := int(resultActive.Shape.Dims[0])
	scaled := resultActive.Clone()
	for i := range scaled.Data {
		scaled.Data[i] *= layer.AltUpCorrectScale.Data[i%scaleWidth]
	}
	rawGate, _ := alternateLinear(*layer.PerLayerInputGate, scaled)
	gate := rawGate.Clone()
	for i := range gate.Data {
		gate.Data[i] = roundedGELUTanh(gate.Data[i]) * perLayer.Data[i]
	}
	injectionPre, _ := alternateLinear(*layer.PerLayerProjection, gate)

	// --- PLE backward. injection added to result[1..count-1]. ---
	dInjection := make([]float32, len(injectionPre.Data))
	for index := 1; index < count; index++ {
		for i := range dInjection {
			dInjection[i] += dResult[index].Data[i]
		}
	}
	dInjectionV := reference.Value{Shape: injectionPre.Shape, Data: dInjection}
	dInjectionPre := make([]float32, len(injectionPre.Data))
	g3nWeightedRMSBackward(dInjectionPre, wg.dPostNorm, injectionPre, *layer.PerLayerPostNorm, dInjectionV, eps)
	dInjectionPreV := reference.Value{Shape: injectionPre.Shape, Data: dInjectionPre}
	dGate := make([]float32, len(gate.Data))
	g3nMatMulBackward(dGate, wg.dProjection, gate, *layer.PerLayerProjection, dInjectionPreV)
	// gate[i] = GELU(rawGate[i]) * perLayer[i].
	dRawGate := make([]float32, len(rawGate.Data))
	for i := range dRawGate {
		gp := hostmath.GELUTanhPrime(float64(rawGate.Data[i]))
		dRawGate[i] = float32(float64(dGate[i]) * float64(perLayer.Data[i]) * gp)
		dPerLayer[i] += float32(float64(dGate[i]) * hostmath.GELUTanh(float64(rawGate.Data[i])))
	}
	dRawGateV := reference.Value{Shape: rawGate.Shape, Data: dRawGate}
	dScaled := make([]float32, len(scaled.Data))
	g3nMatMulBackward(dScaled, wg.dInputGate, scaled, *layer.PerLayerInputGate, dRawGateV)
	// scaled[t,f] = resultActive[t,f]*correctScale[f]; feeds dResult[active].
	dResultActive := make([]float32, len(resultActive.Data))
	for i := range dScaled {
		f := i % scaleWidth
		dResultActive[i] += dScaled[i] * layer.AltUpCorrectScale.Data[f]
		wg.dCorrectScale[f] += dScaled[i] * resultActive.Data[i]
	}
	// Add the direct dResult[active] path (result[active] is also an output).
	for i := range dResultActive {
		dResultActive[i] += dResult[active].Data[i]
	}

	// --- Correction-mix backward. dResult'[index] = dResult[index] for index!=active,
	//     and dResultActive for index==active. ---
	dCoeff := reference.Value{Shape: coefficients.Shape, Data: make([]float32, len(coefficients.Data))}
	dResultEff := func(index int) []float32 {
		if index == active {
			return dResultActive
		}
		return dResult[index].Data
	}
	for index := 0; index < count; index++ {
		dr := dResultEff(index)
		for i := range dPredictions[index].Data {
			dPredictions[index].Data[i] += dr[i] // clone term
		}
		for t := 0; t < tokens; t++ {
			coeff := coefficients.Data[t*count+index] + 1
			var dc float64
			for f := 0; f < width; f++ {
				pos := t*width + f
				innovation := activated.Data[pos] - predictions[active].Data[pos]
				dc += float64(dr[pos]) * float64(innovation)
				dInnov := dr[pos] * coeff
				dActivated[pos] += dInnov
				dPredictions[active].Data[pos] -= dInnov
			}
			dCoeff.Data[t*count+index] += float32(dc)
		}
	}
	// correct-coefficient matmul backward -> dModalities, dCorrectCoeff.
	dModalities := make([]float32, len(modalities.Data))
	g3nMatMulBackward(dModalities, wg.dCorrectCoeff, modalities, *layer.AltUpCorrectCoefficient, dCoeff)
	dModV := reference.Value{Shape: modalities.Shape, Data: dModalities}
	// modalities-of-activated backward -> dActivated (accumulate), dRouter, dRouterNorm.
	g3nModalitiesBackward(dActivated, wg.dRouter, wg.dRouterNorm, activated, layer, spec, dModV)
}

// g3nActivateFFNBackward: VJP of activateAlternateFFN. dGate and dUp are set.
// GELU differentiated as smooth tanh-GELU. Sparse layers route dV through the
// activation-sparsity VJP; dense layers pass it through unchanged.
func g3nActivateFFNBackward(dGate, dUp []float32, gate, up reference.Value, dResult []float32, sparse bool, stdMult float32) {
	width := int(gate.Shape.Dims[0])
	rows := int(gate.Shape.Dims[1])
	// Recompute the (possibly sparse-gated) pre-activation v.
	v := make([]float32, len(gate.Data))
	if sparse {
		hostmath.SparseGateInto(v, gate.Data, rows, width, float64(stdMult))
	} else {
		copy(v, gate.Data)
	}
	dV := make([]float32, len(v))
	for i := range v {
		g := hostmath.GELUTanh(float64(v[i]))
		gp := hostmath.GELUTanhPrime(float64(v[i]))
		dUp[i] = float32(float64(dResult[i]) * g)
		dV[i] = float32(float64(dResult[i]) * float64(up.Data[i]) * gp)
	}
	if sparse {
		hostmath.ActivationSparsityBackward(dGate, gate.Data, dV, rows, width, float64(stdMult))
	} else {
		copy(dGate, dV)
	}
}

// --- Laurel (learned augmented residual layer) host forward + backward. The
// Forward mirrors the compiled activation projection's Laurel subgraph:
//   l1  = LaurelLeft  @ normalized      (emb -> rank)
//   l2  = LaurelRight @ l1              (rank -> emb)
//   ln  = weightedRMS(l2, LaurelPostNorm)
//   out = ln + normalized

func gemma3nLaurelForward(normalized, left, right, postNorm reference.Value, eps float32) reference.Value {
	l1, _ := alternateLinear(left, normalized)
	l2, _ := alternateLinear(right, l1)
	ln, _ := alternateRMSNorm(l2, postNorm, eps)
	out := ln.Clone()
	for i := range out.Data {
		out.Data[i] += normalized.Data[i]
	}
	return out
}

// gemma3nLaurelBackward: VJP of gemma3nLaurelForward. dNormalized and the three
// weight grads accumulate.
func gemma3nLaurelBackward(dNormalized, dLeft, dRight, dPostNorm []float32, normalized, left, right, postNorm reference.Value, dOut reference.Value, eps float32) {
	// Recompute l1, l2.
	l1, _ := alternateLinear(left, normalized)
	l2, _ := alternateLinear(right, l1)
	// out = ln + normalized: identity residual to dNormalized, dLn = dOut.
	for i := range dNormalized {
		dNormalized[i] += dOut.Data[i]
	}
	dL2 := make([]float32, len(l2.Data))
	g3nWeightedRMSBackward(dL2, dPostNorm, l2, postNorm, dOut, eps)
	dL2V := reference.Value{Shape: l2.Shape, Data: dL2}
	dL1 := make([]float32, len(l1.Data))
	g3nMatMulBackward(dL1, dRight, l1, right, dL2V)
	dL1V := reference.Value{Shape: l1.Shape, Data: dL1}
	g3nMatMulBackward(dNormalized, dLeft, normalized, left, dL1V)
}

// --- Assembled gemma3n active-layer host forward + backward. This is a faithful
// Host transcription of the compiled projection prefix and suffix
// (the differentiable composition), used to FD-verify the full-layer gradient wrt
// the residual-stream input and every layer weight, for both the full-causal and
// the sliding-window layer types. The rotary width is config-driven via
// cfg.ropeDim through the shared hostmath.RopeWidth owner (0 => full head_dim,
// as real gemma3n's rope.dimension_count defaults to key_length==head_dim), so
// partial-rope configs rotate only the first RopeWidth dims — matching the
// serving reference.ropeNeoX convention.

type gemma3nLayerWeights struct {
	AttnNorm, LaurelLeft, LaurelRight, LaurelPostNorm reference.Value
	AttnQ, AttnQNorm, AttnK, AttnKNorm, AttnV         reference.Value
	AttnOutput, AttnPostNorm                          reference.Value
	FFNorm, FFGate, FFUp, FFDown, FFPostNorm          reference.Value
}

type gemma3nLayerGrads struct {
	AttnNorm, LaurelLeft, LaurelRight, LaurelPostNorm []float32
	AttnQ, AttnQNorm, AttnK, AttnKNorm, AttnV         []float32
	AttnOutput, AttnPostNorm                          []float32
	FFNorm, FFGate, FFUp, FFDown, FFPostNorm          []float32
}

func newGemma3nLayerGrads(w gemma3nLayerWeights) *gemma3nLayerGrads {
	z := func(v reference.Value) []float32 { return make([]float32, len(v.Data)) }
	return &gemma3nLayerGrads{
		AttnNorm: z(w.AttnNorm), LaurelLeft: z(w.LaurelLeft), LaurelRight: z(w.LaurelRight), LaurelPostNorm: z(w.LaurelPostNorm),
		AttnQ: z(w.AttnQ), AttnQNorm: z(w.AttnQNorm), AttnK: z(w.AttnK), AttnKNorm: z(w.AttnKNorm), AttnV: z(w.AttnV),
		AttnOutput: z(w.AttnOutput), AttnPostNorm: z(w.AttnPostNorm),
		FFNorm: z(w.FFNorm), FFGate: z(w.FFGate), FFUp: z(w.FFUp), FFDown: z(w.FFDown), FFPostNorm: z(w.FFPostNorm),
	}
}

type gemma3nLayerConfig struct {
	emb, tokens, headCount, kvHeads, headDim, inter, window int
	ropeDim                                                 int // rotary width; 0 => full head_dim
	sparse                                                  bool
	eps                                                     float32
	ropeTheta                                               float64
	stdMult                                                 float32
}

func g3nRV(data []float32, dim0, dim1 int) reference.Value {
	return reference.Value{Shape: tensor.MustShape(uint64(dim0), uint64(dim1)), Data: data}
}

// gemma3nActiveLayerForward mirrors the graph attention+FFN stage on the host.
func gemma3nActiveLayerForward(w gemma3nLayerWeights, cfg gemma3nLayerConfig, input []float32) []float32 {
	out, _ := gemma3nActiveLayerTrace(w, cfg, input)
	return out
}

// gemma3nActiveLayerTrace returns the layer output plus every intermediate the
// backward needs, so the VJP composes against the exact forward.
type g3nLayerTrace struct {
	normalized, laurel                          []float32
	qRaw, qNormed, q, kRaw, kNormed, k, vRaw, v []float32
	attn, attnProj, attnPost                    []float32
	residual, ffnInput, gate, up, activated     []float32
	down, downPost                              []float32
}

func gemma3nActiveLayerTrace(w gemma3nLayerWeights, cfg gemma3nLayerConfig, input []float32) ([]float32, *g3nLayerTrace) {
	emb, tk := cfg.emb, cfg.tokens
	hc, kv, hd := cfg.headCount, cfg.kvHeads, cfg.headDim
	qWidth, kvWidth := hc*hd, kv*hd
	eps := float64(cfg.eps)
	tr := &g3nLayerTrace{}

	tr.normalized = make([]float32, emb*tk)
	hostmath.RMSNormInto(tr.normalized, input, w.AttnNorm.Data, tk, emb, eps)

	laurel := gemma3nLaurelForward(g3nRV(tr.normalized, emb, tk), w.LaurelLeft, w.LaurelRight, w.LaurelPostNorm, cfg.eps)
	tr.laurel = laurel.Data

	tr.qRaw = make([]float32, qWidth*tk)
	hostmath.Linear(tr.qRaw, tr.normalized, w.AttnQ.Data, tk, emb, qWidth)
	tr.qNormed = make([]float32, qWidth*tk)
	hostmath.RMSNormInto(tr.qNormed, tr.qRaw, w.AttnQNorm.Data, tk*hc, hd, eps)

	tr.kRaw = make([]float32, kvWidth*tk)
	hostmath.Linear(tr.kRaw, tr.normalized, w.AttnK.Data, tk, emb, kvWidth)
	tr.kNormed = make([]float32, kvWidth*tk)
	hostmath.RMSNormInto(tr.kNormed, tr.kRaw, w.AttnKNorm.Data, tk*kv, hd, eps)

	tr.vRaw = make([]float32, kvWidth*tk)
	hostmath.Linear(tr.vRaw, tr.normalized, w.AttnV.Data, tk, emb, kvWidth)
	tr.v = make([]float32, kvWidth*tk)
	hostmath.RMSNormInto(tr.v, tr.vRaw, nil, tk*kv, hd, eps) // unit-scale value norm

	rd := hostmath.RopeWidth(cfg.ropeDim, hd)
	invFreq := hostmath.RopeInvFreq(cfg.ropeTheta, rd)
	tr.q = append([]float32(nil), tr.qNormed...)
	tr.k = append([]float32(nil), tr.kNormed...)
	for p := 0; p < tk; p++ {
		for h := 0; h < hc; h++ {
			base := (p*hc + h) * hd
			hostmath.ApplyRotaryHalf(tr.q[base:base+rd], invFreq, p)
		}
		for h := 0; h < kv; h++ {
			base := (p*kv + h) * hd
			hostmath.ApplyRotaryHalf(tr.k[base:base+rd], invFreq, p)
		}
	}
	scale := float32(1 / math.Sqrt(float64(hd)))
	for i := range tr.q {
		tr.q[i] *= scale
	}
	tr.attn = make([]float32, qWidth*tk)
	hostmath.WindowedCausalAttention(tr.attn, tr.q, tr.k, tr.v, tk, hc, kv, hd, cfg.window)

	tr.attnProj = make([]float32, emb*tk)
	hostmath.Linear(tr.attnProj, tr.attn, w.AttnOutput.Data, tk, qWidth, emb)
	tr.attnPost = make([]float32, emb*tk)
	hostmath.RMSNormInto(tr.attnPost, tr.attnProj, w.AttnPostNorm.Data, tk, emb, eps)

	invSqrt2 := float32(1 / math.Sqrt2)
	tr.residual = make([]float32, emb*tk)
	for i := range tr.residual {
		tr.residual[i] = (input[i] + tr.attnPost[i] + tr.laurel[i]) * invSqrt2
	}
	tr.ffnInput = make([]float32, emb*tk)
	hostmath.RMSNormInto(tr.ffnInput, tr.residual, w.FFNorm.Data, tk, emb, eps)
	tr.gate = make([]float32, cfg.inter*tk)
	tr.up = make([]float32, cfg.inter*tk)
	hostmath.Linear(tr.gate, tr.ffnInput, w.FFGate.Data, tk, emb, cfg.inter)
	hostmath.Linear(tr.up, tr.ffnInput, w.FFUp.Data, tk, emb, cfg.inter)
	// activateFFN with the smooth training GELU.
	tr.activated = g3nActivateFFNSmooth(tr.gate, tr.up, tk, cfg.inter, cfg.sparse, cfg.stdMult)
	tr.down = make([]float32, emb*tk)
	hostmath.Linear(tr.down, tr.activated, w.FFDown.Data, tk, cfg.inter, emb)
	tr.downPost = make([]float32, emb*tk)
	hostmath.RMSNormInto(tr.downPost, tr.down, w.FFPostNorm.Data, tk, emb, eps)
	out := make([]float32, emb*tk)
	for i := range out {
		out[i] = tr.residual[i] + tr.downPost[i]
	}
	return out, tr
}

// g3nActivateFFNSmooth: the FFN activation with the smooth tanh-GELU (training
// leg), matching g3nActivateFFNBackward's differentiated forward.
func g3nActivateFFNSmooth(gate, up []float32, rows, width int, sparse bool, stdMult float32) []float32 {
	v := make([]float32, len(gate))
	if sparse {
		hostmath.SparseGateInto(v, gate, rows, width, float64(stdMult))
	} else {
		copy(v, gate)
	}
	out := make([]float32, len(gate))
	for i := range v {
		out[i] = float32(hostmath.GELUTanh(float64(v[i]))) * up[i]
	}
	return out
}

// gemma3nActiveLayerBackward: VJP of gemma3nActiveLayerForward. Returns dInput
// and every weight grad. dOutput is the gradient at the layer output.
func gemma3nActiveLayerBackward(w gemma3nLayerWeights, cfg gemma3nLayerConfig, input, dOutput []float32) ([]float32, *gemma3nLayerGrads) {
	emb, tk := cfg.emb, cfg.tokens
	hc, kv, hd := cfg.headCount, cfg.kvHeads, cfg.headDim
	qWidth, kvWidth := hc*hd, kv*hd
	eps := float64(cfg.eps)
	_, tr := gemma3nActiveLayerTrace(w, cfg, input)
	g := newGemma3nLayerGrads(w)
	dInput := make([]float32, emb*tk)

	// output = residual + downPost.
	dResidual := append([]float32(nil), dOutput...)
	dDownPost := dOutput
	dDown := make([]float32, emb*tk)
	hostmath.RMSNormBackward(dDown, g.FFPostNorm, tr.down, w.FFPostNorm.Data, dDownPost, tk, emb, eps, false)
	dActivated := make([]float32, cfg.inter*tk)
	hostmath.LinearBackward(dActivated, g.FFDown, nil, tr.activated, w.FFDown.Data, dDown, tk, cfg.inter, emb, false)
	dGate := make([]float32, cfg.inter*tk)
	dUp := make([]float32, cfg.inter*tk)
	g3nActivateFFNBackward(dGate, dUp, g3nRV(tr.gate, cfg.inter, tk), g3nRV(tr.up, cfg.inter, tk), dActivated, cfg.sparse, cfg.stdMult)
	dFFNInput := make([]float32, emb*tk)
	hostmath.LinearBackward(dFFNInput, g.FFGate, nil, tr.ffnInput, w.FFGate.Data, dGate, tk, emb, cfg.inter, false)
	hostmath.LinearBackward(dFFNInput, g.FFUp, nil, tr.ffnInput, w.FFUp.Data, dUp, tk, emb, cfg.inter, true)
	hostmath.RMSNormBackward(dResidual, g.FFNorm, tr.residual, w.FFNorm.Data, dFFNInput, tk, emb, eps, true)

	// residual = (input + attnPost + laurel)/sqrt2.
	invSqrt2 := float32(1 / math.Sqrt2)
	dInner := make([]float32, emb*tk)
	for i := range dInner {
		dInner[i] = dResidual[i] * invSqrt2
	}
	for i := range dInput {
		dInput[i] += dInner[i]
	}
	dAttnPost := dInner
	dLaurel := g3nRV(append([]float32(nil), dInner...), emb, tk)

	// laurel backward -> dNormalized (accumulate), laurel weight grads.
	dNormalized := make([]float32, emb*tk)
	gemma3nLaurelBackward(dNormalized, g.LaurelLeft, g.LaurelRight, g.LaurelPostNorm,
		g3nRV(tr.normalized, emb, tk), w.LaurelLeft, w.LaurelRight, w.LaurelPostNorm, dLaurel, cfg.eps)

	// attnPost weighted-RMS backward.
	dAttnProj := make([]float32, emb*tk)
	hostmath.RMSNormBackward(dAttnProj, g.AttnPostNorm, tr.attnProj, w.AttnPostNorm.Data, dAttnPost, tk, emb, eps, false)
	dAttn := make([]float32, qWidth*tk)
	hostmath.LinearBackward(dAttn, g.AttnOutput, nil, tr.attn, w.AttnOutput.Data, dAttnProj, tk, qWidth, emb, false)

	// windowed attention backward.
	dq := make([]float32, qWidth*tk)
	dk := make([]float32, kvWidth*tk)
	dv := make([]float32, kvWidth*tk)
	hostmath.WindowedCausalAttentionBackward(dq, dk, dv, tr.q, tr.k, tr.v, dAttn, tk, hc, kv, hd, cfg.window)

	// q scale, then rope backward on dq/dk.
	scale := float32(1 / math.Sqrt(float64(hd)))
	for i := range dq {
		dq[i] *= scale
	}
	rd := hostmath.RopeWidth(cfg.ropeDim, hd)
	invFreq := hostmath.RopeInvFreq(cfg.ropeTheta, rd)
	for p := 0; p < tk; p++ {
		for h := 0; h < hc; h++ {
			base := (p*hc + h) * hd
			hostmath.RotaryHalfBackward(dq[base:base+rd], invFreq, p)
		}
		for h := 0; h < kv; h++ {
			base := (p*kv + h) * hd
			hostmath.RotaryHalfBackward(dk[base:base+rd], invFreq, p)
		}
	}
	// q/k per-head norm backward, value unit-norm backward.
	dQRaw := make([]float32, qWidth*tk)
	hostmath.RMSNormBackward(dQRaw, g.AttnQNorm, tr.qRaw, w.AttnQNorm.Data, dq, tk*hc, hd, eps, false)
	dKRaw := make([]float32, kvWidth*tk)
	hostmath.RMSNormBackward(dKRaw, g.AttnKNorm, tr.kRaw, w.AttnKNorm.Data, dk, tk*kv, hd, eps, false)
	dVRaw := make([]float32, kvWidth*tk)
	hostmath.RMSNormBackward(dVRaw, nil, tr.vRaw, nil, dv, tk*kv, hd, eps, false)

	// Q/K/V projections backward -> dNormalized (accumulate).
	hostmath.LinearBackward(dNormalized, g.AttnQ, nil, tr.normalized, w.AttnQ.Data, dQRaw, tk, emb, qWidth, true)
	hostmath.LinearBackward(dNormalized, g.AttnK, nil, tr.normalized, w.AttnK.Data, dKRaw, tk, emb, kvWidth, true)
	hostmath.LinearBackward(dNormalized, g.AttnV, nil, tr.normalized, w.AttnV.Data, dVRaw, tk, emb, kvWidth, true)

	// input norm backward -> dInput (accumulate).
	hostmath.RMSNormBackward(dInput, g.AttnNorm, input, w.AttnNorm.Data, dNormalized, tk, emb, eps, true)
	return dInput, g
}

var _ = math.Sqrt2
