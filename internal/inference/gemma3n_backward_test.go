// Finite-difference gates for the gemma3n host-backward (VJP) blocks. FD is the
// oracle (no adaptive E4B grad goldens exist). Ops with no GELU are FD'd against
// the real gemma3n.go forward; the correct+inject / FFN-activation paths are
// FD'd against a smooth-GELU mirror, since the serving forward's fp16-rounded
// GELU is not part of the training-leg derivative.
package inference

import (
	"math"
	"math/rand"
	"testing"

	"overgo/internal/hostmath"
	"overgo/internal/model"
	"overgo/internal/tensor"
	"overgo/internal/tensor/reference"
)

const (
	g3nFDStep = 1e-2
	g3nEmb    = 6
	g3nTokens = 3
	g3nCount  = 4
	g3nMods   = 4
	g3nPLI    = 5
)

func g3nFDTol(accum int) float64 { return 24*g3nFDStep*g3nFDStep + 3e-3*math.Sqrt(float64(accum)) }

func g3nRand(rng *rand.Rand, n int, base, s float64) reference.Value {
	return g3nRandShape(rng, tensor.MustShape(uint64(n)), base, s) // 1-D
}

func g3nRandShape(rng *rand.Rand, shape tensor.Shape, base, s float64) reference.Value {
	v := reference.Value{Shape: shape, Data: make([]float32, shapeElems(shape))}
	for i := range v.Data {
		v.Data[i] = float32(base + rng.NormFloat64()*s)
	}
	return v
}

func g3nMat(rng *rand.Rand, inner, out int, s float64) reference.Value {
	return g3nRandShape(rng, tensor.MustShape(uint64(inner), uint64(out)), 0, s)
}

func g3nTestSpec() model.Spec {
	return model.Spec{
		CommonSpec: model.CommonSpec{
			EmbeddingLength: g3nEmb,
			RMSNormEpsilon:  1e-6,
		},
		MultimodalSpec: model.MultimodalSpec{
			AltUpCount:            g3nCount,
			AltUpActive:           0,
			SparsityStdMultiplier: 0.5,
		},
	}
}

func g3nTestLayer(rng *rand.Rand) model.HostLayer {
	routerNorm := g3nRand(rng, g3nEmb, 1.0, 0.1)
	router := g3nMat(rng, g3nEmb, g3nMods, 0.4)
	predictCoeff := g3nMat(rng, g3nMods, g3nCount*g3nCount, 0.3)
	correctCoeff := g3nMat(rng, g3nMods, g3nCount, 0.3)
	correctScale := g3nRand(rng, g3nEmb, 1.0, 0.2)
	inputGate := g3nMat(rng, g3nEmb, g3nPLI, 0.4)
	projection := g3nMat(rng, g3nPLI, g3nEmb, 0.4)
	postNorm := g3nRand(rng, g3nEmb, 1.0, 0.1)
	return model.HostLayer{
		AltUpRouterNorm:         &routerNorm,
		AltUpRouter:             &router,
		AltUpPredictCoefficient: &predictCoeff,
		AltUpCorrectCoefficient: &correctCoeff,
		AltUpCorrectScale:       &correctScale,
		PerLayerInputGate:       &inputGate,
		PerLayerProjection:      &projection,
		PerLayerPostNorm:        &postNorm,
	}
}

// g3nFDCheck FD-checks analytic grads `grad` for parameter buffer `vec` against
// central differences of `loss`.
func g3nFDCheck(t *testing.T, name string, vec, grad []float32, accum int, loss func() float64) {
	t.Helper()
	tol := g3nFDTol(accum)
	for i := range vec {
		orig := vec[i]
		vec[i] = orig + g3nFDStep
		lp := loss()
		vec[i] = orig - g3nFDStep
		lm := loss()
		vec[i] = orig
		fd := (lp - lm) / (2 * g3nFDStep)
		if math.Abs(fd-float64(grad[i])) > tol*(1+math.Abs(fd)) {
			t.Fatalf("%s[%d]: fd %.6g vs analytic %.6g (tol %.1e)", name, i, fd, grad[i], tol)
		}
	}
}

// ---- modalities ----
func TestGemma3nModalitiesBackwardFD(t *testing.T) {
	rng := rand.New(rand.NewSource(1))
	spec := g3nTestSpec()
	layer := g3nTestLayer(rng)
	input := g3nRandShape(rng, tensor.MustShape(g3nEmb, g3nTokens), 0, 0.7)
	seed := g3nRandShape(rng, tensor.MustShape(g3nMods, g3nTokens), 0, 1)

	loss := func() float64 {
		out, _ := gemma3nModalities(input, layer, spec.EmbeddingLength, spec.RMSNormEpsilon)
		var s float64
		for i := range out.Data {
			s += float64(out.Data[i]) * float64(seed.Data[i])
		}
		return s
	}
	dInput := make([]float32, len(input.Data))
	dRouter := make([]float32, len(layer.AltUpRouter.Data))
	dRouterNorm := make([]float32, len(layer.AltUpRouterNorm.Data))
	g3nModalitiesBackward(dInput, dRouter, dRouterNorm, input, layer, spec, seed)

	g3nFDCheck(t, "dInput", input.Data, dInput, g3nEmb, loss)
	g3nFDCheck(t, "dRouter", layer.AltUpRouter.Data, dRouter, g3nEmb, loss)
	g3nFDCheck(t, "dRouterNorm", layer.AltUpRouterNorm.Data, dRouterNorm, g3nEmb, loss)
}

// ---- initialize AltUp ----
func TestGemma3nInitializeAltUpBackwardFD(t *testing.T) {
	rng := rand.New(rand.NewSource(2))
	input := g3nRandShape(rng, tensor.MustShape(g3nEmb, g3nTokens), 0.3, 0.6)
	projection := g3nRandShape(rng, tensor.MustShape(g3nEmb, g3nEmb, g3nCount-1), 0, 0.4)
	seeds := make([]reference.Value, g3nCount)
	for i := range seeds {
		seeds[i] = g3nRandShape(rng, tensor.MustShape(g3nEmb, g3nTokens), 0, 1)
	}
	loss := func() float64 {
		states, _ := gemma3nInitializeAltUp(input, projection, g3nCount)
		var s float64
		for k := range states {
			for i := range states[k].Data {
				s += float64(states[k].Data[i]) * float64(seeds[k].Data[i])
			}
		}
		return s
	}
	dInput := make([]float32, len(input.Data))
	dProjection := make([]float32, len(projection.Data))
	g3nInitializeAltUpBackward(dInput, dProjection, input, projection, seeds)
	g3nFDCheck(t, "dInput", input.Data, dInput, g3nEmb, loss)
	g3nFDCheck(t, "dProjection", projection.Data, dProjection, g3nEmb, loss)
}

// ---- merge AltUp ----
func TestGemma3nMergeAltUpBackwardFD(t *testing.T) {
	rng := rand.New(rand.NewSource(3))
	states := make([]reference.Value, g3nCount)
	for i := range states {
		states[i] = g3nRandShape(rng, tensor.MustShape(g3nEmb, g3nTokens), 0.3, 0.6)
	}
	unembedding := g3nRandShape(rng, tensor.MustShape(g3nEmb, g3nEmb, g3nCount-1), 0, 0.4)
	seed := g3nRandShape(rng, tensor.MustShape(g3nEmb, g3nTokens), 0, 1)
	const active = 0
	loss := func() float64 {
		out, _ := gemma3nMergeAltUp(states, unembedding, active)
		var s float64
		for i := range out.Data {
			s += float64(out.Data[i]) * float64(seed.Data[i])
		}
		return s
	}
	dStates := make([]reference.Value, g3nCount)
	for i := range dStates {
		dStates[i] = g3nZeros(states[i].Shape)
	}
	dUnembedding := make([]float32, len(unembedding.Data))
	g3nMergeAltUpBackward(dStates, dUnembedding, states, unembedding, active, seed)
	for k := range states {
		g3nFDCheck(t, "dStates", states[k].Data, dStates[k].Data, g3nEmb, loss)
	}
	g3nFDCheck(t, "dUnembedding", unembedding.Data, dUnembedding, g3nEmb, loss)
}

// ---- predict ----
func TestGemma3nPredictBackwardFD(t *testing.T) {
	rng := rand.New(rand.NewSource(4))
	spec := g3nTestSpec()
	layer := g3nTestLayer(rng)
	states := make([]reference.Value, g3nCount)
	for i := range states {
		states[i] = g3nRandShape(rng, tensor.MustShape(g3nEmb, g3nTokens), 0.2, 0.6)
	}
	seeds := make([]reference.Value, g3nCount)
	for i := range seeds {
		seeds[i] = g3nRandShape(rng, tensor.MustShape(g3nEmb, g3nTokens), 0, 1)
	}
	loss := func() float64 {
		res, _ := gemma3nPredict(
			states, layer, spec.AltUpActive, spec.EmbeddingLength, spec.RMSNormEpsilon,
		)
		var s float64
		for k := range res {
			for i := range res[k].Data {
				s += float64(res[k].Data[i]) * float64(seeds[k].Data[i])
			}
		}
		return s
	}
	dStates := make([]reference.Value, g3nCount)
	for i := range dStates {
		dStates[i] = g3nZeros(states[i].Shape)
	}
	dPC := make([]float32, len(layer.AltUpPredictCoefficient.Data))
	dR := make([]float32, len(layer.AltUpRouter.Data))
	dRN := make([]float32, len(layer.AltUpRouterNorm.Data))
	g3nPredictBackward(dStates, dPC, dR, dRN, states, layer, spec, seeds)
	for k := range states {
		g3nFDCheck(t, "dStates", states[k].Data, dStates[k].Data, g3nEmb*g3nCount, loss)
	}
	g3nFDCheck(t, "dPredictCoeff", layer.AltUpPredictCoefficient.Data, dPC, g3nEmb*g3nCount, loss)
	g3nFDCheck(t, "dRouter", layer.AltUpRouter.Data, dR, g3nEmb*g3nCount, loss)
	g3nFDCheck(t, "dRouterNorm", layer.AltUpRouterNorm.Data, dRN, g3nEmb*g3nCount, loss)
}

// smoothCorrectAndInject mirrors gemma3nCorrectAndInject with the smooth
// tanh-GELU (the training-leg activation) instead of the fp16-rounded serving
// GELU, so FD matches the analytic backward.
func smoothCorrectAndInject(predictions []reference.Value, activated, perLayer reference.Value, layer model.HostLayer, spec model.Spec) []reference.Value {
	count := len(predictions)
	active := int(spec.AltUpActive)
	tokens := int(activated.Shape.Dims[1])
	width := int(activated.Shape.Dims[0])
	modalities, _ := gemma3nModalities(
		activated, layer, spec.EmbeddingLength, spec.RMSNormEpsilon,
	)
	coefficients, _ := gemma3nMatMul(*layer.AltUpCorrectCoefficient, modalities)
	result := make([]reference.Value, count)
	for index := range predictions {
		result[index] = predictions[index].Clone()
		for t := 0; t < tokens; t++ {
			coeff := coefficients.Data[t*count+index] + 1
			for f := 0; f < width; f++ {
				pos := t*width + f
				innovation := activated.Data[pos] - predictions[active].Data[pos]
				result[index].Data[pos] += innovation * coeff
			}
		}
	}
	scaled := result[active].Clone()
	sw := int(scaled.Shape.Dims[0])
	for i := range scaled.Data {
		scaled.Data[i] *= layer.AltUpCorrectScale.Data[i%sw]
	}
	rawGate, _ := gemma3nMatMul(*layer.PerLayerInputGate, scaled)
	gate := rawGate.Clone()
	for i := range gate.Data {
		gate.Data[i] = float32(hostmath.GELUTanh(float64(rawGate.Data[i]))) * perLayer.Data[i]
	}
	injection, _ := gemma3nMatMul(*layer.PerLayerProjection, gate)
	injection, _ = gemma3nWeightedRMS(injection, *layer.PerLayerPostNorm, spec.RMSNormEpsilon)
	for index := 1; index < count; index++ {
		for i := range result[index].Data {
			result[index].Data[i] += injection.Data[i]
		}
	}
	return result
}

// ---- correct + inject (AltUp correction mix + PLE) ----
func TestGemma3nCorrectAndInjectBackwardFD(t *testing.T) {
	rng := rand.New(rand.NewSource(5))
	spec := g3nTestSpec()
	layer := g3nTestLayer(rng)
	predictions := make([]reference.Value, g3nCount)
	for i := range predictions {
		predictions[i] = g3nRandShape(rng, tensor.MustShape(g3nEmb, g3nTokens), 0.2, 0.5)
	}
	activated := g3nRandShape(rng, tensor.MustShape(g3nEmb, g3nTokens), 0.2, 0.5)
	perLayer := g3nRandShape(rng, tensor.MustShape(g3nPLI, g3nTokens), 0.3, 0.4)
	seeds := make([]reference.Value, g3nCount)
	for i := range seeds {
		seeds[i] = g3nRandShape(rng, tensor.MustShape(g3nEmb, g3nTokens), 0, 1)
	}
	loss := func() float64 {
		res := smoothCorrectAndInject(predictions, activated, perLayer, layer, spec)
		var s float64
		for k := range res {
			for i := range res[k].Data {
				s += float64(res[k].Data[i]) * float64(seeds[k].Data[i])
			}
		}
		return s
	}
	dPredictions := make([]reference.Value, g3nCount)
	for i := range dPredictions {
		dPredictions[i] = g3nZeros(predictions[i].Shape)
	}
	dActivated := make([]float32, len(activated.Data))
	dPerLayer := make([]float32, len(perLayer.Data))
	wg := &g3nCorrectGrads{
		dCorrectCoeff: make([]float32, len(layer.AltUpCorrectCoefficient.Data)),
		dRouter:       make([]float32, len(layer.AltUpRouter.Data)),
		dRouterNorm:   make([]float32, len(layer.AltUpRouterNorm.Data)),
		dCorrectScale: make([]float32, len(layer.AltUpCorrectScale.Data)),
		dInputGate:    make([]float32, len(layer.PerLayerInputGate.Data)),
		dProjection:   make([]float32, len(layer.PerLayerProjection.Data)),
		dPostNorm:     make([]float32, len(layer.PerLayerPostNorm.Data)),
	}
	g3nCorrectAndInjectBackward(dPredictions, dActivated, dPerLayer, wg, predictions, activated, perLayer, layer, spec, seeds)

	accum := g3nEmb * g3nCount
	for k := range predictions {
		g3nFDCheck(t, "dPredictions", predictions[k].Data, dPredictions[k].Data, accum, loss)
	}
	g3nFDCheck(t, "dActivated", activated.Data, dActivated, accum, loss)
	g3nFDCheck(t, "dPerLayer", perLayer.Data, dPerLayer, accum, loss)
	g3nFDCheck(t, "dCorrectCoeff", layer.AltUpCorrectCoefficient.Data, wg.dCorrectCoeff, accum, loss)
	g3nFDCheck(t, "dRouter", layer.AltUpRouter.Data, wg.dRouter, accum, loss)
	g3nFDCheck(t, "dRouterNorm", layer.AltUpRouterNorm.Data, wg.dRouterNorm, accum, loss)
	g3nFDCheck(t, "dCorrectScale", layer.AltUpCorrectScale.Data, wg.dCorrectScale, accum, loss)
	g3nFDCheck(t, "dInputGate", layer.PerLayerInputGate.Data, wg.dInputGate, accum, loss)
	g3nFDCheck(t, "dProjection", layer.PerLayerProjection.Data, wg.dProjection, accum, loss)
	g3nFDCheck(t, "dPostNorm", layer.PerLayerPostNorm.Data, wg.dPostNorm, accum, loss)
}

// ---- FFN activation (sparse + dense) ----
func testActivateFFNBackward(t *testing.T, sparse bool) {
	t.Helper()
	rng := rand.New(rand.NewSource(6))
	const width, tokens = 8, 3
	gate := g3nRandShape(rng, tensor.MustShape(width, tokens), 0, 1.0)
	up := g3nRandShape(rng, tensor.MustShape(width, tokens), 0.3, 0.6)
	seed := g3nRandShape(rng, tensor.MustShape(width, tokens), 0, 1)
	stdMult := float32(0.5)

	smoothForward := func() reference.Value {
		res := reference.Value{Shape: gate.Shape, Data: make([]float32, len(gate.Data))}
		for tk := 0; tk < tokens; tk++ {
			base := tk * width
			cutoff := float32(-math.MaxFloat32)
			if sparse {
				var sum float64
				for i := 0; i < width; i++ {
					sum += float64(gate.Data[base+i])
				}
				mean := sum / float64(width)
				var sq float64
				for i := 0; i < width; i++ {
					d := float64(gate.Data[base+i]) - mean
					sq += d * d
				}
				cutoff = float32(mean + float64(stdMult)*math.Sqrt(sq/float64(width-1)))
			}
			for i := 0; i < width; i++ {
				v := gate.Data[base+i]
				if sparse {
					v = max(v-cutoff, 0)
				}
				res.Data[base+i] = float32(hostmath.GELUTanh(float64(v))) * up.Data[base+i]
			}
		}
		return res
	}
	loss := func() float64 {
		res := smoothForward()
		var s float64
		for i := range res.Data {
			s += float64(res.Data[i]) * float64(seed.Data[i])
		}
		return s
	}
	dGate := make([]float32, len(gate.Data))
	dUp := make([]float32, len(up.Data))
	g3nActivateFFNBackward(dGate, dUp, gate, up, seed.Data, sparse, stdMult)
	g3nFDCheck(t, "dGate", gate.Data, dGate, width, loss)
	g3nFDCheck(t, "dUp", up.Data, dUp, width, loss)
}

func TestGemma3nActivateFFNBackwardDenseFD(t *testing.T)  { testActivateFFNBackward(t, false) }
func TestGemma3nActivateFFNBackwardSparseFD(t *testing.T) { testActivateFFNBackward(t, true) }

// ---- Laurel ----
func TestGemma3nLaurelBackwardFD(t *testing.T) {
	rng := rand.New(rand.NewSource(7))
	const rank = 2
	eps := float32(1e-6)
	normalized := g3nRandShape(rng, tensor.MustShape(g3nEmb, g3nTokens), 0, 0.7)
	left := g3nMat(rng, g3nEmb, rank, 0.4)
	right := g3nMat(rng, rank, g3nEmb, 0.4)
	postNorm := g3nRand(rng, g3nEmb, 1.0, 0.1)
	seed := g3nRandShape(rng, tensor.MustShape(g3nEmb, g3nTokens), 0, 1)
	loss := func() float64 {
		out := gemma3nLaurelForward(normalized, left, right, postNorm, eps)
		var s float64
		for i := range out.Data {
			s += float64(out.Data[i]) * float64(seed.Data[i])
		}
		return s
	}
	dNorm := make([]float32, len(normalized.Data))
	dLeft := make([]float32, len(left.Data))
	dRight := make([]float32, len(right.Data))
	dPostNorm := make([]float32, len(postNorm.Data))
	gemma3nLaurelBackward(dNorm, dLeft, dRight, dPostNorm, normalized, left, right, postNorm, seed, eps)
	g3nFDCheck(t, "dNorm", normalized.Data, dNorm, g3nEmb, loss)
	g3nFDCheck(t, "dLeft", left.Data, dLeft, g3nEmb, loss)
	g3nFDCheck(t, "dRight", right.Data, dRight, g3nEmb, loss)
	g3nFDCheck(t, "dPostNorm", postNorm.Data, dPostNorm, g3nEmb, loss)
}

// ---- assembled active layer (full-causal + sliding-window) ----
func g3nBuildLayerWeights(rng *rand.Rand, cfg gemma3nLayerConfig, rank int) gemma3nLayerWeights {
	m := func(inner, out int, s float64) reference.Value { return g3nMat(rng, inner, out, s) }
	v := func(n int, base, s float64) reference.Value { return g3nRand(rng, n, base, s) }
	qW, kvW := cfg.headCount*cfg.headDim, cfg.kvHeads*cfg.headDim
	return gemma3nLayerWeights{
		AttnNorm:       v(cfg.emb, 1.0, 0.1),
		LaurelLeft:     m(cfg.emb, rank, 0.4),
		LaurelRight:    m(rank, cfg.emb, 0.4),
		LaurelPostNorm: v(cfg.emb, 1.0, 0.1),
		AttnQ:          m(cfg.emb, qW, 0.4),
		AttnQNorm:      v(cfg.headDim, 1.0, 0.1),
		AttnK:          m(cfg.emb, kvW, 0.4),
		AttnKNorm:      v(cfg.headDim, 1.0, 0.1),
		AttnV:          m(cfg.emb, kvW, 0.4),
		AttnOutput:     m(qW, cfg.emb, 0.4),
		AttnPostNorm:   v(cfg.emb, 1.0, 0.1),
		FFNorm:         v(cfg.emb, 1.0, 0.1),
		FFGate:         m(cfg.emb, cfg.inter, 0.4),
		FFUp:           m(cfg.emb, cfg.inter, 0.4),
		FFDown:         m(cfg.inter, cfg.emb, 0.4),
		FFPostNorm:     v(cfg.emb, 1.0, 0.1),
	}
}

func testAssembledLayerBackward(t *testing.T, window int, sparse bool) {
	t.Helper()
	rng := rand.New(rand.NewSource(int64(101 + window)))
	cfg := gemma3nLayerConfig{
		emb: 6, tokens: 4, headCount: 2, kvHeads: 1, headDim: 4, inter: 8,
		window: window, sparse: sparse, eps: 1e-6, ropeTheta: 10000, stdMult: 0.5,
	}
	w := g3nBuildLayerWeights(rng, cfg, 2)
	input := make([]float32, cfg.emb*cfg.tokens)
	for i := range input {
		input[i] = float32(rng.NormFloat64() * 0.6)
	}
	seed := make([]float32, cfg.emb*cfg.tokens)
	for i := range seed {
		seed[i] = float32(rng.NormFloat64())
	}
	loss := func() float64 {
		out := gemma3nActiveLayerForward(w, cfg, input)
		var s float64
		for i := range out {
			s += float64(out[i]) * float64(seed[i])
		}
		return s
	}
	dInput, g := gemma3nActiveLayerBackward(w, cfg, input, seed)

	accum := cfg.emb * cfg.tokens
	g3nFDCheck(t, "dInput", input, dInput, accum, loss)
	for _, p := range []struct {
		name string
		data []float32
		grad []float32
	}{
		{"AttnNorm", w.AttnNorm.Data, g.AttnNorm},
		{"LaurelLeft", w.LaurelLeft.Data, g.LaurelLeft},
		{"LaurelRight", w.LaurelRight.Data, g.LaurelRight},
		{"LaurelPostNorm", w.LaurelPostNorm.Data, g.LaurelPostNorm},
		{"AttnQ", w.AttnQ.Data, g.AttnQ},
		{"AttnQNorm", w.AttnQNorm.Data, g.AttnQNorm},
		{"AttnK", w.AttnK.Data, g.AttnK},
		{"AttnKNorm", w.AttnKNorm.Data, g.AttnKNorm},
		{"AttnV", w.AttnV.Data, g.AttnV},
		{"AttnOutput", w.AttnOutput.Data, g.AttnOutput},
		{"AttnPostNorm", w.AttnPostNorm.Data, g.AttnPostNorm},
		{"FFNorm", w.FFNorm.Data, g.FFNorm},
		{"FFGate", w.FFGate.Data, g.FFGate},
		{"FFUp", w.FFUp.Data, g.FFUp},
		{"FFDown", w.FFDown.Data, g.FFDown},
		{"FFPostNorm", w.FFPostNorm.Data, g.FFPostNorm},
	} {
		g3nFDCheck(t, p.name, p.data, p.grad, accum, loss)
	}
}

func TestGemma3nAssembledLayerBackwardFullCausalFD(t *testing.T) {
	testAssembledLayerBackward(t, -1, false)
}
func TestGemma3nAssembledLayerBackwardSlidingFD(t *testing.T) {
	testAssembledLayerBackward(t, 2, false)
}

// The sparse-FFN activation VJP is FD-verified in isolation
// (TestGemma3nActivateFFNBackwardSparseFD); it is not re-checked through the
// assembled layer because the sparsity cutoff is a relu discontinuity whose kink
// central differences straddle when perturbing the residual-stream input.
