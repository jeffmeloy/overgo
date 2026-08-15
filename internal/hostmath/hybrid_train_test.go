package hostmath

import (
	"math"
	"math/rand"
	"testing"
)

// TestHybridDecoderLayerTrainStep drives the composed layer through real SGD:
// forward -> squared-error loss -> HybridDecoderLayerBackward -> weight update,
// and asserts the loss falls. This is the end-to-end proof the hand-derived
// grads are usable to train qwen3.5's recurrent (gated-delta) layer, not just
// numerically correct. Uses the linear_attention variant (the GDN layer).
func TestHybridDecoderLayerTrainStep(t *testing.T) {
	const (
		T, hidden, inter = 4, 8, 16
		hd, eps          = 4, 1e-6
		hk, hv           = 2, 2
	)
	keyDim, valDim, K := hk*hd, hv*hd, 3
	rng := rand.New(rand.NewSource(7))
	rs := func(n int) []float32 {
		s := make([]float32, n)
		for i := range s {
			s[i] = float32(rng.NormFloat64() * 0.3)
		}
		return s
	}
	w := HybridLayerWeights{
		InputNorm: rs(hidden), PostNorm: rs(hidden), IsLinear: true,
		MLP: QwenMLPWeights{Gate: rs(inter * hidden), Up: rs(inter * hidden), Down: rs(hidden * inter)},
		GDN: GatedDeltaMixWeights{
			Wq: rs(keyDim * hidden), Wk: rs(keyDim * hidden), Wv: rs(valDim * hidden),
			ConvQ: rs(keyDim * K), ConvK: rs(keyDim * K), ConvV: rs(valDim * K),
			ConvBiasQ: rs(keyDim), ConvBiasK: rs(keyDim), ConvBiasV: rs(valDim),
			Wbeta: rs(hv * hidden), Walpha: rs(hv * hidden), TimeStep: rs(hv), A: rs(hv),
			Wz: rs(valDim * hidden), Norm: rs(hd), Wout: rs(hidden * valDim),
		},
	}
	d := HybridLayerDims{Tokens: T, Hidden: hidden, Inter: inter, Eps: eps,
		GDN: GatedDeltaMixDims{Tokens: T, Hidden: hidden, KeyHeads: hk, ValueHeads: hv, HeadDim: hd, ConvK: K, OutDim: hidden, Eps: eps}}
	x := rs(T * hidden)
	state := rs(hv * hd * hd)
	target := rs(T * hidden)

	// pair a weight slice with its grad slice for the SGD update.
	pairs := func(g HybridDecoderLayerGrads) [][2][]float32 {
		return [][2][]float32{
			{w.InputNorm, g.DInputNorm}, {w.PostNorm, g.DPostNorm},
			{w.MLP.Gate, g.DMLP.Gate}, {w.MLP.Up, g.DMLP.Up}, {w.MLP.Down, g.DMLP.Down},
			{w.GDN.Wq, g.DGDN.DWq}, {w.GDN.Wk, g.DGDN.DWk}, {w.GDN.Wv, g.DGDN.DWv},
			{w.GDN.ConvQ, g.DGDN.DConvQ}, {w.GDN.ConvK, g.DGDN.DConvK}, {w.GDN.ConvV, g.DGDN.DConvV},
			{w.GDN.ConvBiasQ, g.DGDN.DConvBiasQ}, {w.GDN.ConvBiasK, g.DGDN.DConvBiasK}, {w.GDN.ConvBiasV, g.DGDN.DConvBiasV},
			{w.GDN.Wbeta, g.DGDN.DWbeta}, {w.GDN.Walpha, g.DGDN.DWalpha},
			{w.GDN.TimeStep, g.DGDN.DTimeStep}, {w.GDN.A, g.DGDN.DA},
			{w.GDN.Wz, g.DGDN.DWz}, {w.GDN.Norm, g.DGDN.DNorm}, {w.GDN.Wout, g.DGDN.DWout},
		}
	}

	step := func() (loss float64) {
		out, c := HybridDecoderLayerForward(x, w, d, state)
		dOut := make([]float32, len(out))
		for i := range out {
			e := float64(out[i]) - float64(target[i])
			loss += 0.5 * e * e
			dOut[i] = float32(e)
		}
		g := HybridDecoderLayerBackward(x, w, d, state, dOut, c)
		const lr = 0.05
		for _, p := range pairs(g) {
			for i := range p[0] {
				p[0][i] -= float32(lr * float64(p[1][i]))
			}
		}
		return loss
	}

	first := step()
	var last float64
	for i := 0; i < 40; i++ {
		last = step()
	}
	t.Logf("loss %.5f -> %.5f over 41 SGD steps", first, last)
	if math.IsNaN(last) || last >= first {
		t.Fatalf("loss did not decrease: %.5f -> %.5f", first, last)
	}
	if last > 0.2*first {
		t.Fatalf("loss decreased too little: %.5f -> %.5f (want < 0.2x)", first, last)
	}
}
