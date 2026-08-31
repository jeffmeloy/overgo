package hostmath

import (
	"math"
	"math/rand"
	"testing"
)

// TestHybridDecoderLayerBackwardFD FD-grad-checks HybridDecoderLayerBackward for
// both mix variants: L = sum(dOut ⊙ HybridDecoderLayerForward(x,w,state)),
// central differences over the input, the state, and every weight, versus the
// composed analytic VJP.
func TestHybridDecoderLayerBackwardFD(t *testing.T) {
	const (
		T, hidden, inter = 4, 8, 16
		hd, eps          = 4, 1e-6
	)
	rng := rand.New(rand.NewSource(20260812))
	rs := func(n int) []float32 {
		s := make([]float32, n)
		for i := range s {
			s[i] = float32(rng.NormFloat64() * 0.4)
		}
		return s
	}
	mlp := QwenMLPWeights{Gate: rs(inter * hidden), Up: rs(inter * hidden), Down: rs(hidden * inter)}

	run := func(t *testing.T, w HybridLayerWeights, d HybridLayerDims, state []float32, pairs func(g HybridDecoderLayerGrads) []fdParam) {
		x := rs(T * hidden)
		dOut := rs(T * hidden)
		loss := func() float64 {
			out, _ := HybridDecoderLayerForward(x, w, d, state)
			var l float64
			for i := range out {
				l += float64(dOut[i]) * float64(out[i])
			}
			return l
		}
		_, c := HybridDecoderLayerForward(x, w, d, state)
		g := HybridDecoderLayerBackward(x, w, d, state, dOut, c)
		ps := append([]fdParam{{"dX", x, g.DX}}, pairs(g)...)
		if state != nil {
			ps = append(ps, fdParam{"dState", state, g.DState})
		}
		fdCheck(t, loss, ps)
	}

	t.Run("linear_attention", func(t *testing.T) {
		hk, hv := 2, 2
		keyDim, valDim := hk*hd, hv*hd // 8, 8 == hidden
		gdn := GatedDeltaMixWeights{
			Wq: rs(keyDim * hidden), Wk: rs(keyDim * hidden), Wv: rs(valDim * hidden),
			ConvQ: rs(keyDim * 3), ConvK: rs(keyDim * 3), ConvV: rs(valDim * 3),
			ConvBiasQ: rs(keyDim), ConvBiasK: rs(keyDim), ConvBiasV: rs(valDim),
			Wbeta: rs(hv * hidden), Walpha: rs(hv * hidden), TimeStep: rs(hv), A: rs(hv),
			Wz: rs(valDim * hidden), Norm: rs(hd), Wout: rs(hidden * valDim),
		}
		w := HybridLayerWeights{InputNorm: rs(hidden), PostNorm: rs(hidden), IsLinear: true, GDN: gdn, MLP: mlp}
		d := HybridLayerDims{Tokens: T, Hidden: hidden, Inter: inter, Eps: eps,
			GDN: GatedDeltaMixDims{Tokens: T, Hidden: hidden, KeyHeads: hk, ValueHeads: hv, HeadDim: hd, ConvK: 3, OutDim: hidden, Eps: eps}}
		state := rs(hv * hd * hd)
		run(t, w, d, state, func(g HybridDecoderLayerGrads) []fdParam {
			return []fdParam{
				{"dInputNorm", w.InputNorm, g.DInputNorm}, {"dPostNorm", w.PostNorm, g.DPostNorm},
				{"dMLP.Gate", w.MLP.Gate, g.DMLP.Gate}, {"dMLP.Up", w.MLP.Up, g.DMLP.Up}, {"dMLP.Down", w.MLP.Down, g.DMLP.Down},
				{"dGDN.Wq", w.GDN.Wq, g.DGDN.DWq}, {"dGDN.Wk", w.GDN.Wk, g.DGDN.DWk}, {"dGDN.Wv", w.GDN.Wv, g.DGDN.DWv},
				{"dGDN.ConvV", w.GDN.ConvV, g.DGDN.DConvV}, {"dGDN.Wbeta", w.GDN.Wbeta, g.DGDN.DWbeta},
				{"dGDN.Walpha", w.GDN.Walpha, g.DGDN.DWalpha}, {"dGDN.A", w.GDN.A, g.DGDN.DA},
				{"dGDN.Wz", w.GDN.Wz, g.DGDN.DWz}, {"dGDN.Norm", w.GDN.Norm, g.DGDN.DNorm}, {"dGDN.Wout", w.GDN.Wout, g.DGDN.DWout},
			}
		})
	})

	t.Run("full_attention", func(t *testing.T) {
		heads, kv := 2, 1
		qDim, kvDim := heads*hd, kv*hd // 8, 4
		attn := AttentionMixWeights{
			Wq: rs(qDim * hidden), Wk: rs(kvDim * hidden), Wv: rs(kvDim * hidden), Wo: rs(hidden * qDim),
			QNorm: rs(hd), KNorm: rs(hd),
		}
		w := HybridLayerWeights{InputNorm: rs(hidden), PostNorm: rs(hidden), IsLinear: false, Attn: attn, MLP: mlp}
		d := HybridLayerDims{Tokens: T, Hidden: hidden, Inter: inter, Eps: eps,
			Attn: AttentionMixDims{Tokens: T, Hidden: hidden, Heads: heads, KVHeads: kv, HeadDim: hd, RopeTheta: 10000, Eps: eps}}
		run(t, w, d, nil, func(g HybridDecoderLayerGrads) []fdParam {
			return []fdParam{
				{"dInputNorm", w.InputNorm, g.DInputNorm}, {"dPostNorm", w.PostNorm, g.DPostNorm},
				{"dMLP.Gate", w.MLP.Gate, g.DMLP.Gate}, {"dMLP.Up", w.MLP.Up, g.DMLP.Up}, {"dMLP.Down", w.MLP.Down, g.DMLP.Down},
				{"dAttn.Wq", w.Attn.Wq, g.DAttn.Wq}, {"dAttn.Wk", w.Attn.Wk, g.DAttn.Wk}, {"dAttn.Wv", w.Attn.Wv, g.DAttn.Wv},
				{"dAttn.Wo", w.Attn.Wo, g.DAttn.Wo}, {"dAttn.QNorm", w.Attn.QNorm, g.DAttn.QNorm}, {"dAttn.KNorm", w.Attn.KNorm, g.DAttn.KNorm},
			}
		})
	})

	// Partial rope (qwen3.5 head_dim*partial_rotary_factor): head_dim 8, rotary
	// width 4 -> only the first 4 of each head's 8 dims rotate, the tail passes
	// through. FD-checks the analytic backward routes cotangents through the
	// partial-rope RotaryHalfBackward correctly.
	t.Run("full_attention_partial_rope", func(t *testing.T) {
		const phd, ropeDim = 8, 4 // head_dim, partial rotary width
		heads, kv := 1, 1
		qDim, kvDim := heads*phd, kv*phd // 8, 8 (== hidden)
		attn := AttentionMixWeights{
			Wq: rs(qDim * hidden), Wk: rs(kvDim * hidden), Wv: rs(kvDim * hidden), Wo: rs(hidden * qDim),
			QNorm: rs(phd), KNorm: rs(phd),
		}
		w := HybridLayerWeights{InputNorm: rs(hidden), PostNorm: rs(hidden), IsLinear: false, Attn: attn, MLP: mlp}
		d := HybridLayerDims{Tokens: T, Hidden: hidden, Inter: inter, Eps: eps,
			Attn: AttentionMixDims{Tokens: T, Hidden: hidden, Heads: heads, KVHeads: kv, HeadDim: phd, RopeDim: ropeDim, RopeTheta: 1000000, Eps: eps}}
		run(t, w, d, nil, func(g HybridDecoderLayerGrads) []fdParam {
			return []fdParam{
				{"dInputNorm", w.InputNorm, g.DInputNorm}, {"dPostNorm", w.PostNorm, g.DPostNorm},
				{"dAttn.Wq", w.Attn.Wq, g.DAttn.Wq}, {"dAttn.Wk", w.Attn.Wk, g.DAttn.Wk}, {"dAttn.Wv", w.Attn.Wv, g.DAttn.Wv},
				{"dAttn.Wo", w.Attn.Wo, g.DAttn.Wo}, {"dAttn.QNorm", w.Attn.QNorm, g.DAttn.QNorm}, {"dAttn.KNorm", w.Attn.KNorm, g.DAttn.KNorm},
			}
		})
	})
}

type fdParam struct {
	name string
	arr  []float32
	grad []float32
}

func fdCheck(t *testing.T, loss func() float64, params []fdParam) {
	const step = 1e-3
	for _, p := range params {
		var maxd, maxrel float64
		for i := range p.arr {
			o := p.arr[i]
			p.arr[i] = o + float32(step)
			lp := loss()
			p.arr[i] = o - float32(step)
			lm := loss()
			p.arr[i] = o
			fd := (lp - lm) / (2 * step)
			ad := math.Abs(fd - float64(p.grad[i]))
			maxd = max(maxd, ad)
			if den := math.Abs(fd) + math.Abs(float64(p.grad[i])) + 1e-6; ad/den > maxrel {
				maxrel = ad / den
			}
		}
		t.Logf("%-13s max|abs| %.3e  max|rel| %.3e", p.name, maxd, maxrel)
		if maxd > 5e-3 && maxrel > 5e-3 {
			t.Fatalf("%s grad off: abs %.3e rel %.3e", p.name, maxd, maxrel)
		}
	}
}
