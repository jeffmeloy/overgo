package hostmath

import (
	"math"
	"math/rand"
	"testing"
)

// TestGatedDeltaMixBackwardFD FD-grad-checks GatedDeltaMixBackward end to end:
// L = sum(dOut ⊙ GatedDeltaMixForward(x,w,state)), central differences over the
// input, the state, and every weight, versus the composed analytic VJP.
func TestGatedDeltaMixBackwardFD(t *testing.T) {
	d := GatedDeltaMixDims{Tokens: 4, Hidden: 6, KeyHeads: 2, ValueHeads: 2, HeadDim: 3, ConvK: 3, OutDim: 5, Eps: 1e-6}
	T, H, hk, hv, hd := d.Tokens, d.Hidden, d.KeyHeads, d.ValueHeads, d.HeadDim
	keyDim, valDim := hk*hd, hv*hd
	rng := rand.New(rand.NewSource(20260812))
	rs := func(n int) []float32 {
		s := make([]float32, n)
		for i := range s {
			s[i] = float32(rng.NormFloat64() * 0.4)
		}
		return s
	}
	w := GatedDeltaMixWeights{
		Wq: rs(keyDim * H), Wk: rs(keyDim * H), Wv: rs(valDim * H),
		ConvQ: rs(keyDim * d.ConvK), ConvK: rs(keyDim * d.ConvK), ConvV: rs(valDim * d.ConvK),
		ConvBiasQ: rs(keyDim), ConvBiasK: rs(keyDim), ConvBiasV: rs(valDim),
		Wbeta: rs(hv * H), Walpha: rs(hv * H), TimeStep: rs(hv), A: rs(hv),
		Wz: rs(valDim * H), Norm: rs(hd), Wout: rs(d.OutDim * valDim),
	}
	x := rs(T * H)
	state := rs(hv * hd * hd)
	dOut := rs(T * d.OutDim)

	fwd := func() []float32 {
		out, _ := GatedDeltaMixForward(x, w, d, state)
		return out
	}
	loss := func() float64 {
		out := fwd()
		var l float64
		for i := range out {
			l += float64(dOut[i]) * float64(out[i])
		}
		return l
	}

	_, c := GatedDeltaMixForward(x, w, d, state)
	g := GatedDeltaMixBackward(x, w, d, state, dOut, c)

	const step = 1e-3
	check := func(name string, arr, grad []float32) {
		var maxd, maxrel float64
		for i := range arr {
			o := arr[i]
			arr[i] = o + float32(step)
			lp := loss()
			arr[i] = o - float32(step)
			lm := loss()
			arr[i] = o
			fd := (lp - lm) / (2 * step)
			ad := math.Abs(fd - float64(grad[i]))
			if ad > maxd {
				maxd = ad
			}
			if den := math.Abs(fd) + math.Abs(float64(grad[i])) + 1e-6; ad/den > maxrel {
				maxrel = ad / den
			}
		}
		t.Logf("%-11s max|abs| %.3e  max|rel| %.3e", name, maxd, maxrel)
		if maxd > 5e-3 && maxrel > 5e-3 {
			t.Fatalf("%s grad off: abs %.3e rel %.3e", name, maxd, maxrel)
		}
	}

	check("dX", x, g.DX)
	check("dState", state, g.DState)
	check("dWq", w.Wq, g.DWq)
	check("dWk", w.Wk, g.DWk)
	check("dWv", w.Wv, g.DWv)
	check("dConvQ", w.ConvQ, g.DConvQ)
	check("dConvK", w.ConvK, g.DConvK)
	check("dConvV", w.ConvV, g.DConvV)
	check("dConvBiasQ", w.ConvBiasQ, g.DConvBiasQ)
	check("dConvBiasK", w.ConvBiasK, g.DConvBiasK)
	check("dConvBiasV", w.ConvBiasV, g.DConvBiasV)
	check("dWbeta", w.Wbeta, g.DWbeta)
	check("dWalpha", w.Walpha, g.DWalpha)
	check("dTimeStep", w.TimeStep, g.DTimeStep)
	check("dA", w.A, g.DA)
	check("dWz", w.Wz, g.DWz)
	check("dNorm", w.Norm, g.DNorm)
	check("dWout", w.Wout, g.DWout)
}
