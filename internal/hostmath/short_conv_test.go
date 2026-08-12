package hostmath

import (
	"math"
	"math/rand"
	"testing"
)

// TestShortConvBackwardFD FD-grad-checks ShortConvBackward against central
// differences of L = sum(dY ⊙ ShortConvForward(x, w, bias)).
func TestShortConvBackwardFD(t *testing.T) {
	const channels, T, k = 3, 6, 4
	rng := rand.New(rand.NewSource(11))
	rs := func(n int) []float32 {
		s := make([]float32, n)
		for i := range s {
			s[i] = float32(rng.NormFloat64() * 0.4)
		}
		return s
	}
	x := rs(channels * T)
	w := rs(channels * k)
	bias := rs(channels)
	dY := rs(channels * T)

	dX, dW, dBias := ShortConvBackward(x, dY, channels, T, w, bias, k)

	loss := func() float64 {
		out := ShortConvForward(x, channels, T, w, bias, k)
		var l float64
		for i := range out {
			l += float64(dY[i]) * float64(out[i])
		}
		return l
	}
	const eps = 1e-3
	check := func(name string, arr, grad []float32) {
		var maxd float64
		for i := range arr {
			o := arr[i]
			arr[i] = o + float32(eps)
			lp := loss()
			arr[i] = o - float32(eps)
			lm := loss()
			arr[i] = o
			if d := math.Abs((lp-lm)/(2*eps) - float64(grad[i])); d > maxd {
				maxd = d
			}
		}
		t.Logf("%s max|analytic-fd| %.3e", name, maxd)
		if maxd > 1e-3 {
			t.Fatalf("%s grad %.3e > 1e-3", name, maxd)
		}
	}
	check("dX", x, dX)
	check("dW", w, dW)
	check("dBias", bias, dBias)
}
