package hostmath

import (
	"math"
	"math/rand"
	"testing"
)

// TestL2NormBackwardFD FD-grad-checks L2NormBackward against central differences
// of L = sum(dY ⊙ L2NormForward(x)) in the unclamped region (norm > eps).
func TestL2NormBackwardFD(t *testing.T) {
	const rows, width = 4, 5
	const eps = 1e-6
	rng := rand.New(rand.NewSource(13))
	x := make([]float32, rows*width)
	dY := make([]float32, rows*width)
	for i := range x {
		x[i] = float32(rng.NormFloat64()*0.5 + 0.3) // norm well above eps
		dY[i] = float32(rng.NormFloat64())
	}
	dX := L2NormBackward(x, dY, rows, width, eps)
	loss := func() float64 {
		out := L2NormForward(x, rows, width, eps)
		var l float64
		for i := range out {
			l += float64(dY[i]) * float64(out[i])
		}
		return l
	}
	const step = 1e-3
	var maxd float64
	for i := range x {
		o := x[i]
		x[i] = o + float32(step)
		lp := loss()
		x[i] = o - float32(step)
		lm := loss()
		x[i] = o
		if d := math.Abs((lp-lm)/(2*step) - float64(dX[i])); d > maxd {
			maxd = d
		}
	}
	t.Logf("dX max|analytic-fd| %.3e", maxd)
	if maxd > 1e-3 {
		t.Fatalf("dX grad %.3e > 1e-3", maxd)
	}
}
