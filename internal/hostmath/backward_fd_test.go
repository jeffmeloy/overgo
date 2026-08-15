// Finite-difference gates for the training-leg backwards added for the
// speech-flow port: central FD against the exact forward each claims to
// differentiate (in-Go FD, no torch fixture — the forwards are already
// golden-gated).
package hostmath

import (
	"math"
	"math/rand"
	"testing"
)

func TestMaskedBidirectionalAttentionBackwardFiniteDifference(t *testing.T) {
	const querySeq, keySeq, heads, kvHeads, headDim = 2, 3, 2, 1, 2
	q := []float32{0.2, -0.1, 0.4, 0.3, -0.2, 0.5, 0.1, -0.4}
	k := []float32{0.3, -0.2, -0.1, 0.4, 0.5, 0.2}
	v := []float32{0.7, -0.3, 0.2, 0.6, -0.4, 0.8}
	dOut := []float32{0.5, -0.2, 0.1, 0.4, -0.3, 0.6, 0.2, -0.5}
	mask := []bool{true, false, true}
	dq, dk, dv := make([]float32, len(q)), make([]float32, len(k)), make([]float32, len(v))
	MaskedBidirectionalAttentionBackward(dq, dk, dv, q, k, v, dOut, querySeq, keySeq, heads, kvHeads, headDim, mask)
	objective := func() float64 {
		out := make([]float32, len(q))
		MaskedBidirectionalAttention(out, q, k, v, querySeq, keySeq, heads, kvHeads, headDim, mask)
		var sum float64
		for index, value := range out {
			sum += float64(value) * float64(dOut[index])
		}
		return sum
	}
	const epsilon = float32(1e-3)
	for name, pair := range map[string]struct{ values, gradients []float32 }{
		"q": {q, dq}, "k": {k, dk}, "v": {v, dv},
	} {
		for index := range pair.values {
			original := pair.values[index]
			pair.values[index] = original + epsilon
			plus := objective()
			pair.values[index] = original - epsilon
			minus := objective()
			pair.values[index] = original
			finiteDifference := (plus - minus) / (2 * float64(epsilon))
			if delta := math.Abs(float64(pair.gradients[index]) - finiteDifference); delta > 2e-4 {
				t.Fatalf("%s[%d] analytic=%g finite_difference=%g delta=%g", name, index, pair.gradients[index], finiteDifference, delta)
			}
		}
	}
}

// fdTol: fp32 roundoff plus truncation by accumulation depth (the reference
// gradCheckTolF32Depth contract, adaptive operators_backward_test.go).
func fdTol(step float64, accumDepth int) float64 {
	const epsF32 = 1.1920929e-7
	const safety = 50.0 // test-only margin, matching the reference
	return safety * (epsF32*math.Sqrt(float64(accumDepth))/step + step*step/6.0)
}

const fdStep = 1e-2

func TestLayerNormBackwardFiniteDifference(t *testing.T) {
	const rows, d, eps = 2, 6, 1e-5
	rng := rand.New(rand.NewSource(11))
	vec := func(n int, base, s float64) []float32 {
		v := make([]float32, n)
		for i := range v {
			v[i] = float32(base + rng.NormFloat64()*s)
		}
		return v
	}
	x := vec(rows*d, 0, 0.8)
	weight := vec(d, 1, 0.1)
	bias := vec(d, 0, 0.1)
	dy := vec(rows*d, 0, 1)

	loss := func(w, b []float32) float64 {
		out := make([]float32, rows*d)
		LayerNormInto(out, x, w, b, rows, d, eps)
		var s float64
		for i := range out {
			s += float64(out[i]) * float64(dy[i])
		}
		return s
	}

	dx := make([]float32, rows*d)
	dW := make([]float32, d)
	dB := make([]float32, d)
	LayerNormBackward(dx, dW, dB, x, weight, dy, rows, d, eps, false)

	tol := fdTol(fdStep, rows*d)
	check := func(name string, vec, grad []float32, eval func() float64) {
		t.Helper()
		for i := range vec {
			orig := vec[i]
			vec[i] = orig + fdStep
			lp := eval()
			vec[i] = orig - fdStep
			lm := eval()
			vec[i] = orig
			fd := (lp - lm) / (2 * fdStep)
			if math.Abs(fd-float64(grad[i])) > tol*(1+math.Abs(fd)) {
				t.Fatalf("%s[%d]: fd %.6g vs analytic %.6g (tol %.1e)", name, i, fd, grad[i], tol)
			}
		}
	}
	check("dx", x, dx, func() float64 { return loss(weight, bias) })
	check("dW", weight, dW, func() float64 { return loss(weight, bias) })
	check("dB", bias, dB, func() float64 { return loss(weight, bias) })

	// No-affine variant: LN(x) with nil weight, plus the addDX contract.
	dxPlain := make([]float32, rows*d)
	seed := vec(rows*d, 0, 0.5)
	copy(dxPlain, seed)
	LayerNormBackward(dxPlain, nil, nil, x, nil, dy, rows, d, eps, true)
	lossPlain := func() float64 {
		out := make([]float32, rows*d)
		LayerNormInto(out, x, nil, nil, rows, d, eps)
		var s float64
		for i := range out {
			s += float64(out[i]) * float64(dy[i])
		}
		return s
	}
	for i := range x {
		orig := x[i]
		x[i] = orig + fdStep
		lp := lossPlain()
		x[i] = orig - fdStep
		lm := lossPlain()
		x[i] = orig
		fd := (lp-lm)/(2*fdStep) + float64(seed[i]) // addDX kept the seed
		if math.Abs(fd-float64(dxPlain[i])) > tol*(1+math.Abs(fd)) {
			t.Fatalf("no-affine dx[%d]: fd %.6g vs analytic %.6g", i, fd, dxPlain[i])
		}
	}
}

func TestGELUErfBackwardFiniteDifference(t *testing.T) {
	x := []float32{-3, -1.5, -0.5, 0, 0.25, 0.9, 2, 4}
	dy := []float32{1, -2, 0.5, 3, -1, 0.7, 1.3, -0.4}
	dx := make([]float32, len(x))
	GELUErfBackward(dx, x, dy)
	tol := fdTol(fdStep, 1)
	for i := range x {
		lp := GELUErf(float64(x[i])+fdStep) * float64(dy[i])
		lm := GELUErf(float64(x[i])-fdStep) * float64(dy[i])
		fd := (lp - lm) / (2 * fdStep)
		if math.Abs(fd-float64(dx[i])) > tol*(1+math.Abs(fd)) {
			t.Fatalf("gelu-erf dx[%d]: fd %.6g vs analytic %.6g", i, fd, dx[i])
		}
	}
	// Aliasing contract: dst may be dy.
	aliased := append([]float32(nil), dy...)
	GELUErfBackward(aliased, x, aliased)
	for i := range aliased {
		if aliased[i] != dx[i] {
			t.Fatalf("aliased gelu-erf backward diverges at %d", i)
		}
	}
}
