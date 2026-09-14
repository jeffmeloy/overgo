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

	x := e4bRandVec(rng, rows*d, 0, 0.8)
	weight := e4bRandVec(rng, d, 1, 0.1)
	bias := e4bRandVec(rng, d, 0, 0.1)
	dy := e4bRandVec(rng, rows*d, 0, 1)

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

	checkVectorBackwardFD(t, "dx", x, dx, func() float64 { return loss(weight, bias) }, tol)
	checkVectorBackwardFD(t, "dW", weight, dW, func() float64 { return loss(weight, bias) }, tol)
	checkVectorBackwardFD(t, "dB", bias, dB, func() float64 { return loss(weight, bias) }, tol)

	// No-affine variant: LN(x) with nil weight, plus the addDX contract.
	dxPlain := make([]float32, rows*d)
	seed := e4bRandVec(rng, rows*d, 0, 0.5)
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
	checkUnaryBackwardFD(t, "gelu-erf", x, dy, GELUErf, GELUErfBackward)
}

// checkVectorBackwardFD perturbs float32 inputs and restores each before comparison.
func checkVectorBackwardFD(t *testing.T, name string, values, gradients []float32, loss func() float64, tolerance float64) {
	t.Helper()
	for i := range values {
		original := values[i]
		values[i] = original + fdStep
		plus := loss()
		values[i] = original - fdStep
		minus := loss()
		values[i] = original
		fd := (plus - minus) / (2 * fdStep)
		if math.IsInf(fd, 0) || !(math.Abs(fd-float64(gradients[i])) <= tolerance*(1+math.Abs(fd))) {
			t.Fatalf("%s[%d]: fd %.6g vs analytic %.6g (tol %.1e)", name, i, fd, gradients[i], tolerance)
		}
	}
}

// checkUnaryBackwardFD evaluates float64 perturbations and verifies destination aliasing.
func checkUnaryBackwardFD(t *testing.T, name string, x, dy []float32, forward func(float64) float64, backward func([]float32, []float32, []float32)) {
	t.Helper()
	dx := make([]float32, len(x))
	backward(dx, x, dy)
	tolerance := fdTol(fdStep, 1)
	for i := range x {
		plus := forward(float64(x[i])+fdStep) * float64(dy[i])
		minus := forward(float64(x[i])-fdStep) * float64(dy[i])
		fd := (plus - minus) / (2 * fdStep)
		if math.IsInf(fd, 0) || !(math.Abs(fd-float64(dx[i])) <= tolerance*(1+math.Abs(fd))) {
			t.Fatalf("%s dx[%d]: fd %.6g vs analytic %.6g (tol %.1e)", name, i, fd, dx[i], tolerance)
		}
	}
	aliased := append([]float32(nil), dy...)
	backward(aliased, x, aliased)
	for i := range aliased {
		if aliased[i] != dx[i] {
			t.Fatalf("aliased %s backward diverges at %d", name, i)
		}
	}
}
