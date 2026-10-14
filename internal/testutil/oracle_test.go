package testutil

import (
	"math"
	"math/rand"
	"testing"

	"overgo/internal/hostmath"
)

func randomVector(rng *rand.Rand, n int, base, s float64) []float32 {
	v := make([]float32, n)
	for i := range v {
		v[i] = float32(base + rng.NormFloat64()*s)
	}
	return v
}

// fdTol bounds a central difference by fp32 roundoff plus truncation at an accumulation depth.
//   - source = the reference gradCheckTolF32Depth contract (adaptive operators_backward_test.go)
func fdTol(step float64, accumDepth int) float64 {
	const epsF32 = 1.1920929e-7
	const safety = 50.0 // test-only margin, matching the reference
	return safety * (epsF32*math.Sqrt(float64(accumDepth))/step + step*step/6.0)
}

const fdStep = 1e-2

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

// TestWindowedCausalAttentionBackwardFiniteDifference holds the windowed VJP to a central difference of hostmath.WindowedCausalAttention.
func TestWindowedCausalAttentionBackwardFiniteDifference(t *testing.T) {
	t.Parallel()
	const seq, heads, kvHeads, headDim, window = 5, 2, 1, 3, 3
	rng := rand.New(rand.NewSource(7))
	q := randomVector(rng, seq*heads*headDim, 0, 0.7)
	k := randomVector(rng, seq*kvHeads*headDim, 0, 0.7)
	v := randomVector(rng, seq*kvHeads*headDim, 0, 0.7)
	dOut := randomVector(rng, seq*heads*headDim, 0, 1)

	loss := func() float64 {
		out := make([]float32, seq*heads*headDim)
		hostmath.WindowedCausalAttention(out, q, k, v, seq, heads, kvHeads, headDim, window)
		var s float64
		for i := range out {
			s += float64(out[i]) * float64(dOut[i])
		}
		return s
	}

	dq := make([]float32, len(q))
	dk := make([]float32, len(k))
	dv := make([]float32, len(v))
	WindowedCausalAttentionBackward(dq, dk, dv, q, k, v, dOut, seq, heads, kvHeads, headDim, window)

	tol := fdTol(fdStep, seq*headDim)

	checkVectorBackwardFD(t, "dq", q, dq, loss, tol)
	checkVectorBackwardFD(t, "dk", k, dk, loss, tol)
	checkVectorBackwardFD(t, "dv", v, dv, loss, tol)
}

// TestWindowedCausalAttentionBackwardReducesToFullCausal holds the windowed VJP with window <= 0 to hostmath.CausalAttentionBackward.
func TestWindowedCausalAttentionBackwardReducesToFullCausal(t *testing.T) {
	t.Parallel()
	const seq, heads, kvHeads, headDim = 4, 4, 2, 3
	rng := rand.New(rand.NewSource(13))
	q := randomVector(rng, seq*heads*headDim, 0, 0.8)
	k := randomVector(rng, seq*kvHeads*headDim, 0, 0.8)
	v := randomVector(rng, seq*kvHeads*headDim, 0, 0.8)
	dOut := randomVector(rng, seq*heads*headDim, 0, 1)

	dqW := make([]float32, len(q))
	dkW := make([]float32, len(k))
	dvW := make([]float32, len(v))
	WindowedCausalAttentionBackward(dqW, dkW, dvW, q, k, v, dOut, seq, heads, kvHeads, headDim, 0)

	dqF := make([]float32, len(q))
	dkF := make([]float32, len(k))
	dvF := make([]float32, len(v))
	hostmath.CausalAttentionBackward(dqF, dkF, dvF, q, k, v, dOut, seq, heads, kvHeads, headDim)

	eq := func(name string, a, b []float32) {
		for i := range a {
			if math.Abs(float64(a[i]-b[i])) > 1e-6 {
				t.Fatalf("%s[%d]: windowed %.7g vs full-causal %.7g", name, i, a[i], b[i])
			}
		}
	}
	eq("dq", dqW, dqF)
	eq("dk", dkW, dkF)
	eq("dv", dvW, dvF)
}
