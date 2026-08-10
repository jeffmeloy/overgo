package hostmath

import (
	"math"
	"math/rand"
	"testing"
)

// The loss is sum(B_ds * dOut), whose gradient w.r.t. B_ds is exactly dOut. It
// calls the SHIPPED forward rather than reimplementing it, so there is only one
// spelling to keep in agreement.
func sinkhornLoss(logits, dOut []float32, count, n, iters int) float64 {
	m := make([]float32, len(logits))
	copy(m, logits)
	SinkhornFromLogitsInPlace(m, count, n, iters)
	s := 0.0
	for i := range m {
		s += float64(m[i]) * float64(dOut[i])
	}
	return s
}

// Bracketed finite differences: assert the deviation MINIMUM is interior to the
// sweep, so both truncation and cancellation are visible and the number is the
// gradient's error rather than the oracle's floor.
func TestSinkhornBackwardMatchesFiniteDifference(t *testing.T) {
	rng := rand.New(rand.NewSource(20260804))
	for _, n := range []int{2, 3} {
		const count, iters = 3, 20
		stride := n * n
		logits := make([]float32, count*stride)
		for i := range logits {
			logits[i] = float32((rng.Float64()*2 - 1) * 2)
		}
		dOut := make([]float32, count*stride)
		for i := range dOut {
			dOut[i] = float32(rng.Float64()*2 - 1)
		}

		grad := SinkhornFromLogitsBackward(logits, dOut, count, n, iters)

		idxs := []int{0, stride/2 + 1, count*stride - 1}
		steps := []float64{1e-1, 3e-2, 1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 1e-5}
		devs := make([]float64, len(steps))
		for si, h := range steps {
			worst := 0.0
			for _, idx := range idxs {
				orig := logits[idx]
				logits[idx] = float32(float64(orig) + h)
				up := sinkhornLoss(logits, dOut, count, n, iters)
				logits[idx] = float32(float64(orig) - h)
				dn := sinkhornLoss(logits, dOut, count, n, iters)
				logits[idx] = orig

				num := (up - dn) / (2 * h)
				den := math.Max(1e-3, math.Abs(float64(grad[idx])))
				if rel := math.Abs(num-float64(grad[idx])) / den; rel > worst {
					worst = rel
				}
			}
			devs[si] = worst
			t.Logf("n=%d h=%-8.0e max relative deviation %.3e", n, h, worst)
		}

		best, bestAt := devs[0], 0
		for i, v := range devs {
			if v < best {
				best, bestAt = v, i
			}
		}
		if bestAt == 0 || bestAt == len(devs)-1 {
			t.Fatalf("n=%d: deviation minimum at an ENDPOINT (h=%.0e, dev=%.3e): the sweep never "+
				"bracketed the optimum", n, steps[bestAt], best)
		}
		// The bar is 1e-4, the float32 forward's own cancellation floor:
		// SinkhornFromLogitsInPlace carries M in float32 between sweeps, so the
		// loss cannot be evaluated more precisely than float32 allows.
		if best > 1e-4 {
			t.Fatalf("n=%d: best relative deviation %.3e at h=%.0e exceeds 1e-4", n, best, steps[bestAt])
		}
		t.Logf("n=%d optimum bracketed at h=%.0e with relative deviation %.3e", n, steps[bestAt], best)
	}
}

// Sinkhorn is invariant to a global positive rescale of M, so shifting every
// logit by a constant must not change the output -- and therefore the gradient
// summed over any one matrix must be ~0. This is the property the
// max-subtraction term exists to preserve, checkable without an oracle.
func TestSinkhornBackwardRespectsScaleInvariance(t *testing.T) {
	rng := rand.New(rand.NewSource(11))
	const n, count, iters = 3, 4, 20
	stride := n * n
	logits := make([]float32, count*stride)
	for i := range logits {
		logits[i] = float32((rng.Float64()*2 - 1) * 3)
	}
	dOut := make([]float32, count*stride)
	for i := range dOut {
		dOut[i] = float32(rng.Float64()*2 - 1)
	}
	grad := SinkhornFromLogitsBackward(logits, dOut, count, n, iters)
	for b := 0; b < count; b++ {
		s := 0.0
		for i := 0; i < stride; i++ {
			s += float64(grad[b*stride+i])
		}
		if math.Abs(s) > 1e-4 {
			t.Fatalf("matrix %d: gradient sums to %.3g, but a uniform logit shift cannot change "+
				"a scale-invariant projection", b, s)
		}
	}
}

// The exact 2x2 closed form is doubly stochastic by construction: every 2x2
// projection is [[p,1-p],[1-p,p]], so rows and columns sum to 1 and the two
// off-diagonals match. Deterministic, no iteration to compare against. This
// keeps the unwired fast path (ADV-126) exercised and its invariant pinned.
func TestSinkhorn2x2ClosedFormIsDoublyStochastic(t *testing.T) {
	rng := rand.New(rand.NewSource(7))
	for trial := 0; trial < 200; trial++ {
		mat := make([]float32, 4)
		for i := range mat {
			mat[i] = float32((rng.Float64()*2 - 1) * 3)
		}
		sinkhorn2x2FromLogits(mat)
		if d := math.Abs(float64(mat[0]+mat[1]) - 1); d > 1e-6 {
			t.Fatalf("trial %d: row 0 sums to %.9f", trial, mat[0]+mat[1])
		}
		if d := math.Abs(float64(mat[0]+mat[2]) - 1); d > 1e-6 {
			t.Fatalf("trial %d: col 0 sums to %.9f", trial, mat[0]+mat[2])
		}
		if d := math.Abs(float64(mat[1] - mat[2])); d > 1e-6 {
			t.Fatalf("trial %d: off-diagonals differ (%.9f vs %.9f)", trial, mat[1], mat[2])
		}
		if mat[0] < 0 || mat[0] > 1 {
			t.Fatalf("trial %d: p=%.9f outside [0,1]", trial, mat[0])
		}
	}
}
