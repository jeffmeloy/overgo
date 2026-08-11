//go:build windows

package devicemath

import (
	"math"
	"math/rand"
	"testing"

	"overgo/internal/cuda/device"
	cudatest "overgo/internal/cuda/testutil"
)

func randSlice(rng *rand.Rand, n int) []float32 {
	s := make([]float32, n)
	for i := range s {
		s[i] = float32(rng.NormFloat64())
	}
	return s
}

// hostLinearLoss returns L = sum(dY ⊙ (X·W)) in fp64 -- the scalar whose
// gradients w.r.t. X and W are exactly dY·Wᵀ and Xᵀ·dY.
func hostLinearLoss(x, w, dY []float32, rows, in, out int) float64 {
	var loss float64
	for r := 0; r < rows; r++ {
		for o := 0; o < out; o++ {
			var y float64
			for i := 0; i < in; i++ {
				y += float64(x[r*in+i]) * float64(w[i*out+o])
			}
			loss += float64(dY[r*out+o]) * y
		}
	}
	return loss
}

// TestLinearBackwardGradCheck verifies device LinearBackward against a central
// finite-difference of the host loss. The loss is linear in X and W, so the FD
// is exact up to fp rounding; the residual is fp32 GEMM precision.
func TestLinearBackwardGradCheck(t *testing.T) {
	cudatest.Require(t)
	worker, err := device.New(0)
	if err != nil {
		t.Fatal(err)
	}
	defer worker.Close()

	const rows, in, out = 6, 5, 4
	rng := rand.New(rand.NewSource(4))
	x := randSlice(rng, rows*in)
	w := randSlice(rng, in*out)
	dY := randSlice(rng, rows*out)

	dX, dW, err := LinearBackward(worker, x, w, dY, rows, in, out)
	if err != nil {
		t.Fatal(err)
	}

	const eps = 1e-2
	fd := func(perturb []float32, idx int) float64 {
		orig := perturb[idx]
		perturb[idx] = orig + float32(eps)
		lp := hostLinearLoss(x, w, dY, rows, in, out)
		perturb[idx] = orig - float32(eps)
		lm := hostLinearLoss(x, w, dY, rows, in, out)
		perturb[idx] = orig
		return (lp - lm) / (2 * eps)
	}
	var maxX, maxW float64
	for i := range x {
		if d := math.Abs(fd(x, i) - float64(dX[i])); d > maxX {
			maxX = d
		}
	}
	for j := range w {
		if d := math.Abs(fd(w, j) - float64(dW[j])); d > maxW {
			maxW = d
		}
	}
	t.Logf("grad-check: max |dX-fd| %.3e  max |dW-fd| %.3e", maxX, maxW)
	// Observed ~2e-6 / ~5e-6 (fp32 GEMM; the linear loss makes FD exact).
	const tolerance = 1e-3
	if maxX > tolerance || maxW > tolerance {
		t.Fatalf("dX %.3e / dW %.3e > %.1e", maxX, maxW, tolerance)
	}
}
