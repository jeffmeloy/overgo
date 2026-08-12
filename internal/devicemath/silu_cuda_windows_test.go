//go:build windows

package devicemath

import (
	"math"
	"math/rand"
	"testing"

	"overgo/internal/cuda/device"
	cudatest "overgo/internal/cuda/testutil"
)

func hostSiLU(x float64) float64 { return x / (1 + math.Exp(-x)) }

// TestSiLUBackwardGradCheck verifies device SiLUBackward against a central
// finite difference of the host loss L = sum(dy ⊙ silu(x)). SiLU is nonlinear,
// so the FD carries O(eps^2) truncation; the tolerance covers that plus fp32.
func TestSiLUBackwardGradCheck(t *testing.T) {
	cudatest.Require(t)
	worker, err := device.New(0)
	if err != nil {
		t.Fatal(err)
	}
	defer worker.Close()

	const n = 64
	rng := rand.New(rand.NewSource(5))
	x := randSlice(rng, n)
	dy := randSlice(rng, n)

	dx, err := SiLUBackward(worker, x, dy)
	if err != nil {
		t.Fatal(err)
	}

	loss := func() float64 {
		var l float64
		for i := range x {
			l += float64(dy[i]) * hostSiLU(float64(x[i]))
		}
		return l
	}
	const eps = 1e-3
	var maxDiff float64
	for i := range x {
		orig := x[i]
		x[i] = orig + float32(eps)
		lp := loss()
		x[i] = orig - float32(eps)
		lm := loss()
		x[i] = orig
		num := (lp - lm) / (2 * eps)
		if d := math.Abs(num - float64(dx[i])); d > maxDiff {
			maxDiff = d
		}
	}
	t.Logf("grad-check: max |dx-fd| %.3e", maxDiff)
	const tolerance = 1e-3
	if maxDiff > tolerance {
		t.Fatalf("dx vs fd %.3e > %.1e", maxDiff, tolerance)
	}
}
