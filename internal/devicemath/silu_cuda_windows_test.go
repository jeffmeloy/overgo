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

// TestSiLUGateForwardMatchesHost verifies device SiLUGateForward returns
// a = silu(gate) and hMLP = a*up matching the fp64 reference.
func TestSiLUGateForwardMatchesHost(t *testing.T) {
	cudatest.Require(t)
	worker, err := device.New(0)
	if err != nil {
		t.Fatal(err)
	}
	defer worker.Close()

	const n = 256
	rng := rand.New(rand.NewSource(51))
	gate := randSlice(rng, n)
	up := randSlice(rng, n)

	a, hMLP, err := SiLUGateForward(worker, gate, up)
	if err != nil {
		t.Fatal(err)
	}
	var maxA, maxH float64
	for i := range n {
		wantA := hostSiLU(float64(gate[i]))
		wantH := wantA * float64(up[i])
		if d := math.Abs(float64(a[i]) - wantA); d > maxA {
			maxA = d
		}
		if d := math.Abs(float64(hMLP[i]) - wantH); d > maxH {
			maxH = d
		}
	}
	t.Logf("silu-gate forward: max |a-host| %.3e |hMLP-host| %.3e", maxA, maxH)
	const tolerance = 1e-5
	if maxA > tolerance || maxH > tolerance {
		t.Fatalf("silu-gate forward a %.3e hMLP %.3e > %.1e", maxA, maxH, tolerance)
	}
}

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
