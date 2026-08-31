//go:build windows

package devicemath

import (
	"math"
	"math/rand"
	"testing"

	"overgo/internal/cuda/device"
	cudatest "overgo/internal/cuda/testutil"
	"overgo/internal/hostmath"
)

// TestRMSNormForwardMatchesHost checks device RMSNormForward against
// hostmath.RMSNormInto (the forward the device layer forward must reproduce).
func TestRMSNormForwardMatchesHost(t *testing.T) {
	cudatest.Require(t)
	worker, err := device.New(0)
	if err != nil {
		t.Fatal(err)
	}
	defer worker.Close()

	const rows, d = 24, 64
	eps := 1e-6
	rng := rand.New(rand.NewSource(31))
	x := randSlice(rng, rows*d)
	weight := randSlice(rng, d)

	got, err := RMSNormForward(worker, x, weight, rows, d, eps)
	if err != nil {
		t.Fatal(err)
	}
	want := make([]float32, rows*d)
	hostmath.RMSNormInto(want, x, weight, rows, d, eps)

	var maxDiff float64
	for i := range want {
		if diff := math.Abs(float64(got[i]) - float64(want[i])); diff > maxDiff {
			maxDiff = diff
		}
	}
	t.Logf("rmsnorm forward: max |dev-host| %.3e", maxDiff)
	const tolerance = 1e-5
	if maxDiff > tolerance {
		t.Fatalf("rmsnorm forward %.3e > %.1e", maxDiff, tolerance)
	}
}

// rmsNormLossF64 returns L = sum(dy ⊙ y) for affine RMSNorm in float64.
func rmsNormLossF64(x, weight, dy []float32, rows, d int, eps float64) float64 {
	var loss float64
	for r := range rows {
		var ss float64
		for i := range d {
			v := float64(x[r*d+i])
			ss += v * v
		}
		inv := 1.0 / math.Sqrt(ss/float64(d)+eps)
		for i := range d {
			y := float64(x[r*d+i]) * inv * float64(weight[i])
			loss += float64(dy[r*d+i]) * y
		}
	}
	return loss
}

// TestRMSNormBackwardGradCheck verifies device RMSNormBackward (dx and the weight
// gradient dscale) against float64 central finite differences of L = sum(dy ⊙ y).
func TestRMSNormBackwardGradCheck(t *testing.T) {
	cudatest.Require(t)
	worker, err := device.New(0)
	if err != nil {
		t.Fatal(err)
	}
	defer worker.Close()

	const rows, d = 8, 16
	const eps = 1e-6
	rng := rand.New(rand.NewSource(7))
	x := randSlice(rng, rows*d)
	weight := randSlice(rng, d)
	dy := randSlice(rng, rows*d)

	dx, dscale, err := RMSNormBackward(worker, x, weight, dy, rows, d, eps)
	if err != nil {
		t.Fatal(err)
	}

	const feps = 1e-3
	gradCheck := func(name string, param, analytic []float32) float64 {
		var maxDiff float64
		for i := range param {
			orig := param[i]
			param[i] = orig + float32(feps)
			lp := rmsNormLossF64(x, weight, dy, rows, d, eps)
			param[i] = orig - float32(feps)
			lm := rmsNormLossF64(x, weight, dy, rows, d, eps)
			param[i] = orig
			if diff := math.Abs((lp-lm)/(2*feps) - float64(analytic[i])); diff > maxDiff {
				maxDiff = diff
			}
		}
		t.Logf("%s: max |grad-fd| %.3e", name, maxDiff)
		return maxDiff
	}
	const tolerance = 2e-3
	worst := gradCheck("dx", x, dx)
	worst = max(worst, gradCheck("dscale", weight, dscale))
	if worst > tolerance {
		t.Fatalf("worst grad-check %.3e > %.1e", worst, tolerance)
	}
}
