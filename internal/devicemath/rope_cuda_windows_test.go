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

// hostRopeHalf applies split-half RoPE to a [seq, nHeads*hd] tensor in float64.
func hostRopeHalf(x []float32, invFreq []float64, seq, nHeads, hd int) []float64 {
	out := make([]float64, len(x))
	half := hd / 2
	for p := range seq {
		for h := range nHeads {
			base := p*nHeads*hd + h*hd
			for i := range half {
				a := float64(p) * invFreq[i]
				c, s := math.Cos(a), math.Sin(a)
				x1, x2 := float64(x[base+i]), float64(x[base+i+half])
				out[base+i] = x1*c - x2*s
				out[base+i+half] = x2*c + x1*s
			}
		}
	}
	return out
}

// TestRoPEHalfForwardMatchesHost verifies device RoPEHalfForward reproduces the
// float64 split-half rope reference (hostmath.ApplyRotaryHalf's math) within fp32
// tolerance.
func TestRoPEHalfForwardMatchesHost(t *testing.T) {
	cudatest.Require(t)
	worker, err := device.New(0)
	if err != nil {
		t.Fatal(err)
	}
	defer worker.Close()

	const seq, nHeads, hd = 7, 3, 8
	invF64 := hostmath.RopeInvFreq(10000, hd)
	invF32 := make([]float32, len(invF64))
	for i := range invF64 {
		invF32[i] = float32(invF64[i])
	}
	rng := rand.New(rand.NewSource(21))
	x := randSlice(rng, seq*nHeads*hd)

	got, err := RoPEHalfForward(worker, x, invF32, seq, nHeads, hd)
	if err != nil {
		t.Fatal(err)
	}
	want := hostRopeHalf(x, invF64, seq, nHeads, hd)
	var maxDiff float64
	for i := range want {
		if diff := math.Abs(float64(got[i]) - want[i]); diff > maxDiff {
			maxDiff = diff
		}
	}
	t.Logf("rope forward: max |dev-host| %.3e", maxDiff)
	const tolerance = 1e-5
	if maxDiff > tolerance {
		t.Fatalf("rope forward %.3e > %.1e", maxDiff, tolerance)
	}
}

// TestRoPEHalfBackwardGradCheck verifies device RoPEHalfBackward against float64
// finite differences of L = sum(dOut ⊙ RoPE(x)) w.r.t. x, using the same invFreq
// as hostmath.
func TestRoPEHalfBackwardGradCheck(t *testing.T) {
	cudatest.Require(t)
	worker, err := device.New(0)
	if err != nil {
		t.Fatal(err)
	}
	defer worker.Close()

	const seq, nHeads, hd = 5, 2, 8
	invF64 := hostmath.RopeInvFreq(10000, hd)
	invF32 := make([]float32, len(invF64))
	for i := range invF64 {
		invF32[i] = float32(invF64[i])
	}
	rng := rand.New(rand.NewSource(12))
	x := randSlice(rng, seq*nHeads*hd)
	dOut := randSlice(rng, seq*nHeads*hd)

	dx, err := RoPEHalfBackward(worker, dOut, invF32, seq, nHeads, hd)
	if err != nil {
		t.Fatal(err)
	}

	loss := func() float64 {
		out := hostRopeHalf(x, invF64, seq, nHeads, hd)
		var l float64
		for i := range out {
			l += float64(dOut[i]) * out[i]
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
		if diff := math.Abs((lp-lm)/(2*eps) - float64(dx[i])); diff > maxDiff {
			maxDiff = diff
		}
	}
	t.Logf("rope backward: max |dx-fd| %.3e", maxDiff)
	const tolerance = 1e-3
	if maxDiff > tolerance {
		t.Fatalf("dx vs fd %.3e > %.1e", maxDiff, tolerance)
	}
}
