//go:build windows

package devicemath

import (
	"math"
	"math/rand"
	"testing"

	"overgo/internal/cuda/device"
	"overgo/internal/cuda/driver"
	cudatest "overgo/internal/cuda/testutil"
	"overgo/internal/testutil"
)

func randSlice(rng *rand.Rand, n int) []float32 {
	s := make([]float32, n)
	for i := range s {
		s[i] = float32(rng.NormFloat64())
	}
	return s
}

func TestLinearBackwardTResidentMatchesShared(t *testing.T) {
	cudatest.Require(t)
	worker, err := device.New(0)
	if err != nil {
		t.Fatal(err)
	}
	defer worker.Close()
	const rows, in, out = 7, 5, 9
	rng := rand.New(rand.NewSource(41))
	x, weight, dY := randSlice(rng, rows*in), randSlice(rng, out*in), randSlice(rng, rows*out)
	wantX, wantWeight, err := LinearBackwardT(worker, x, weight, dY, rows, in, out)
	if err != nil {
		t.Fatal(err)
	}
	var pointers []driver.DevicePtr
	allocate := func(count int, initial []float32) driver.DevicePtr {
		t.Helper()
		pointer, err := AllocResidentF32(worker, count, initial)
		if err != nil {
			t.Fatal(err)
		}
		pointers = append(pointers, pointer)
		return pointer
	}
	defer func() {
		if err := FreeResident(worker, pointers...); err != nil {
			t.Error(err)
		}
	}()
	xPtr := allocate(len(x), x)
	weightPtr := allocate(len(weight), weight)
	dYPtr := allocate(len(dY), dY)
	dXPtr := allocate(len(wantX), nil)
	dWeightPtr := allocate(len(wantWeight), nil)
	if err := LinearBackwardTResident(worker, xPtr, weightPtr, dYPtr, dXPtr, dWeightPtr, rows, in, out); err != nil {
		t.Fatal(err)
	}
	gotX, gotWeight := make([]float32, len(wantX)), make([]float32, len(wantWeight))
	if err := ReadResident(worker, dXPtr, ResidentSlice{Data: gotX}); err != nil {
		t.Fatal(err)
	}
	if err := ReadResident(worker, dWeightPtr, ResidentSlice{Data: gotWeight}); err != nil {
		t.Fatal(err)
	}
	xDelta, weightDelta := testutil.MaxAbsDiff(gotX, wantX), testutil.MaxAbsDiff(gotWeight, wantWeight)
	t.Logf("resident/shared linear VJP dX=%.3e dW=%.3e", xDelta, weightDelta)
	if xDelta != 0 || weightDelta != 0 {
		t.Fatalf("resident linear VJP differs: dX=%.3e dW=%.3e", xDelta, weightDelta)
	}
}

// hostLinearLoss returns L = sum(dY ⊙ (X·W)) in fp64 -- the scalar whose
// gradients w.r.t. X and W are exactly dY·Wᵀ and Xᵀ·dY.
func hostLinearLoss(x, w, dY []float32, rows, in, out int) float64 {
	var loss float64
	for r := range rows {
		for o := range out {
			var y float64
			for i := range in {
				y += float64(x[r*in+i]) * float64(w[i*out+o])
			}
			loss += float64(dY[r*out+o]) * y
		}
	}
	return loss
}

// TestLinearForwardTMatchesHost verifies device LinearForwardT (Y = X·Wᵀ,
// W[outDim,in]) against an fp64 reference.
func TestLinearForwardTMatchesHost(t *testing.T) {
	cudatest.Require(t)
	worker, err := device.New(0)
	if err != nil {
		t.Fatal(err)
	}
	defer worker.Close()

	const rows, in, outDim = 12, 20, 16
	rng := rand.New(rand.NewSource(41))
	x := randSlice(rng, rows*in)
	w := randSlice(rng, outDim*in) // [outDim, in]

	got, err := LinearForwardT(worker, x, w, rows, in, outDim)
	if err != nil {
		t.Fatal(err)
	}
	var maxDiff float64
	for r := range rows {
		for o := range outDim {
			var y float64
			for i := range in {
				y += float64(x[r*in+i]) * float64(w[o*in+i]) // Wᵀ: w[o,i]
			}
			if diff := math.Abs(float64(got[r*outDim+o]) - y); diff > maxDiff {
				maxDiff = diff
			}
		}
	}
	t.Logf("linearT forward: max |dev-host| %.3e", maxDiff)
	const tolerance = 1e-4
	if maxDiff > tolerance {
		t.Fatalf("linearT forward %.3e > %.1e", maxDiff, tolerance)
	}
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

// hostLinearLossT returns L = sum(dY ⊙ (X·Wᵀ)) in fp64 (W stored [outDim,in]).
func hostLinearLossT(x, w, dY []float32, rows, in, outDim int) float64 {
	var loss float64
	for r := range rows {
		for o := range outDim {
			var y float64
			for i := range in {
				y += float64(x[r*in+i]) * float64(w[o*in+i])
			}
			loss += float64(dY[r*outDim+o]) * y
		}
	}
	return loss
}

// TestLinearBackwardTGradCheck verifies device LinearBackwardT (the Y=X·Wᵀ
// densecausal convention) against a central finite-difference of the host loss.
func TestLinearBackwardTGradCheck(t *testing.T) {
	cudatest.Require(t)
	worker, err := device.New(0)
	if err != nil {
		t.Fatal(err)
	}
	defer worker.Close()

	const rows, in, outDim = 6, 5, 4
	rng := rand.New(rand.NewSource(5))
	x := randSlice(rng, rows*in)
	w := randSlice(rng, outDim*in)
	dY := randSlice(rng, rows*outDim)

	dX, dW, err := LinearBackwardT(worker, x, w, dY, rows, in, outDim)
	if err != nil {
		t.Fatal(err)
	}

	const eps = 1e-2
	fd := func(perturb []float32, idx int) float64 {
		orig := perturb[idx]
		perturb[idx] = orig + float32(eps)
		lp := hostLinearLossT(x, w, dY, rows, in, outDim)
		perturb[idx] = orig - float32(eps)
		lm := hostLinearLossT(x, w, dY, rows, in, outDim)
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
	t.Logf("LinearBackwardT: max |dX-fd| %.3e  max |dW-fd| %.3e", maxX, maxW)
	const tolerance = 1e-3
	if maxX > tolerance || maxW > tolerance {
		t.Fatalf("dX %.3e / dW %.3e > %.1e", maxX, maxW, tolerance)
	}
}
