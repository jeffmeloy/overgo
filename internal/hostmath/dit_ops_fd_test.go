// Finite-difference and adjoint gates for the diffusion-transformer training
// primitives: adaptive shift/scale modulation, axis-partitioned interleaved
// rotary, scaled full-span attention, patchify layout transposes, and the
// parallel linear gradient kernels.
package hostmath

import (
	"math"
	"math/rand"
	"testing"
)

func TestAdaptiveShiftScaleBackwardFiniteDifference(t *testing.T) {
	const rows, d = 3, 5
	rng := rand.New(rand.NewSource(7))
	x := e4bRandVec(rng, rows*d, 0, 0.8)
	shift := e4bRandVec(rng, d, 0, 0.3)
	scale := e4bRandVec(rng, d, 0, 0.3)
	dy := e4bRandVec(rng, rows*d, 0, 1)

	loss := func() float64 {
		out := make([]float32, rows*d)
		AdaptiveShiftScale(out, x, shift, scale, rows, d)
		var s float64
		for i := range out {
			s += float64(out[i]) * float64(dy[i])
		}
		return s
	}
	dx := make([]float32, rows*d)
	dShift := make([]float32, d)
	dScale := make([]float32, d)
	AdaptiveShiftScaleBackward(dx, dShift, dScale, x, scale, dy, rows, d)

	tol := fdTol(fdStep, rows*d)

	checkVectorBackwardFD(t, "dx", x, dx, loss, tol)
	checkVectorBackwardFD(t, "dShift", shift, dShift, loss, tol)
	checkVectorBackwardFD(t, "dScale", scale, dScale, loss, tol)
}

func TestMultiplyBackward(t *testing.T) {
	dLeft, dRight := make([]float32, 2), make([]float32, 2)
	MultiplyBackward(dLeft, dRight, []float32{2, 3}, []float32{4, 5}, []float32{7, 11})
	if dLeft[0] != 28 || dLeft[1] != 55 || dRight[0] != 14 || dRight[1] != 33 {
		t.Fatalf("dLeft=%v dRight=%v", dLeft, dRight)
	}
}

func TestAxisRotaryInterleavedBackwardFiniteDifference(t *testing.T) {
	spans := [3]int{4, 4, 4}
	invFreq := AxisRotaryInvFreq(10000, spans)
	positions := [3]int{3, 1, 2}
	rng := rand.New(rand.NewSource(9))
	x := e4bRandVec(rng, 12, 0, 0.7)
	dy := e4bRandVec(rng, 12, 0, 1)

	loss := func() float64 {
		out := append([]float32(nil), x...)
		ApplyAxisRotaryInterleaved(out, spans, invFreq, positions)
		var s float64
		for i := range out {
			s += float64(out[i]) * float64(dy[i])
		}
		return s
	}
	dx := append([]float32(nil), dy...)
	AxisRotaryInterleavedBackward(dx, spans, invFreq, positions)

	tol := fdTol(fdStep, len(x))
	checkVectorBackwardFD(t, "dx", x, dx, loss, tol)
	// Empty-span variant (head widths that give an axis zero channels).
	narrow := [3]int{4, 0, 0}
	narrowFreq := AxisRotaryInvFreq(10000, narrow)
	row := e4bRandVec(rng, 4, 0, 0.5)
	ApplyAxisRotaryInterleaved(row, narrow, narrowFreq, positions)
	AxisRotaryInterleavedBackward(row, narrow, narrowFreq, positions)
}

func TestScaledMaskedBidirectionalAttentionBackwardFiniteDifference(t *testing.T) {
	const querySeq, keySeq, heads, kvHeads, headDim = 2, 3, 2, 1, 2
	scale := 1 / math.Sqrt(float64(headDim))
	q := []float32{0.2, -0.1, 0.4, 0.3, -0.2, 0.5, 0.1, -0.4}
	k := []float32{0.3, -0.2, -0.1, 0.4, 0.5, 0.2}
	v := []float32{0.7, -0.3, 0.2, 0.6, -0.4, 0.8}
	dOut := []float32{0.5, -0.2, 0.1, 0.4, -0.3, 0.6, 0.2, -0.5}
	mask := []bool{true, false, true}
	dq, dk, dv := make([]float32, len(q)), make([]float32, len(k)), make([]float32, len(v))
	ScaledMaskedBidirectionalAttentionBackward(dq, dk, dv, q, k, v, dOut, querySeq, keySeq, heads, kvHeads, headDim, scale, mask)
	objective := func() float64 {
		out := make([]float32, len(q))
		ScaledMaskedBidirectionalAttention(out, q, k, v, querySeq, keySeq, heads, kvHeads, headDim, scale, mask)
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

// TestScaledAttentionScaleOneBitIdentical: the scale-1 path must reproduce
// the historical unscaled kernels exactly (existing callers unchanged).
func TestScaledAttentionScaleOneBitIdentical(t *testing.T) {
	const querySeq, keySeq, heads, kvHeads, headDim = 3, 4, 4, 2, 3
	rng := rand.New(rand.NewSource(21))
	q := e4bRandVec(rng, querySeq*heads*headDim, 0, 1)
	k := e4bRandVec(rng, keySeq*kvHeads*headDim, 0, 1)
	v := e4bRandVec(rng, keySeq*kvHeads*headDim, 0, 1)
	dOut := e4bRandVec(rng, querySeq*heads*headDim, 0, 1)
	a := make([]float32, len(q))
	b := make([]float32, len(q))
	MaskedBidirectionalAttention(a, q, k, v, querySeq, keySeq, heads, kvHeads, headDim, nil)
	ScaledMaskedBidirectionalAttention(b, q, k, v, querySeq, keySeq, heads, kvHeads, headDim, 1, nil)
	for i := range a {
		if a[i] != b[i] {
			t.Fatalf("forward diverges at %d: %g vs %g", i, a[i], b[i])
		}
	}
	dqA, dkA, dvA := make([]float32, len(q)), make([]float32, len(k)), make([]float32, len(v))
	dqB, dkB, dvB := make([]float32, len(q)), make([]float32, len(k)), make([]float32, len(v))
	MaskedBidirectionalAttentionBackward(dqA, dkA, dvA, q, k, v, dOut, querySeq, keySeq, heads, kvHeads, headDim, nil)
	ScaledMaskedBidirectionalAttentionBackward(dqB, dkB, dvB, q, k, v, dOut, querySeq, keySeq, heads, kvHeads, headDim, 1, nil)
	for i := range dqA {
		if dqA[i] != dqB[i] {
			t.Fatalf("dq diverges at %d", i)
		}
	}
	for i := range dkA {
		if dkA[i] != dkB[i] || dvA[i] != dvB[i] {
			t.Fatalf("dk/dv diverges at %d", i)
		}
	}
}

// TestPatchifyChannelMajorAdjoint: both layout maps are permutations; the
// transposes must satisfy the exact adjoint identity <A x, y> == <x, A^T y>
// and invert the forward bit-for-bit.
func TestPatchifyChannelMajorAdjoint(t *testing.T) {
	const channels, frames, height, width = 3, 2, 4, 6
	const p0, p1, p2 = 2, 2, 3
	seq := frames / p0 * (height / p1) * (width / p2)
	elements := channels * frames * height * width
	rng := rand.New(rand.NewSource(31))

	x := e4bRandVec(rng, elements, 0, 1)
	y := e4bRandVec(rng, seq*channels*p0*p1*p2, 0, 1)
	ax := make([]float32, len(y))
	PatchifyChannelMajor(ax, x, channels, frames, height, width, p0, p1, p2)
	// The transpose scatter via the shared index map (no production transpose
	// exists: the latent is data, nothing backpropagates through patchify).
	aty := make([]float32, len(x))
	patchifyChannelMajorIndexed(channels, frames, height, width, p0, p1, p2, func(rowIndex, latentIndex int) {
		aty[latentIndex] = y[rowIndex]
	})
	var lhs, rhs float64
	for i := range y {
		lhs += float64(ax[i]) * float64(y[i])
	}
	for i := range x {
		rhs += float64(x[i]) * float64(aty[i])
	}
	// Same multiset of products on both sides; only f64 summation order differs.
	if math.Abs(lhs-rhs) > 1e-9*(1+math.Abs(lhs)) {
		t.Fatalf("patchify adjoint identity: %g vs %g", lhs, rhs)
	}
	roundTrip := make([]float32, len(x))
	patchifyChannelMajorIndexed(channels, frames, height, width, p0, p1, p2, func(rowIndex, latentIndex int) {
		roundTrip[latentIndex] = ax[rowIndex]
	})
	for i := range x {
		if roundTrip[i] != x[i] {
			t.Fatalf("patchify round trip diverges at %d", i)
		}
	}

	rows := e4bRandVec(rng, seq*channels*p0*p1*p2, 0, 1)
	latentGrad := e4bRandVec(rng, elements, 0, 1)
	latent := make([]float32, elements)
	UnpatchifyChannelMajor(latent, rows, channels, frames, height, width, p0, p1, p2)
	rowGrad := make([]float32, len(rows))
	UnpatchifyChannelMajorTranspose(rowGrad, latentGrad, channels, frames, height, width, p0, p1, p2)
	lhs, rhs = 0, 0
	for i := range latent {
		lhs += float64(latent[i]) * float64(latentGrad[i])
	}
	for i := range rows {
		rhs += float64(rows[i]) * float64(rowGrad[i])
	}
	if math.Abs(lhs-rhs) > 1e-9*(1+math.Abs(lhs)) {
		t.Fatalf("unpatchify adjoint identity: %g vs %g", lhs, rhs)
	}
	rowsBack := make([]float32, len(rows))
	UnpatchifyChannelMajorTranspose(rowsBack, latent, channels, frames, height, width, p0, p1, p2)
	for i := range rows {
		if rowsBack[i] != rows[i] {
			t.Fatalf("unpatchify round trip diverges at %d", i)
		}
	}
}

// TestLinearGradientKernelsMatchLinearBackward: the parallel dW/dx kernels
// must agree with the serial LinearBackward reference.
func TestLinearGradientKernelsMatchLinearBackward(t *testing.T) {
	const rows, inDim, outDim = 4, 5, 3
	rng := rand.New(rand.NewSource(41))
	x := e4bRandVec(rng, rows*inDim, 0, 1)
	w := e4bRandVec(rng, outDim*inDim, 0, 1)
	dy := e4bRandVec(rng, rows*outDim, 0, 1)

	dxWant := make([]float32, rows*inDim)
	dWWant := make([]float32, outDim*inDim)
	dBWant := make([]float32, outDim)
	LinearBackward(dxWant, dWWant, dBWant, x, w, dy, rows, inDim, outDim, false)

	dxGot := make([]float32, rows*inDim)
	LinearBackwardInput(dxGot, dy, w, rows, inDim, outDim)
	dWGot := make([]float32, outDim*inDim)
	LinearWeightGradient(dWGot, x, dy, rows, inDim, outDim)
	for i := range dxWant {
		if dxWant[i] != dxGot[i] {
			t.Fatalf("dx diverges at %d: %g vs %g", i, dxWant[i], dxGot[i])
		}
	}
	for i := range dWWant {
		if dWWant[i] != dWGot[i] {
			t.Fatalf("dW diverges at %d: %g vs %g", i, dWWant[i], dWGot[i])
		}
	}
}
