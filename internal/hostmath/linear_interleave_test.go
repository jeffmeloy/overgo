package hostmath

import (
	"math/rand/v2"
	"slices"
	"testing"
)

// linearSerial is the one-output loop Linear's kernel was: each output element
// the dot of a weight row and an input row, summed in order.
func linearSerial(dst, x, w []float32, rows, inDim, outDim int) {
	for o := range outDim {
		for r := range rows {
			var sum float32
			for c := range inDim {
				sum += w[o*inDim+c] * x[r*inDim+c]
			}
			dst[r*outDim+o] = sum
		}
	}
}

// TestInterleavedLinearMatchesTheSerialDot holds the three linear kernels, which
// advance four outputs side by side, to the one-output loops they replaced bit
// for bit: over row counts that leave none, one, two and three rows past a
// block of four, with zero gradients in the input of the two that skip them,
// the input gradient added into existing values and set.
func TestInterleavedLinearMatchesTheSerialDot(t *testing.T) {
	t.Parallel()
	random := rand.New(rand.NewPCG(7, 11))
	fill := func(count int, zeros bool) []float32 {
		values := make([]float32, count)
		for index := range values {
			if zeros && random.IntN(5) == 0 {
				continue
			}
			values[index] = random.Float32()*2 - 1
		}
		return values
	}
	for _, shape := range []struct{ rows, inDim, outDim int }{{1, 5, 3}, {4, 8, 5}, {5, 9, 6}, {6, 9, 10}, {7, 13, 11}, {8, 16, 9}, {33, 65, 129}} {
		x, w, dy := fill(shape.rows*shape.inDim, false), fill(shape.outDim*shape.inDim, false), fill(shape.rows*shape.outDim, true)
		want, got := make([]float32, shape.rows*shape.outDim), make([]float32, shape.rows*shape.outDim)
		linearSerial(want, x, w, shape.rows, shape.inDim, shape.outDim)
		Linear(got, x, w, shape.rows, shape.inDim, shape.outDim)
		if !slices.Equal(got, want) {
			t.Errorf("%dx%dx%d: the forward differs from the serial dot", shape.rows, shape.inDim, shape.outDim)
		}
		for _, add := range []bool{false, true} {
			base := fill(shape.rows*shape.inDim, false)
			wantDX, gotDX := slices.Clone(base), slices.Clone(base)
			wantDW, gotDW := fill(shape.outDim*shape.inDim, false), make([]float32, shape.outDim*shape.inDim)
			copy(gotDW, wantDW)
			linearBackwardSerial(wantDX, wantDW, nil, x, w, dy, shape.rows, shape.inDim, shape.outDim, add)
			LinearBackward(gotDX, gotDW, nil, x, w, dy, shape.rows, shape.inDim, shape.outDim, add)
			if !slices.Equal(gotDX, wantDX) || !slices.Equal(gotDW, wantDW) {
				t.Errorf("%dx%dx%d, accumulate %t: the backward differs from the serial loop", shape.rows, shape.inDim, shape.outDim, add)
			}
		}
	}
}
