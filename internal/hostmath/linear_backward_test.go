package hostmath

import (
	"math/rand/v2"
	"slices"
	"testing"
)

// linearBackwardSerial is the one-thread loop LinearBackward was: every batch
// row in order, each output in order, each element adding its terms in that
// order.
func linearBackwardSerial(dx, dW, dB, x, w, dy []float32, rows, inDim, outDim int, addDX bool) {
	for r := range rows {
		xRow := x[r*inDim : (r+1)*inDim]
		dyRow := dy[r*outDim : (r+1)*outDim]
		var dxRow []float32
		if dx != nil {
			dxRow = dx[r*inDim : (r+1)*inDim]
			if !addDX {
				clear(dxRow)
			}
		}
		for o := range outDim {
			g := dyRow[o]
			if dB != nil {
				dB[o] += g
			}
			if g == 0 {
				continue
			}
			if dW != nil {
				dWRow := dW[o*inDim : (o+1)*inDim]
				for c := range inDim {
					dWRow[c] += g * xRow[c]
				}
			}
			if dxRow != nil {
				wRow := w[o*inDim : (o+1)*inDim]
				for c := range inDim {
					dxRow[c] += g * wRow[c]
				}
			}
		}
	}
}

// TestLinearBackwardRunsOnThePool holds LinearBackward, which fans out over the
// worker pool by the rows each part owns, to the serial loop's result bit for
// bit: over shapes that split unevenly across workers and one large enough for
// the calibrated dispatch to fan out, with dx set and accumulated, and with
// each of dx, dW and dB absent. Zero gradients, which the loop skips, are among
// the inputs.
func TestLinearBackwardRunsOnThePool(t *testing.T) {
	t.Parallel()
	random := rand.New(rand.NewPCG(1, 2))
	fill := func(count int, zeros bool) []float32 {
		values := make([]float32, count)
		for index := range values {
			if zeros && random.IntN(4) == 0 {
				continue
			}
			values[index] = random.Float32()*2 - 1
		}
		return values
	}
	for _, shape := range []struct{ rows, inDim, outDim int }{{1, 5, 3}, {7, 13, 11}, {33, 65, 129}, {64, 512, 512}} {
		x, w, dy := fill(shape.rows*shape.inDim, false), fill(shape.outDim*shape.inDim, false), fill(shape.rows*shape.outDim, true)
		for _, mode := range []struct {
			name            string
			dx, dW, dB, add bool
		}{
			{"all set", true, true, true, false}, {"all accumulated", true, true, true, true},
			{"no input gradient", false, true, true, false}, {"no weight gradient", true, false, true, true}, {"no bias gradient", true, true, false, false},
		} {
			buffers := func() (dx, dW, dB []float32) {
				if mode.dx {
					dx = fill(shape.rows*shape.inDim, false)
				}
				if mode.dW {
					dW = fill(shape.outDim*shape.inDim, false)
				}
				if mode.dB {
					dB = fill(shape.outDim, false)
				}
				return dx, dW, dB
			}
			random = rand.New(rand.NewPCG(3, uint64(shape.rows)))
			wantDX, wantDW, wantDB := buffers()
			random = rand.New(rand.NewPCG(3, uint64(shape.rows)))
			gotDX, gotDW, gotDB := buffers()
			linearBackwardSerial(wantDX, wantDW, wantDB, x, w, dy, shape.rows, shape.inDim, shape.outDim, mode.add)
			LinearBackward(gotDX, gotDW, gotDB, x, w, dy, shape.rows, shape.inDim, shape.outDim, mode.add)
			if !slices.Equal(gotDX, wantDX) || !slices.Equal(gotDW, wantDW) || !slices.Equal(gotDB, wantDB) {
				t.Errorf("%dx%dx%d, %s: the pooled result differs from the serial loop", shape.rows, shape.inDim, shape.outDim, mode.name)
			}
		}
	}
}
