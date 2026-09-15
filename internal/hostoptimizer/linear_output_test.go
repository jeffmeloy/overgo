package hostoptimizer

import (
	"math/rand/v2"
	"slices"
	"testing"

	"overgo/internal/hostmath"
)

// TestLinearOutputProjectionMatchesHostmath holds the training-side kernel
// to hostmath.LinearInputProjection bit for bit, with and without the input
// projection and the bias, on shapes that exercise the fan-out.
func TestLinearOutputProjectionMatchesHostmath(t *testing.T) {
	t.Parallel()
	generator := rand.New(rand.NewPCG(7, 11))
	fill := func(n int) []float32 {
		values := make([]float32, n)
		for index := range values {
			values[index] = float32(generator.NormFloat64())
		}
		return values
	}
	for _, shape := range []struct{ rows, inDim, outDim int }{{181, 64, 4099}, {1, 32, 33}, {7, 16, 1}} {
		x := fill(shape.rows * shape.inDim)
		projection := fill(shape.inDim * shape.inDim)
		weight := fill(shape.outDim * shape.inDim)
		bias := fill(shape.outDim)
		for _, variant := range []struct {
			projection, bias []float32
		}{{nil, nil}, {projection, nil}, {nil, bias}, {projection, bias}} {
			want := make([]float32, shape.rows*shape.outDim)
			wantProjected := make([]float32, shape.rows*shape.inDim)
			hostmath.LinearInputProjection(want, wantProjected, x, variant.projection, weight, variant.bias, shape.rows, shape.inDim, shape.outDim)
			got := make([]float32, shape.rows*shape.outDim)
			gotProjected := make([]float32, shape.rows*shape.inDim)
			LinearOutputProjection(got, gotProjected, x, variant.projection, weight, variant.bias, shape.rows, shape.inDim, shape.outDim)
			if !slices.Equal(got, want) || !slices.Equal(gotProjected, wantProjected) {
				t.Fatalf("%dx%dx%d projection=%v bias=%v: training kernel differs from hostmath", shape.rows, shape.inDim, shape.outDim, variant.projection != nil, variant.bias != nil)
			}
		}
	}
}
