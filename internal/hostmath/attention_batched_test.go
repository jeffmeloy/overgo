package hostmath

import (
	"math"
	"testing"
)

func TestLayerNormAndAdaptiveAffineF64(t *testing.T) {
	normed := LayerNormF64([]float64{1, 3}, 1, 2, 0)
	if math.Abs(normed[0]+1) > 1e-12 || math.Abs(normed[1]-1) > 1e-12 {
		t.Fatalf("layer norm=%v", normed)
	}
	got := AdaptiveAffineF64(normed, []float64{0, 0}, []float32{0, 0, 1, 2}, 1, 2)
	if math.Abs(got[0]) > 1e-12 || math.Abs(got[1]-3) > 1e-12 {
		t.Fatalf("adaptive affine=%v", got)
	}
}
