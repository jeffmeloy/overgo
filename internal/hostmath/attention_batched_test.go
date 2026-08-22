package hostmath

import (
	"math"
	"testing"
)

func TestGroupedCausalAttentionF64Mask(t *testing.T) {
	q := []float64{1, 1}
	k := []float64{1, 1}
	v := []float64{2, 4}
	got := GroupedCausalAttentionF64(q, k, v, 2, 1, 1, 1, []bool{true, false})
	if len(got) != 2 || got[0] != 2 || got[1] != 2 {
		t.Fatalf("attention=%v", got)
	}
}

func TestNormalizeHeadsRotaryHalfF64(t *testing.T) {
	values := []float64{3, 4}
	NormalizeHeadsRotaryHalfF64(values, []float32{1, 1}, 1, 1, 2, 0, []float64{1, 1}, []float64{0, 0})
	want := []float64{3 / math.Sqrt(12.5), 4 / math.Sqrt(12.5)}
	if math.Abs(values[0]-want[0]) > 1e-12 || math.Abs(values[1]-want[1]) > 1e-12 {
		t.Fatalf("normalized=%v want=%v", values, want)
	}
}

func TestStandardRMSNormF64(t *testing.T) {
	got := StandardRMSNormF64([]float64{3, 4}, []float32{1, 2}, 1, 2, 0)
	want := []float64{3 / math.Sqrt(12.5), 8 / math.Sqrt(12.5)}
	if math.Abs(got[0]-want[0]) > 1e-12 || math.Abs(got[1]-want[1]) > 1e-12 {
		t.Fatalf("normalized=%v want=%v", got, want)
	}
}

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

func TestAddResidualF64IntoIdentity(t *testing.T) {
	output := []float32{1, 2}
	if err := AddResidualF64Into(output, []float32{3, 4}, nil, nil, 1, 1, 2); err != nil {
		t.Fatal(err)
	}
	if output[0] != 4 || output[1] != 6 {
		t.Fatalf("residual=%v", output)
	}
}
