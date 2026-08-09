package seriesforecast

import (
	"math"
	"testing"
)

func TestPadToPatchesFrontPadsWithMaskOnes(t *testing.T) {
	series := []float32{1, 2, 3, 4, 5}
	padded, masks, err := padToPatches(series, nil, 4)
	if err != nil {
		t.Fatal(err)
	}
	if len(padded) != 8 || len(masks) != 8 {
		t.Fatalf("padded=%d masks=%d, want 8", len(padded), len(masks))
	}
	for i := 0; i < 3; i++ {
		if masks[i] != 1 || padded[i] != 0 {
			t.Fatalf("front element %d: padded=%v mask=%v, want 0/1", i, padded[i], masks[i])
		}
	}
	for i := 3; i < 8; i++ {
		if masks[i] != 0 || padded[i] != series[i-3] {
			t.Fatalf("element %d not preserved", i)
		}
	}
	exact, exactMasks, err := padToPatches([]float32{1, 2, 3, 4}, nil, 4)
	if err != nil || len(exact) != 4 {
		t.Fatalf("exact multiple should pass through: %v", err)
	}
	for _, m := range exactMasks {
		if m != 0 {
			t.Fatal("exact multiple must have all-legit masks")
		}
	}
}

// Running stats must accumulate ACROSS patches: with two patches of constant
// values a then b, patch 0 sees mu=a sigma=0 and patch 1 sees the merged
// stats of both.
func TestPatchStatsAreRunningAcrossPatches(t *testing.T) {
	series := []float32{2, 2, 6, 6}
	masks := make([]float32, 4)
	mu := make([]float64, 2)
	sigma := make([]float64, 2)
	patchStats(series, masks, 2, mu, sigma)
	if mu[0] != 2 || sigma[0] != 0 {
		t.Fatalf("patch 0 stats mu=%v sigma=%v, want 2/0", mu[0], sigma[0])
	}
	if mu[1] != 4 || math.Abs(sigma[1]-2) > 1e-12 {
		t.Fatalf("patch 1 running stats mu=%v sigma=%v, want 4/2", mu[1], sigma[1])
	}
}

// Masked entries are excluded from statistics entirely.
func TestPatchStatsIgnoreMaskedEntries(t *testing.T) {
	series := []float32{100, 3, 100, 5}
	masks := []float32{1, 0, 1, 0}
	mu := make([]float64, 2)
	sigma := make([]float64, 2)
	patchStats(series, masks, 2, mu, sigma)
	if mu[1] != 4 || math.Abs(sigma[1]-1) > 1e-12 {
		t.Fatalf("masked stats mu=%v sigma=%v, want 4/1", mu[1], sigma[1])
	}
}

// A hand-computed tiny residual block pins the exact composition
// dst = W_out*SiLU(W_h*x + b_h) + b_out + W_r*x + b_r.
func TestResidualBlockTinyCase(t *testing.T) {
	model := &Model{
		Dims: Dims{},
		Shapes: map[string][]int{
			"blk.hidden_layer.weight":   {1, 2},
			"blk.output_layer.weight":   {1, 1},
			"blk.residual_layer.weight": {1, 2},
		},
		Weights: map[string][]float32{
			"blk.hidden_layer.weight":   {1, 1}, // sum of inputs
			"blk.hidden_layer.bias":     {0},
			"blk.output_layer.weight":   {2},     // doubled
			"blk.residual_layer.weight": {1, -1}, // difference
		},
	}
	dst := make([]float32, 1)
	if err := model.residualBlock(dst, "blk", []float32{3, 1}); err != nil {
		t.Fatal(err)
	}
	// hidden = 4; SiLU(4) = 4/(1+e^-4); out = 2*SiLU(4); residual = 2.
	want := float32(2*(4/(1+math.Exp(-4))) + 2)
	if math.Abs(float64(dst[0]-want)) > 1e-6 {
		t.Fatalf("residual block = %v, want %v", dst[0], want)
	}
}

// Forecast must refuse until the decoder stack lands: withheld is honest,
// wrong numbers are not.
func TestForecastRefusesWithoutDecoder(t *testing.T) {
	model := &Model{Dims: Dims{Layers: 20}}
	if _, err := model.Forecast([]float32{1, 2, 3}, 8); err == nil {
		t.Fatal("Forecast produced output without a ported decoder")
	}
}
