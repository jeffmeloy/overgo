package tensorstats

import (
	"math"
	"testing"
)

func TestEffectiveRankUniformSpectrumIsFull(t *testing.T) {
	// Equal singular values spread energy across every direction -> full rank 1.
	got, ok := EffectiveRank([]float64{1, 1, 1, 1})
	if !ok || math.Abs(got-1) > 1e-12 {
		t.Errorf("uniform effective rank = %.6g (ok=%v), want 1", got, ok)
	}
}

func TestEffectiveRankDominantIsMinimal(t *testing.T) {
	// One direction carries all energy -> effective rank 1 of n -> 1/n.
	got, ok := EffectiveRank([]float64{5, 0, 0, 0})
	if !ok || math.Abs(got-0.25) > 1e-12 {
		t.Errorf("dominant effective rank = %.6g, want 0.25", got)
	}
}

func TestEffectiveRankTwoOfFour(t *testing.T) {
	got, _ := EffectiveRank([]float64{3, 3, 0, 0}) // exp(ln2)/4 = 2/4
	if math.Abs(got-0.5) > 1e-12 {
		t.Errorf("two-of-four effective rank = %.6g, want 0.5", got)
	}
}

func TestEffectiveRankIsScaleInvariant(t *testing.T) {
	a, _ := EffectiveRank([]float64{8, 4, 2, 1})
	b, _ := EffectiveRank([]float64{80, 40, 20, 10})
	if math.Abs(a-b) > 1e-12 {
		t.Errorf("scaling changed effective rank: %.9g vs %.9g", a, b)
	}
	if a <= 0.25 || a >= 1 {
		t.Errorf("power-law effective rank = %.4f, want strictly between 1/n and 1", a)
	}
}

func TestEffectiveRankOrdering(t *testing.T) {
	// Flatter spectra have higher effective rank.
	flat, _ := EffectiveRank([]float64{4, 3, 2, 1})
	peaked, _ := EffectiveRank([]float64{100, 2, 1, 1})
	if flat <= peaked {
		t.Errorf("flat spectrum %.4f should exceed peaked %.4f", flat, peaked)
	}
}

func TestEffectiveRankInvalidInputs(t *testing.T) {
	cases := [][]float64{
		{},               // empty
		{0, 0, 0},        // zero matrix
		{1, -1},          // negative singular value
		{1, math.NaN()},  // non-finite
		{1, math.Inf(1)}, // non-finite
	}
	for _, c := range cases {
		if _, ok := EffectiveRank(c); ok {
			t.Errorf("EffectiveRank(%v) reported ok, want false", c)
		}
	}
}
