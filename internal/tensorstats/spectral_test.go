package tensorstats

import (
	"math"
	"sort"
	"testing"
)

func sortedDesc(v []float64) []float64 {
	out := append([]float64(nil), v...)
	sort.Sort(sort.Reverse(sort.Float64Slice(out)))
	return out
}

func closeVec(t *testing.T, label string, got, want []float64) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s length = %d, want %d", label, len(got), len(want))
	}
	for i := range want {
		if math.Abs(got[i]-want[i]) > 1e-9 {
			t.Errorf("%s[%d] = %.9g, want %.9g", label, i, got[i], want[i])
		}
	}
}

func TestSymmetricEigenvalues2x2(t *testing.T) {
	// [[2,1],[1,2]] has eigenvalues 3 and 1.
	closeVec(t, "eig", sortedDesc(symmetricEigenvalues([]float64{2, 1, 1, 2}, 2)), []float64{3, 1})
}

func TestSymmetricEigenvaluesTridiagonal(t *testing.T) {
	// [[2,-1,0],[-1,2,-1],[0,-1,2]] has eigenvalues 2 and 2±√2.
	s := []float64{2, -1, 0, -1, 2, -1, 0, -1, 2}
	closeVec(t, "eig", sortedDesc(symmetricEigenvalues(s, 3)), []float64{2 + math.Sqrt2, 2, 2 - math.Sqrt2})
}

func TestSingularValuesKnownMatrices(t *testing.T) {
	sv, ok := SingularValues([]float64{3, 0, 0, 1}, 2, 2) // diag(3,1)
	if !ok {
		t.Fatal("diag failed")
	}
	closeVec(t, "diag", sv, []float64{3, 1})
	sv, _ = SingularValues([]float64{1, 1, 1, 1}, 2, 2) // rank-1 ones -> {2,0}
	closeVec(t, "ones", sv, []float64{2, 0})
	sv, _ = SingularValues([]float64{1, 0, 0, 0, 1, 0}, 2, 3) // wide, cols>rows
	closeVec(t, "wide", sv, []float64{1, 1})
	sv, _ = SingularValues([]float64{1, 0, 0, 1, 0, 0}, 3, 2) // tall, rows>cols
	closeVec(t, "tall", sv, []float64{1, 1})
}

func TestEffectiveRankOfComposition(t *testing.T) {
	er, ok := EffectiveRankOf([]float64{1, 0, 0, 1}, 2, 2) // identity -> full
	if !ok || math.Abs(er-1) > 1e-9 {
		t.Errorf("identity effective rank = %.9g, want 1", er)
	}
	er, _ = EffectiveRankOf([]float64{1, 1, 1, 1}, 2, 2) // rank-1 -> 1/2
	if math.Abs(er-0.5) > 1e-9 {
		t.Errorf("rank-1 effective rank = %.9g, want 0.5", er)
	}
}

func TestSingularValuesRejectsBadDims(t *testing.T) {
	if _, ok := SingularValues([]float64{1, 2, 3}, 2, 2); ok {
		t.Error("length mismatch must fail")
	}
	if _, ok := SingularValues(nil, 0, 2); ok {
		t.Error("zero dimension must fail")
	}
}

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
