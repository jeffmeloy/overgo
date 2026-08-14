package tensorstats

import (
	"math"
	"testing"
)

func mustCharacterize(t *testing.T, samples []float64) Characterization {
	t.Helper()
	c, ok := Characterize(uint64(len(samples)), samples)
	if !ok {
		t.Fatalf("characterize failed for %v", samples)
	}
	return c
}

func TestShapeFeaturesAreScaleInvariant(t *testing.T) {
	// Scale-free by construction: positive scaling leaves each feature invariant
	// up to floating-point rounding in the energy accumulation.
	base := ShapeFeatures(mustCharacterize(t, []float64{1, 2, 3, 4, 5, 6, 7, 8}))
	scaled := ShapeFeatures(mustCharacterize(t, []float64{10, 20, 30, 40, 50, 60, 70, 80}))
	for i := range base {
		if math.Abs(base[i]-scaled[i]) > 1e-9 {
			t.Errorf("feature %d changed under scaling: base=%.12g scaled=%.12g", i, base[i], scaled[i])
		}
	}
}

func TestNearestByRankOrdersAndSkipsSelf(t *testing.T) {
	pool := []Characterization{
		mustCharacterize(t, []float64{1, 2, 3, 4, 5, 6, 7, 8}),     // 0: target
		mustCharacterize(t, []float64{2, 4, 6, 8, 10, 12, 14, 16}), // 1: same shape, 2x -> identical features -> nearest
		mustCharacterize(t, []float64{1, 1, 1, 1, 1, 1, 1, 50}),    // 2: heavy right tail
		mustCharacterize(t, []float64{0, 0, 0, 3, 0, 0, 6, 0}),     // 3: sparse
	}
	neighbors := Nearest(pool, 0, 2)
	if len(neighbors) != 2 {
		t.Fatalf("got %d neighbors, want 2", len(neighbors))
	}
	if neighbors[0].Index != 1 {
		t.Errorf("nearest = index %d (dist %.4f), want 1 (same shape)", neighbors[0].Index, neighbors[0].Distance)
	}
	if math.Abs(neighbors[0].Distance) > 1e-12 {
		t.Errorf("identical shape must have zero rank distance, got %.6g", neighbors[0].Distance)
	}
	for _, n := range neighbors {
		if n.Index == 0 {
			t.Error("Nearest must skip the target itself")
		}
	}
	if neighbors[0].Distance > neighbors[1].Distance {
		t.Error("neighbors must be ascending by distance")
	}
}

func TestFootruleIsBoundedAndSymmetric(t *testing.T) {
	// Two maximally different rank vectors give footrule 1; a vector with itself
	// gives 0; the measure is symmetric.
	lo := [7]float64{0, 0, 0, 0, 0, 0, 0}
	hi := [7]float64{1, 1, 1, 1, 1, 1, 1}
	if d := footrule(lo, hi); math.Abs(d-1) > 1e-12 {
		t.Errorf("extreme footrule = %.6g, want 1", d)
	}
	if d := footrule(hi, hi); d != 0 {
		t.Errorf("self footrule = %.6g, want 0", d)
	}
	if footrule(lo, hi) != footrule(hi, lo) {
		t.Error("footrule must be symmetric")
	}
}

func TestFeatureRanksAreEmpiricalCDFPositions(t *testing.T) {
	// Four tensors with strictly increasing sparsity map to evenly spaced ranks
	// 0, 1/3, 2/3, 1 on the zero-fraction feature (index 2), regardless of the
	// raw fraction magnitudes.
	pool := []Characterization{
		mustCharacterize(t, []float64{1, 2, 3, 4, 5, 6, 7, 8}), // 0 zeros
		mustCharacterize(t, []float64{0, 2, 3, 4, 5, 6, 7, 8}), // 1 zero
		mustCharacterize(t, []float64{0, 0, 3, 4, 5, 6, 7, 8}), // 2 zeros
		mustCharacterize(t, []float64{0, 0, 0, 4, 5, 6, 7, 8}), // 3 zeros
	}
	ranks := featureRanks(pool)
	want := []float64{0, 1.0 / 3, 2.0 / 3, 1}
	for i, w := range want {
		if math.Abs(ranks[i][2]-w) > 1e-12 {
			t.Errorf("zero-fraction rank[%d] = %.4f, want %.4f", i, ranks[i][2], w)
		}
	}
}

func TestNearestTieBreakAndGuards(t *testing.T) {
	c := mustCharacterize(t, []float64{1, 2, 3, 4, 5, 6, 7, 8})
	pool := []Characterization{c, c, c}
	neighbors := Nearest(pool, 0, 2)
	if len(neighbors) != 2 || neighbors[0].Index != 1 || neighbors[1].Index != 2 {
		t.Fatalf("tie order = %+v, want indices 1,2", neighbors)
	}
	if Nearest(pool, 0, 0) != nil {
		t.Error("k<=0 must return nil")
	}
	if Nearest(pool, -1, 2) != nil || Nearest(pool, 9, 2) != nil {
		t.Error("out-of-range targetIndex must return nil")
	}
}
