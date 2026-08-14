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
	base := mustCharacterize(t, []float64{1, 2, 3, 4, 5, 6, 7, 8})
	scaled := mustCharacterize(t, []float64{10, 20, 30, 40, 50, 60, 70, 80})
	if d := FeatureDistance(base, scaled); d > 1e-9 {
		t.Errorf("positive scaling changed shape features: distance = %.3g, want ~0", d)
	}
}

func TestFeatureDistanceSeparatesShapes(t *testing.T) {
	symmetric := mustCharacterize(t, []float64{-4, -3, -2, -1, 1, 2, 3, 4})
	skewed := mustCharacterize(t, []float64{1, 1, 1, 1, 1, 1, 2, 40})
	self := FeatureDistance(symmetric, symmetric)
	if self != 0 {
		t.Errorf("self distance = %.3g, want 0", self)
	}
	if FeatureDistance(symmetric, skewed) <= self {
		t.Error("a differently-shaped tensor must be farther than self")
	}
}

func TestNearestOrdersByDistanceAndSkipsSelf(t *testing.T) {
	pool := []Characterization{
		mustCharacterize(t, []float64{1, 2, 3, 4, 5, 6, 7, 8}),    // 0: target-like
		mustCharacterize(t, []float64{2, 4, 6, 8, 10, 12, 14, 16}), // 1: same shape, scaled -> nearest
		mustCharacterize(t, []float64{1, 1, 1, 1, 1, 1, 1, 50}),    // 2: heavy right tail -> far
		mustCharacterize(t, []float64{0, 0, 0, 3, 0, 0, 6, 0}),     // 3: sparse -> far
	}
	neighbors := Nearest(pool[0], pool, 2, 0)
	if len(neighbors) != 2 {
		t.Fatalf("got %d neighbors, want 2", len(neighbors))
	}
	if neighbors[0].Index != 1 {
		t.Errorf("nearest = index %d (dist %.4f), want index 1 (same shape)", neighbors[0].Index, neighbors[0].Distance)
	}
	for _, n := range neighbors {
		if n.Index == 0 {
			t.Error("Nearest must skip the target index 0")
		}
	}
	if neighbors[0].Distance > neighbors[1].Distance {
		t.Error("neighbors must be ascending by distance")
	}
}

func TestNearestDeterministicTieBreak(t *testing.T) {
	// Two identical-shape candidates tie; the lower index must come first.
	c := mustCharacterize(t, []float64{1, 2, 3, 4, 5, 6, 7, 8})
	pool := []Characterization{c, c, c}
	neighbors := Nearest(pool[0], pool, 2, 0)
	if neighbors[0].Index != 1 || neighbors[1].Index != 2 {
		t.Errorf("tie order = %d,%d, want 1,2", neighbors[0].Index, neighbors[1].Index)
	}
	if math.Abs(neighbors[0].Distance) > 1e-9 {
		t.Errorf("identical shapes must have zero distance, got %.3g", neighbors[0].Distance)
	}
}

func TestNearestNonPositiveKReturnsNil(t *testing.T) {
	c := mustCharacterize(t, []float64{1, 2, 3, 4})
	if Nearest(c, []Characterization{c}, 0, -1) != nil {
		t.Error("k<=0 must return nil")
	}
}
