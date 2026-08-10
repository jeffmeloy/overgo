package server

import (
	"math"
	"testing"
)

func TestDistanceMatricesAreExact(t *testing.T) {
	vectors := [][]float32{{1, 0}, {0, 1}, {1, 0}}
	cos := cosineDistanceMatrix(vectors)
	if math.Abs(cos[0][1]-1) > 1e-9 || math.Abs(cos[0][2]-0) > 1e-9 {
		t.Fatalf("cosine = %v", cos)
	}
	euc := euclideanDistanceMatrix(vectors)
	if math.Abs(euc[0][1]-math.Sqrt2) > 1e-9 || euc[0][2] != 0 {
		t.Fatalf("euclidean = %v", euc)
	}
	for i := range euc {
		if euc[i][i] != 0 {
			t.Fatalf("diagonal[%d] = %v", i, euc[i][i])
		}
		for j := range euc {
			if euc[i][j] != euc[j][i] {
				t.Fatalf("asymmetric at %d,%d", i, j)
			}
		}
	}
}

// Spearman dissimilarity depends only on each vector's coordinate ordering, so
// applying any strictly-monotone transform to the coordinates must leave every
// distance unchanged — it assumes neither a per-dimension scale nor a Euclidean
// geometry. This is the assumption-light property that motivates it as default.
func TestSpearmanIsRankInvariant(t *testing.T) {
	vectors := [][]float32{{0.1, 0.9, 0.4, 0.2}, {0.8, 0.2, 0.5, 0.1}, {0.3, 0.3, 0.7, 0.6}}
	transformed := make([][]float32, len(vectors))
	for i, v := range vectors {
		row := make([]float32, len(v))
		for d, x := range v {
			// Strictly increasing, strongly nonlinear per-coordinate transform.
			row[d] = float32(math.Expm1(float64(x)*3)) + float32(i) // per-vector offset too
		}
		transformed[i] = row
	}
	base := spearmanDistanceMatrix(vectors)
	moved := spearmanDistanceMatrix(transformed)
	for i := range base {
		for j := range base {
			if math.Abs(base[i][j]-moved[i][j]) > 1e-9 {
				t.Fatalf("spearman changed under monotone transform at %d,%d: %v vs %v", i, j, base[i][j], moved[i][j])
			}
		}
	}
	// Identical vectors → distance 0; the diagonal is always 0.
	same := spearmanDistanceMatrix([][]float32{{1, 2, 3}, {1, 2, 3}})
	if math.Abs(same[0][1]) > 1e-9 {
		t.Fatalf("identical vectors spearman distance = %v", same[0][1])
	}
}

func TestDissimilarityMetricDispatch(t *testing.T) {
	vectors := [][]float32{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}}
	if !equalMatrix(dissimilarityMatrix(vectors, metricEuclidean), euclideanDistanceMatrix(vectors)) {
		t.Fatal("euclidean dispatch mismatch")
	}
	if !equalMatrix(dissimilarityMatrix(vectors, metricCosine), cosineDistanceMatrix(vectors)) {
		t.Fatal("cosine dispatch mismatch")
	}
	if !equalMatrix(dissimilarityMatrix(vectors, metricSpearman), spearmanDistanceMatrix(vectors)) {
		t.Fatal("spearman dispatch mismatch")
	}
	// Unknown metric falls back to the assumption-light default (spearman).
	if !equalMatrix(dissimilarityMatrix(vectors, "nonsense"), spearmanDistanceMatrix(vectors)) {
		t.Fatal("unknown metric should default to spearman")
	}
	if validMetric("nonsense") || !validMetric(metricSpearman) {
		t.Fatal("validMetric wrong")
	}
}

func TestDefaultNeighborCountIsDerived(t *testing.T) {
	// round(log2 n), clamped to [1, n-1].
	cases := map[int]int{2: 1, 3: 2, 4: 2, 8: 3, 16: 4, 64: 6}
	for n, want := range cases {
		if got := defaultNeighborCount(n); got != want {
			t.Fatalf("defaultNeighborCount(%d) = %d, want %d", n, got, want)
		}
	}
}

func TestKNNAdjacencyPicksNearest(t *testing.T) {
	distance := [][]float64{
		{0, 1, 2, 3},
		{1, 0, 1, 2},
		{2, 1, 0, 1},
		{3, 2, 1, 0},
	}
	adj := kNNAdjacency(distance, 1)
	if adj[0][0] != 1 || adj[3][0] != 2 {
		t.Fatalf("adjacency = %v", adj)
	}
	adj2 := kNNAdjacency(distance, 2)
	if len(adj2[1]) != 2 || adj2[1][0] != 0 || adj2[1][1] != 2 {
		t.Fatalf("k=2 adjacency[1] = %v", adj2[1])
	}
}

// Non-metric MDS depends ONLY on the rank order of the dissimilarities: any
// strictly-monotone transform of the distance matrix yields the same layout.
func TestNonMetricMDSIsRankInvariant(t *testing.T) {
	base := squareCluster()
	transformed := newSquare(len(base))
	for i := range base {
		for j := range base {
			transformed[i][j] = math.Log1p(base[i][j]) + math.Pow(base[i][j], 1.5)
		}
	}
	coordsA, _, _ := nonMetricMDS(base, 500, 1e-9)
	coordsB, _, _ := nonMetricMDS(transformed, 500, 1e-9)
	for i := range coordsA {
		if math.Abs(coordsA[i][0]-coordsB[i][0]) > 1e-6 || math.Abs(coordsA[i][1]-coordsB[i][1]) > 1e-6 {
			t.Fatalf("rank-variance at %d: %v vs %v", i, coordsA[i], coordsB[i])
		}
	}
}

// The solver iterates to convergence (relative stress improvement below the
// tolerance) rather than a fixed count, and stress never increases.
func TestNonMetricMDSConvergesAndDecreases(t *testing.T) {
	distance := squareCluster()
	_, finalStress, iterations := nonMetricMDS(distance, 1000, 1e-6)
	if iterations >= 1000 {
		t.Fatalf("did not converge before the safety cap: %d iters", iterations)
	}
	_, earlyStress, _ := nonMetricMDS(distance, 3, 0)
	if finalStress > earlyStress+1e-9 {
		t.Fatalf("stress increased: final %v > early %v", finalStress, earlyStress)
	}
	if finalStress < 0 || finalStress > 1.0001 {
		t.Fatalf("stress out of range: %v", finalStress)
	}
}

func TestIsotonicFitIsMonotone(t *testing.T) {
	values := []float64{3, 1, 2, 5, 4}
	fitted := make([]float64, len(values))
	isotonicFit(values, fitted)
	for i := 1; i < len(fitted); i++ {
		if fitted[i] < fitted[i-1]-1e-12 {
			t.Fatalf("not monotone: %v", fitted)
		}
	}
	var sumIn, sumOut float64
	for i := range values {
		sumIn += values[i]
		sumOut += fitted[i]
	}
	if math.Abs(sumIn-sumOut) > 1e-9 {
		t.Fatalf("sum changed: %v vs %v", sumIn, sumOut)
	}
}

func squareCluster() [][]float64 {
	points := [][]float32{{0, 0}, {1, 0}, {1, 1}, {0, 1}, {0, 0}}
	return euclideanDistanceMatrix(points)
}

func equalMatrix(a, b [][]float64) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if len(a[i]) != len(b[i]) {
			return false
		}
		for j := range a[i] {
			if math.Abs(a[i][j]-b[i][j]) > 1e-12 {
				return false
			}
		}
	}
	return true
}
