package server

import (
	"math"
	"testing"
)

func TestDistanceMatricesAreExact(t *testing.T) {
	vectors := [][]float32{{1, 0}, {0, 1}, {1, 0}}
	cos := cosineDistanceMatrix(vectors)
	// Orthogonal → cosine distance 1; identical → 0.
	if math.Abs(cos[0][1]-1) > 1e-9 || math.Abs(cos[0][2]-0) > 1e-9 {
		t.Fatalf("cosine = %v", cos)
	}
	euc := euclideanDistanceMatrix(vectors)
	if math.Abs(euc[0][1]-math.Sqrt2) > 1e-9 || euc[0][2] != 0 {
		t.Fatalf("euclidean = %v", euc)
	}
	// Symmetry + zero diagonal.
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

func TestKNNAdjacencyPicksNearest(t *testing.T) {
	// Points on a line: 0,1,2,3. Nearest neighbor of 0 is 1; of 3 is 2.
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
	// 1's two nearest are 0 and 2 (both distance 1), tie broken by index → [0,2].
	if len(adj2[1]) != 2 || adj2[1][0] != 0 || adj2[1][1] != 2 {
		t.Fatalf("k=2 adjacency[1] = %v", adj2[1])
	}
}

// The load-bearing distribution-free property: non-metric MDS depends ONLY on
// the rank order of the dissimilarities, so any strictly-monotone transform of
// the distance matrix must yield the same layout. This is what makes the method
// assumption-free (no metric/linear/Gaussian structure presumed).
func TestNonMetricMDSIsRankInvariant(t *testing.T) {
	base := squareCluster()
	transformed := newSquare(len(base))
	for i := range base {
		for j := range base {
			// Strictly monotone, strongly nonlinear transform of the distances.
			transformed[i][j] = math.Log1p(base[i][j]) + math.Pow(base[i][j], 1.5)
		}
	}
	coordsA, _ := nonMetricMDS(base, 150)
	coordsB, _ := nonMetricMDS(transformed, 150)
	for i := range coordsA {
		if math.Abs(coordsA[i][0]-coordsB[i][0]) > 1e-6 || math.Abs(coordsA[i][1]-coordsB[i][1]) > 1e-6 {
			t.Fatalf("rank-variance at %d: %v vs %v", i, coordsA[i], coordsB[i])
		}
	}
}

func TestNonMetricMDSStressDecreases(t *testing.T) {
	distance := squareCluster()
	_, finalStress := nonMetricMDS(distance, 200)
	_, earlyStress := nonMetricMDS(distance, 5)
	if !(finalStress <= earlyStress+1e-9) {
		t.Fatalf("stress did not decrease: final %v > early %v", finalStress, earlyStress)
	}
	if finalStress < 0 || finalStress > 1.0001 {
		t.Fatalf("stress out of range: %v", finalStress)
	}
}

func TestForceLayoutIsDeterministic(t *testing.T) {
	adjacency := [][]int{{1}, {0, 2}, {1}}
	a := forceDirectedLayout(adjacency, 50)
	b := forceDirectedLayout(adjacency, 50)
	for i := range a {
		if a[i] != b[i] {
			t.Fatalf("nondeterministic at %d: %v vs %v", i, a[i], b[i])
		}
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
	// PAVA preserves the total sum.
	var sumIn, sumOut float64
	for i := range values {
		sumIn += values[i]
		sumOut += fitted[i]
	}
	if math.Abs(sumIn-sumOut) > 1e-9 {
		t.Fatalf("sum changed: %v vs %v", sumIn, sumOut)
	}
}

// squareCluster: pairwise distances of 4 points at the corners of a unit square
// plus a duplicate of corner 0 — a small structure with clear neighbors.
func squareCluster() [][]float64 {
	points := [][]float32{{0, 0}, {1, 0}, {1, 1}, {0, 1}, {0, 0}}
	return euclideanDistanceMatrix(points)
}
