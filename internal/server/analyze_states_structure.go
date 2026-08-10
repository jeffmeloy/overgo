package server

import (
	"math"
	"sort"
)

// mdsPair: an upper-triangle index pair (i<j) used by the non-metric MDS solver.
type mdsPair struct{ i, j int }

// Distribution-free structure of a set of hidden-state vectors. The dissimilarity
// between vectors is an *explicit, caller-chosen* geometric assumption — never one
// assumed silently — and the downstream layout consumes only the rank order of
// those dissimilarities (no PCA, no metric embedding of the inputs, no Gaussian
// or linear assumption). Inputs are row vectors; N is bounded upstream.

// dissimilarityMetric: how two hidden-state vectors are compared. Ordered from
// fewest to most assumptions; the default is the assumption-light one.
//
//	spearman  — 1 - Spearman rank correlation. Invariant to any strictly-monotone
//	            transform of a vector's coordinates, so it assumes neither a
//	            per-dimension scale nor a Euclidean/inner-product geometry. Default.
//	cosine    — 1 - cosine similarity. Scale-invariant, but assumes a linear
//	            inner-product geometry.
//	euclidean — L2 distance. Full metric; scale-sensitive and dominated by
//	            high-variance ("massive activation") dimensions.
type dissimilarityMetric string

const (
	metricSpearman  dissimilarityMetric = "spearman"
	metricCosine    dissimilarityMetric = "cosine"
	metricEuclidean dissimilarityMetric = "euclidean"
)

func validMetric(metric dissimilarityMetric) bool {
	switch metric {
	case metricSpearman, metricCosine, metricEuclidean:
		return true
	default:
		return false
	}
}

// dissimilarityMatrix: N×N dissimilarities under the chosen metric.
func dissimilarityMatrix(vectors [][]float32, metric dissimilarityMetric) [][]float64 {
	switch metric {
	case metricEuclidean:
		return euclideanDistanceMatrix(vectors)
	case metricCosine:
		return cosineDistanceMatrix(vectors)
	default:
		return spearmanDistanceMatrix(vectors)
	}
}

// spearmanDistanceMatrix: 1 - Spearman rank correlation. Coordinates are
// rank-transformed per vector (fractional ranks for ties), then compared by
// Pearson correlation of the rank vectors. Because only the coordinate *ordering*
// is used, any monotone per-coordinate transform leaves every distance unchanged.
//
// Cost: O(N·D·logD) for the per-vector rank sorts plus O(N²·D) for the pairwise
// correlations. N is bounded by analyzeStatesMaxPositions, so this is cheap; if
// that cap is raised the rank sort's D·logD term is what grows per vector.
func spearmanDistanceMatrix(vectors [][]float32) [][]float64 {
	n := len(vectors)
	ranks := make([][]float64, n)
	for i, v := range vectors {
		ranks[i] = fractionalRanks(v)
	}
	distance := newSquare(n)
	for i := 0; i < n; i++ {
		for j := i + 1; j < n; j++ {
			value := 1 - pearson(ranks[i], ranks[j])
			distance[i][j] = value
			distance[j][i] = value
		}
	}
	return distance
}

// cosineDistanceMatrix: N×N matrix of 1 - cosine similarity (in [0,2]). Exact.
func cosineDistanceMatrix(vectors [][]float32) [][]float64 {
	n := len(vectors)
	norms := make([]float64, n)
	for i, v := range vectors {
		var sum float64
		for _, x := range v {
			sum += float64(x) * float64(x)
		}
		norms[i] = math.Sqrt(sum)
	}
	distance := newSquare(n)
	for i := 0; i < n; i++ {
		for j := i + 1; j < n; j++ {
			var dot float64
			for d := range vectors[i] {
				dot += float64(vectors[i][d]) * float64(vectors[j][d])
			}
			similarity := 0.0
			if norms[i] > 0 && norms[j] > 0 {
				similarity = dot / (norms[i] * norms[j])
			}
			value := 1 - similarity
			distance[i][j] = value
			distance[j][i] = value
		}
	}
	return distance
}

// euclideanDistanceMatrix: N×N matrix of L2 distances. Exact.
func euclideanDistanceMatrix(vectors [][]float32) [][]float64 {
	n := len(vectors)
	distance := newSquare(n)
	for i := 0; i < n; i++ {
		for j := i + 1; j < n; j++ {
			var sum float64
			for d := range vectors[i] {
				delta := float64(vectors[i][d]) - float64(vectors[j][d])
				sum += delta * delta
			}
			value := math.Sqrt(sum)
			distance[i][j] = value
			distance[j][i] = value
		}
	}
	return distance
}

// defaultNeighborCount: a documented heuristic for the kNN graph when the caller
// does not specify k — round(log2 N), clamped to [1, N-1]. It is only a default;
// k is an explicit, definitional parameter of the graph, surfaced to the caller.
func defaultNeighborCount(n int) int {
	if n < 3 {
		return 1
	}
	k := int(math.Round(math.Log2(float64(n))))
	if k < 1 {
		k = 1
	}
	if k > n-1 {
		k = n - 1
	}
	return k
}

// kNNAdjacency: for each row, the indices of its k nearest neighbors (self
// excluded), ascending by distance. Ties break by index for determinism.
func kNNAdjacency(distance [][]float64, k int) [][]int {
	n := len(distance)
	if k < 1 {
		k = 1
	}
	if k > n-1 {
		k = n - 1
	}
	adjacency := make([][]int, n)
	for i := 0; i < n; i++ {
		order := make([]int, 0, n-1)
		for j := 0; j < n; j++ {
			if j != i {
				order = append(order, j)
			}
		}
		sort.SliceStable(order, func(a, b int) bool {
			da, db := distance[i][order[a]], distance[i][order[b]]
			if da == db {
				return order[a] < order[b]
			}
			return da < db
		})
		if k < len(order) {
			order = order[:k]
		}
		adjacency[i] = order
	}
	return adjacency
}

// nonMetricMDS: a 2D layout that uses ONLY the rank order of the pairwise
// dissimilarities — Kruskal non-metric MDS by SMACOF with isotonic (monotone)
// regression of the disparities each iteration. Any strictly monotone transform
// of the input dissimilarities yields the same layout. Iterates to convergence
// (relative stress improvement below `tolerance`) rather than a fixed count, with
// `maxIterations` only as a safety bound. Returns the coordinates, the final
// Kruskal stress-1, and the number of iterations actually run.
func nonMetricMDS(distance [][]float64, maxIterations int, tolerance float64) ([][2]float64, float64, int) {
	n := len(distance)
	coords := make([][2]float64, n)
	if n < 2 {
		return coords, 0, 0
	}

	// Upper-triangle pairs, ordered by the *rank* of their dissimilarity. Only
	// this ordering is consumed downstream — magnitudes never are.
	pairs := make([]mdsPair, 0, n*(n-1)/2)
	for i := 0; i < n; i++ {
		for j := i + 1; j < n; j++ {
			pairs = append(pairs, mdsPair{i, j})
		}
	}
	sort.SliceStable(pairs, func(a, b int) bool {
		return distance[pairs[a].i][pairs[a].j] < distance[pairs[b].i][pairs[b].j]
	})

	// Rank-based deterministic init: coordinates from each point's rank of
	// dissimilarity to the two ends of the farthest pair (ordinal, not metric).
	initRankLayout(distance, coords)

	stress := math.Inf(1)
	iterationsRun := 0
	current := make([]float64, len(pairs)) // embedding distances, pair order
	disparities := make([]float64, len(pairs))
	for iteration := 0; iteration < maxIterations; iteration++ {
		for index, p := range pairs {
			current[index] = euclid2(coords[p.i], coords[p.j])
		}
		// Isotonic regression: monotone-nondecreasing disparities fit to the
		// embedding distances taken in dissimilarity-rank order (PAVA).
		isotonicFit(current, disparities)
		newStress := kruskalStress(current, disparities)
		iterationsRun = iteration + 1
		if iteration > 0 && stress-newStress <= tolerance*stress {
			stress = newStress
			break
		}
		stress = newStress
		guttmanUpdate(coords, pairs, disparities, current)
	}
	return coords, stress, iterationsRun
}

// ---- helpers ----

func newSquare(n int) [][]float64 {
	m := make([][]float64, n)
	for i := range m {
		m[i] = make([]float64, n)
	}
	return m
}

func euclid2(a, b [2]float64) float64 {
	return math.Hypot(a[0]-b[0], a[1]-b[1])
}

// fractionalRanks: ascending ranks of a vector's coordinates, ties assigned their
// average rank. Only the ordering of coordinates is captured — no magnitudes.
func fractionalRanks(v []float32) []float64 {
	d := len(v)
	order := make([]int, d)
	for i := range order {
		order[i] = i
	}
	sort.SliceStable(order, func(a, b int) bool { return v[order[a]] < v[order[b]] })
	ranks := make([]float64, d)
	i := 0
	for i < d {
		j := i
		for j+1 < d && v[order[j+1]] == v[order[i]] {
			j++
		}
		average := float64(i+j) / 2.0
		for k := i; k <= j; k++ {
			ranks[order[k]] = average
		}
		i = j + 1
	}
	return ranks
}

// pearson: Pearson correlation; 0 when either input has no variance.
func pearson(a, b []float64) float64 {
	n := len(a)
	if n == 0 {
		return 0
	}
	var meanA, meanB float64
	for i := range a {
		meanA += a[i]
		meanB += b[i]
	}
	meanA /= float64(n)
	meanB /= float64(n)
	var numerator, varA, varB float64
	for i := range a {
		x := a[i] - meanA
		y := b[i] - meanB
		numerator += x * y
		varA += x * x
		varB += y * y
	}
	if varA == 0 || varB == 0 {
		return 0
	}
	return numerator / math.Sqrt(varA*varB)
}

// initRankLayout: (x,y) = normalized rank of dissimilarity to the two ends of
// the farthest pair. Ordinal only — no metric embedding.
func initRankLayout(distance [][]float64, coords [][2]float64) {
	n := len(distance)
	anchorA, anchorB := 0, 0
	best := math.Inf(-1)
	for i := 0; i < n; i++ {
		for j := i + 1; j < n; j++ {
			if distance[i][j] > best {
				best, anchorA, anchorB = distance[i][j], i, j
			}
		}
	}
	rankX := rankOf(distanceColumn(distance, anchorA))
	rankY := rankOf(distanceColumn(distance, anchorB))
	for i := 0; i < n; i++ {
		coords[i] = [2]float64{
			2*float64(rankX[i])/float64(n-1) - 1,
			2*float64(rankY[i])/float64(n-1) - 1,
		}
	}
}

func distanceColumn(distance [][]float64, column int) []float64 {
	values := make([]float64, len(distance))
	for i := range distance {
		values[i] = distance[i][column]
	}
	return values
}

// rankOf: ascending rank (0-based) of each value; ties by index for determinism.
func rankOf(values []float64) []int {
	order := make([]int, len(values))
	for i := range order {
		order[i] = i
	}
	sort.SliceStable(order, func(a, b int) bool {
		if values[order[a]] == values[order[b]] {
			return order[a] < order[b]
		}
		return values[order[a]] < values[order[b]]
	})
	rank := make([]int, len(values))
	for position, index := range order {
		rank[index] = position
	}
	return rank
}

// isotonicFit: pool-adjacent-violators, fitting a monotone-nondecreasing
// sequence `fitted` to `values` (values already in dissimilarity-rank order).
func isotonicFit(values, fitted []float64) {
	n := len(values)
	if n == 0 {
		return
	}
	level := make([]float64, 0, n)
	weight := make([]float64, 0, n)
	count := make([]int, 0, n)
	for _, v := range values {
		level = append(level, v)
		weight = append(weight, 1)
		count = append(count, 1)
		for len(level) > 1 && level[len(level)-2] > level[len(level)-1] {
			w := weight[len(level)-2] + weight[len(level)-1]
			mean := (level[len(level)-2]*weight[len(level)-2] + level[len(level)-1]*weight[len(level)-1]) / w
			c := count[len(level)-2] + count[len(level)-1]
			level = level[:len(level)-1]
			weight = weight[:len(weight)-1]
			count = count[:len(count)-1]
			level[len(level)-1] = mean
			weight[len(weight)-1] = w
			count[len(count)-1] = c
		}
	}
	index := 0
	for block := range level {
		for c := 0; c < count[block]; c++ {
			fitted[index] = level[block]
			index++
		}
	}
}

func kruskalStress(distances, disparities []float64) float64 {
	var numerator, denominator float64
	for i := range distances {
		diff := distances[i] - disparities[i]
		numerator += diff * diff
		denominator += distances[i] * distances[i]
	}
	if denominator == 0 {
		return 0
	}
	return math.Sqrt(numerator / denominator)
}

// guttmanUpdate: one SMACOF majorization step toward the disparities.
func guttmanUpdate(coords [][2]float64, pairs []mdsPair, disparities, current []float64) {
	n := len(coords)
	next := make([][2]float64, n)
	for index, p := range pairs {
		d := current[index]
		ratio := 0.0
		if d > 1e-12 {
			ratio = disparities[index] / d
		}
		bx := ratio * (coords[p.i][0] - coords[p.j][0])
		by := ratio * (coords[p.i][1] - coords[p.j][1])
		next[p.i][0] += coords[p.j][0] + bx
		next[p.i][1] += coords[p.j][1] + by
		next[p.j][0] += coords[p.i][0] - bx
		next[p.j][1] += coords[p.i][1] - by
	}
	for i := 0; i < n; i++ {
		coords[i][0] = next[i][0] / float64(n)
		coords[i][1] = next[i][1] / float64(n)
	}
}
