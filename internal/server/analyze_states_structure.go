package server

import (
	"math"
	"sort"
)

// mdsPair: an upper-triangle index pair (i<j) used by the non-metric MDS solver.
type mdsPair struct{ i, j int }

// Distribution-free structure of a set of hidden-state vectors. Every function
// here reports measured geometry or uses only the *rank order* of dissimilarities
// — no PCA, no Gaussian/linear assumption, no fitted generative model (see the
// workbench's analysis principle). Inputs are row vectors; N is bounded upstream.

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
// regression of the disparities each iteration. No metric/linear/Gaussian
// assumption: any strictly monotone transform of the input distances yields the
// same layout. Deterministic (rank-based init, fixed schedule). Returns the
// coordinates and the final Kruskal stress-1.
func nonMetricMDS(distance [][]float64, iterations int) ([][2]float64, float64) {
	n := len(distance)
	coords := make([][2]float64, n)
	if n == 0 {
		return coords, 0
	}
	if n == 1 {
		return coords, 0
	}
	if iterations < 1 {
		iterations = 200
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

	stress := 1.0
	current := make([]float64, len(pairs)) // embedding distances, pair order
	disparities := make([]float64, len(pairs))
	for iteration := 0; iteration < iterations; iteration++ {
		for index, p := range pairs {
			current[index] = euclid2(coords[p.i], coords[p.j])
		}
		// Isotonic regression: monotone-nondecreasing disparities fit to the
		// embedding distances taken in dissimilarity-rank order (PAVA).
		isotonicFit(current, disparities)
		stress = kruskalStress(current, disparities)
		guttmanUpdate(coords, pairs, disparities, current)
	}
	return coords, stress
}

// forceDirectedLayout: Fruchterman–Reingold over the kNN graph — local neighbor
// relations only, deterministic schedule. No assumption about global shape.
func forceDirectedLayout(adjacency [][]int, iterations int) [][2]float64 {
	n := len(adjacency)
	coords := make([][2]float64, n)
	if n == 0 {
		return coords
	}
	if iterations < 1 {
		iterations = 300
	}
	// Deterministic circular init.
	for i := 0; i < n; i++ {
		angle := 2 * math.Pi * float64(i) / float64(n)
		coords[i] = [2]float64{math.Cos(angle), math.Sin(angle)}
	}
	area := 1.0
	kSpring := math.Sqrt(area / float64(n))
	temperature := 0.1
	edges := undirectedEdges(adjacency)
	for iteration := 0; iteration < iterations; iteration++ {
		displacement := make([][2]float64, n)
		// Repulsion between all pairs.
		for i := 0; i < n; i++ {
			for j := i + 1; j < n; j++ {
				dx := coords[i][0] - coords[j][0]
				dy := coords[i][1] - coords[j][1]
				dist := math.Hypot(dx, dy)
				if dist < 1e-9 {
					dx, dy, dist = 1e-4, 1e-4, 1e-4
				}
				force := kSpring * kSpring / dist
				ux, uy := dx/dist, dy/dist
				displacement[i][0] += ux * force
				displacement[i][1] += uy * force
				displacement[j][0] -= ux * force
				displacement[j][1] -= uy * force
			}
		}
		// Attraction along edges.
		for _, e := range edges {
			dx := coords[e[0]][0] - coords[e[1]][0]
			dy := coords[e[0]][1] - coords[e[1]][1]
			dist := math.Hypot(dx, dy)
			if dist < 1e-9 {
				continue
			}
			force := dist * dist / kSpring
			ux, uy := dx/dist, dy/dist
			displacement[e[0]][0] -= ux * force
			displacement[e[0]][1] -= uy * force
			displacement[e[1]][0] += ux * force
			displacement[e[1]][1] += uy * force
		}
		for i := 0; i < n; i++ {
			d := math.Hypot(displacement[i][0], displacement[i][1])
			if d > 1e-9 {
				limit := math.Min(d, temperature)
				coords[i][0] += displacement[i][0] / d * limit
				coords[i][1] += displacement[i][1] / d * limit
			}
		}
		temperature *= 0.99 // deterministic cooling
	}
	return coords
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
		// B contribution: off-diagonal -ratio, accumulated as Guttman transform.
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

func undirectedEdges(adjacency [][]int) [][2]int {
	seen := make(map[[2]int]struct{})
	edges := make([][2]int, 0)
	for i, neighbors := range adjacency {
		for _, j := range neighbors {
			a, b := i, j
			if a > b {
				a, b = b, a
			}
			key := [2]int{a, b}
			if _, ok := seen[key]; !ok {
				seen[key] = struct{}{}
				edges = append(edges, key)
			}
		}
	}
	return edges
}
