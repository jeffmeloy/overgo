package tensorstats

import (
	"math"
	"sort"
)

// ShapeFeatures is the scale-free portion of a characterization: bounded shape
// and concentration descriptors comparable across tensors regardless of
// magnitude. Location and scale (median, L-scale, max-magnitude) are excluded so
// similarity reflects distribution shape, not raw size.
func ShapeFeatures(c Characterization) [7]float64 {
	return [7]float64{
		c.LMoments.Tau3,                  // L-skewness    [-1, 1]
		c.LMoments.Tau4,                  // L-kurtosis    [-1, 1]
		c.Values.ZeroFraction,            // sparsity      [0, 1]
		c.Values.NormalizedL1L2,          // spread        [0, 1]
		c.Values.MaxEnergyFraction,       // concentration [0, 1]
		c.Values.NormalizedEnergyEntropy, // dispersion    [0, 1]
		c.Values.MeanRMSRatio,            // sign balance  [-1, 1]
	}
}

// Neighbor pairs a pool index with its dissimilarity to a target.
type Neighbor struct {
	Index    int     `json:"index"`
	Distance float64 `json:"distance"`
}

// Nearest returns up to k pool entries closest in distribution shape to the
// tensor at targetIndex, ascending by dissimilarity, excluding the target
// itself. Ties break by ascending index.
//
// Dissimilarity is a normalized Spearman footrule: each feature is rank-
// transformed across the pool (its empirical-CDF position, average ranks for
// ties) and the distance is the mean absolute rank difference, bounded in
// [0, 1]. Ranks are used instead of a metric on the raw features so no arbitrary
// per-feature normalization or equal-unit assumption is imposed: each dimension
// contributes only its order within the catalog, invariant to any monotone
// rescaling of that feature. Exact O(n log n) -- at within-model scale no
// approximate index is warranted.
func Nearest(pool []Characterization, targetIndex, k int) []Neighbor {
	if k <= 0 || targetIndex < 0 || targetIndex >= len(pool) {
		return nil
	}
	ranks := featureRanks(pool)
	target := ranks[targetIndex]
	neighbors := make([]Neighbor, 0, len(pool))
	for i := range pool {
		if i == targetIndex {
			continue
		}
		neighbors = append(neighbors, Neighbor{Index: i, Distance: footrule(target, ranks[i])})
	}
	sort.Slice(neighbors, func(a, b int) bool {
		if neighbors[a].Distance != neighbors[b].Distance {
			return neighbors[a].Distance < neighbors[b].Distance
		}
		return neighbors[a].Index < neighbors[b].Index
	})
	if len(neighbors) > k {
		neighbors = neighbors[:k]
	}
	return neighbors
}

// footrule is the mean absolute per-feature rank difference, bounded in [0, 1]
// because ranks are in [0, 1].
func footrule(a, b [7]float64) float64 {
	var sum float64
	for i := range a {
		sum += math.Abs(a[i] - b[i])
	}
	return sum / float64(len(a))
}

// featureRanks rank-transforms each feature column across the pool: every tensor
// receives, per feature, its empirical-CDF position in [0, 1] using average
// ranks for ties. This is the distribution-free replacement for normalizing
// heterogeneous features -- order is compared, not magnitude.
func featureRanks(pool []Characterization) [][7]float64 {
	n := len(pool)
	out := make([][7]float64, n)
	if n <= 1 {
		return out // 0 or 1 element: ranks carry no information
	}
	feats := make([][7]float64, n)
	for i := range pool {
		feats[i] = ShapeFeatures(pool[i])
	}
	denom := float64(n - 1)
	order := make([]int, n)
	for f := 0; f < 7; f++ {
		for i := range order {
			order[i] = i
		}
		sort.Slice(order, func(a, b int) bool { return feats[order[a]][f] < feats[order[b]][f] })
		for i := 0; i < n; {
			j := i
			for j+1 < n && feats[order[j+1]][f] == feats[order[i]][f] {
				j++
			}
			midrank := float64(i+j) / 2.0 / denom // average of tied 0-based ranks, in [0,1]
			for t := i; t <= j; t++ {
				out[order[t]][f] = midrank
			}
			i = j + 1
		}
	}
	return out
}
