package tensorstats

import (
	"math"
	"sort"
)

// ShapeFeatures is the scale-free portion of a characterization: bounded shape
// and concentration descriptors comparable across tensors regardless of
// magnitude. Location and scale (median, L-scale, max-magnitude) are excluded so
// similarity reflects distribution shape, not raw size. Every component is
// already bounded, so the feature space needs no normalization.
func ShapeFeatures(c Characterization) [7]float64 {
	return [7]float64{
		c.LMoments.Tau3,                  // L-skewness   [-1, 1]
		c.LMoments.Tau4,                  // L-kurtosis   [-1, 1]
		c.Values.ZeroFraction,            // sparsity     [0, 1]
		c.Values.NormalizedL1L2,          // spread       [0, 1]
		c.Values.MaxEnergyFraction,       // concentration[0, 1]
		c.Values.NormalizedEnergyEntropy, // dispersion   [0, 1]
		c.Values.MeanRMSRatio,            // sign balance [-1, 1]
	}
}

// FeatureDistance is the Euclidean distance between two characterizations in
// scale-free shape-feature space. Because every feature is bounded, the distance
// is bounded and directly comparable without normalization.
func FeatureDistance(a, b Characterization) float64 {
	fa, fb := ShapeFeatures(a), ShapeFeatures(b)
	var sum float64
	for i := range fa {
		d := fa[i] - fb[i]
		sum += d * d
	}
	return math.Sqrt(sum)
}

// Neighbor pairs a pool index with its shape-feature distance to a target.
type Neighbor struct {
	Index    int     `json:"index"`
	Distance float64 `json:"distance"`
}

// Nearest returns up to k pool entries closest to target in shape-feature space,
// ascending by distance, skipping the entry at skipIndex (pass -1 to keep all).
// Ties break by ascending index so the ordering is deterministic. This is an
// exact O(n) scan: at within-model scale (hundreds of tensors) no approximate
// index is warranted.
func Nearest(target Characterization, pool []Characterization, k, skipIndex int) []Neighbor {
	if k <= 0 {
		return nil
	}
	neighbors := make([]Neighbor, 0, len(pool))
	for i := range pool {
		if i == skipIndex {
			continue
		}
		neighbors = append(neighbors, Neighbor{Index: i, Distance: FeatureDistance(target, pool[i])})
	}
	sort.Slice(neighbors, func(i, j int) bool {
		if neighbors[i].Distance != neighbors[j].Distance {
			return neighbors[i].Distance < neighbors[j].Distance
		}
		return neighbors[i].Index < neighbors[j].Index
	})
	if len(neighbors) > k {
		neighbors = neighbors[:k]
	}
	return neighbors
}
