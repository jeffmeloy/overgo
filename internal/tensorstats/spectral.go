package tensorstats

import "math"

// EffectiveRank returns the normalized effective rank in (0, 1]: exp(H)/n, where
// H is the Shannon entropy of the singular-value distribution (pᵢ = σᵢ/Σσ) and n
// is the number of singular values. It measures spectral flatness — the intrinsic
// dimensionality of a matrix relative to its full rank. A value near 1 means the
// spectral energy is spread evenly across all singular directions (full effective
// rank); near 1/n means a single direction dominates (rank-collapsed / highly
// redundant). Unlike the value-population descriptors, this captures matrix
// structure — two tensors with identical value distributions can differ sharply
// in effective rank.
//
// It is a measured functional of the empirical spectrum: no distribution
// assumption and no threshold (it is the spectral analog of the energy entropy
// over element magnitudes). singularValues must be non-negative and finite;
// their count is the full-rank ceiling min(rows, cols). Returns false when no
// positive singular value exists (a zero matrix) or an input is invalid.
func EffectiveRank(singularValues []float64) (float64, bool) {
	n := len(singularValues)
	if n == 0 {
		return 0, false
	}
	var sum float64
	for _, s := range singularValues {
		if s < 0 || math.IsNaN(s) || math.IsInf(s, 0) {
			return 0, false
		}
		sum += s
	}
	if sum == 0 {
		return 0, false
	}
	var entropy float64
	for _, s := range singularValues {
		if s == 0 {
			continue // 0·log0 = 0
		}
		p := s / sum
		entropy -= p * math.Log(p)
	}
	return math.Exp(entropy) / float64(n), true
}
