package tensorstats

import (
	"math"
	"sort"
)

// IsMatrix reports the spectral operator's dimensional contract.
func IsMatrix(extents []uint64) bool { return len(extents) == 2 }

// SingularValues returns the min(rows, cols) singular values of a row-major
// rows×cols matrix, descending. They are computed as the square roots of the
// eigenvalues of the smaller Gram matrix (AᵀA or AAᵀ), found by a cyclic Jacobi
// sweep. This is exact but O(k³) in the smaller dimension k, so it suits small
// and moderate matrices; production-scale weights need a randomized method.
// Returns false on invalid dimensions or a data-length mismatch.
func SingularValues(data []float64, rows, cols int) ([]float64, bool) {
	if rows <= 0 || cols <= 0 || len(data) != rows*cols {
		return nil, false
	}
	// Gram over the smaller dimension; its eigenvalues are the squared singular
	// values. AᵀA is cols×cols, AAᵀ is rows×rows — pick the cheaper.
	var k int
	var gram []float64
	if cols <= rows {
		k = cols
		gram = make([]float64, k*k)
		for i := 0; i < cols; i++ {
			for j := i; j < cols; j++ {
				var s float64
				for r := 0; r < rows; r++ {
					s += data[r*cols+i] * data[r*cols+j]
				}
				gram[i*k+j], gram[j*k+i] = s, s
			}
		}
	} else {
		k = rows
		gram = make([]float64, k*k)
		for i := 0; i < rows; i++ {
			for j := i; j < rows; j++ {
				var s float64
				for c := 0; c < cols; c++ {
					s += data[i*cols+c] * data[j*cols+c]
				}
				gram[i*k+j], gram[j*k+i] = s, s
			}
		}
	}
	eigenvalues := symmetricEigenvalues(gram, k)
	values := make([]float64, k)
	for i, lambda := range eigenvalues {
		values[i] = math.Sqrt(math.Max(0, lambda))
	}
	sort.Sort(sort.Reverse(sort.Float64Slice(values)))
	return values, true
}

// EffectiveRankOf is the normalized effective rank of a row-major rows×cols
// matrix: its singular values passed through EffectiveRank. Returns false when
// the matrix is invalid or has no positive singular value.
func EffectiveRankOf(data []float64, rows, cols int) (float64, bool) {
	values, ok := SingularValues(data, rows, cols)
	if !ok {
		return 0, false
	}
	return EffectiveRank(values)
}

// symmetricEigenvalues returns the eigenvalues of a k×k row-major symmetric
// matrix via a cyclic Jacobi sweep. The matrix is treated as read-only (a copy
// is rotated). Eigenvalues only — the rotations are not accumulated.
func symmetricEigenvalues(symmetric []float64, k int) []float64 {
	if k <= 0 {
		return nil
	}
	s := append([]float64(nil), symmetric...)
	if k == 1 {
		return []float64{s[0]}
	}
	const maxSweeps = 80
	for sweep := 0; sweep < maxSweeps; sweep++ {
		var off float64
		for p := 0; p < k; p++ {
			for q := p + 1; q < k; q++ {
				off += s[p*k+q] * s[p*k+q]
			}
		}
		if off <= 1e-30 {
			break
		}
		for p := 0; p < k; p++ {
			for q := p + 1; q < k; q++ {
				apq := s[p*k+q]
				if apq == 0 {
					continue
				}
				tau := (s[q*k+q] - s[p*k+p]) / (2 * apq)
				t := math.Copysign(1, tau) / (math.Abs(tau) + math.Sqrt(1+tau*tau))
				c := 1 / math.Sqrt(1+t*t)
				sn := t * c
				// Left rotation: rows p, q.
				for j := 0; j < k; j++ {
					rp, rq := s[p*k+j], s[q*k+j]
					s[p*k+j] = c*rp - sn*rq
					s[q*k+j] = sn*rp + c*rq
				}
				// Right rotation: columns p, q.
				for i := 0; i < k; i++ {
					cp, cq := s[i*k+p], s[i*k+q]
					s[i*k+p] = c*cp - sn*cq
					s[i*k+q] = sn*cp + c*cq
				}
			}
		}
	}
	eigenvalues := make([]float64, k)
	for i := 0; i < k; i++ {
		eigenvalues[i] = s[i*k+i]
	}
	return eigenvalues
}

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

// ValidEffectiveRank reports the normalized spectral interval.
func ValidEffectiveRank(value float64) bool {
	return value > 0 && value <= 1 && !math.IsNaN(value) && !math.IsInf(value, 0)
}
