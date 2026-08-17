package hostmath

import (
	"math"
)

// Sinkhorn-Knopp projection onto the Birkhoff polytope (doubly stochastic
// matrices). Alternating row/column normalisation of a positive matrix
// converges to the unique doubly stochastic matrix in its diagonal-equivalence
// class (Sinkhorn & Knopp 1967), which bounds the spectral norm at 1 -- the
// property that makes it usable as a residual mixing matrix in a deep stack
// without the norm compounding across layers.
//
// Neutral home on purpose: this is a matrix operation, not a model component.
// The first consumer generates the logits from a hyper-connection layer, but
// nothing here knows that. Ported from adaptive's verified go/matrix/sinkhorn.

// SinkhornEpsilon guards the row/column sums against division by zero. It is
// the additive floor in the denominator, not a convergence tolerance: the
// iteration count is fixed by the caller, so this only has to keep an all-zero
// row from producing NaN rather than a finite (if uninformative) result.
const SinkhornEpsilon = 1e-8

// sinkhorn2x2FromLogits writes the EXACT doubly-stochastic projection of a 2x2
// positive matrix, in closed form, with no iteration.
//
// Sinkhorn is diagonal rescaling, and the cross-ratio ad/(bc) is invariant under
// it. Every doubly-stochastic 2x2 is [[p,1-p],[1-p,p]], whose cross-ratio is
// (p/(1-p))^2. Setting those equal gives p = sqrt(r)/(1+sqrt(r)) -- so the fixed
// point is determined, and iterating toward it is solving a problem that has an
// answer. In log space p collapses to a logistic:
//
//	p = sigmoid((l00 + l11 - l01 - l10) / 2)
//
// NOT wired to the shipped path: it is exact where the loop is approximate, and
// checkpoint parity is against the reference's approximate (20-sweep, +eps
// guarded) result, so it cannot be swapped in silently (ADV-126).
func sinkhorn2x2FromLogits(mat []float32) {
	p := 1 / (1 + math.Exp(-0.5*(float64(mat[0])+float64(mat[3])-float64(mat[1])-float64(mat[2]))))
	q := float32(1 - p)
	mat[0], mat[1], mat[2], mat[3] = float32(p), q, q, float32(p)
}

// SinkhornFromLogitsInPlace projects logits in caller-owned storage.
func SinkhornFromLogitsInPlace(m []float32, count, n, iters int) {
	stride := n * n
	rowSum := make([]float64, n)
	colSum := make([]float64, n)
	for b := 0; b < count; b++ {
		mat := m[b*stride : (b+1)*stride]

		// exp with a per-matrix max subtraction; see the numerical note above.
		maxLogit := float64(mat[0])
		for _, v := range mat[1:] {
			if f := float64(v); f > maxLogit {
				maxLogit = f
			}
		}
		for i, v := range mat {
			mat[i] = float32(math.Exp(float64(v) - maxLogit))
		}

		for it := 0; it < iters; it++ {
			for i := 0; i < n; i++ {
				rowSum[i] = 0
			}
			for i := 0; i < n; i++ {
				base := i * n
				for j := 0; j < n; j++ {
					rowSum[i] += float64(mat[base+j])
				}
			}
			for i := 0; i < n; i++ {
				inv := 1.0 / (rowSum[i] + SinkhornEpsilon)
				base := i * n
				for j := 0; j < n; j++ {
					mat[base+j] = float32(float64(mat[base+j]) * inv)
				}
			}

			for j := 0; j < n; j++ {
				colSum[j] = 0
			}
			for i := 0; i < n; i++ {
				base := i * n
				for j := 0; j < n; j++ {
					colSum[j] += float64(mat[base+j])
				}
			}
			for j := 0; j < n; j++ {
				inv := 1.0 / (colSum[j] + SinkhornEpsilon)
				for i := 0; i < n; i++ {
					mat[i*n+j] = float32(float64(mat[i*n+j]) * inv)
				}
			}
		}
	}
}

// SinkhornFromLogitsBackward differentiates the SHIPPED projection: the
// alternating row/column normalisation as actually run, epsilon guards and all,
// not the exact fixed point. Matching the forward the checkpoint was trained
// under is the contract; a backward for the closed form would be a gradient of
// a function the shipped forward does not compute.
//
// Reverse mode through the iteration. For y_i = x_i / (S + eps) with S = sum(x):
//
//	dx_i = (dy_i - sum_k dy_k * y_k) / (S + eps)
//
// Using y_k rather than x_k/(S+eps) keeps the epsilon exact. The forward's
// max-subtraction before exp: with M_i = exp(l_i - l_m) where m is the argmax,
//
//	dl_j = dM_j * M_j - [j == m] * sum_i dM_i * M_i
//
// The second term is zero in exact arithmetic (scale invariance) but the
// epsilon guards break the invariance slightly, so it is carried.
func SinkhornFromLogitsBackward(logits, dOut []float32, count, n, iters int) []float32 {
	stride := n * n
	dLogits := make([]float32, count*stride)
	if len(logits) != count*stride || len(dOut) != count*stride {
		return dLogits
	}

	m := make([]float64, stride)
	tape := make([][]float64, 0, 2*iters)
	rowSums := make([]float64, n)
	colSums := make([]float64, n)

	for b := 0; b < count; b++ {
		lg := logits[b*stride : (b+1)*stride]
		maxLogit := float64(lg[0])
		argmax := 0
		for i, v := range lg[1:] {
			if f := float64(v); f > maxLogit {
				maxLogit, argmax = f, i+1
			}
		}
		for i, v := range lg {
			m[i] = math.Exp(float64(v) - maxLogit)
		}
		expM := make([]float64, stride)
		copy(expM, m)

		tape = tape[:0]
		for it := 0; it < iters; it++ {
			snap := make([]float64, stride)
			copy(snap, m)
			tape = append(tape, snap)
			for i := 0; i < n; i++ {
				s := 0.0
				for j := 0; j < n; j++ {
					s += m[i*n+j]
				}
				rowSums[i] = s + SinkhornEpsilon
				for j := 0; j < n; j++ {
					m[i*n+j] /= rowSums[i]
				}
			}
			snap2 := make([]float64, stride)
			copy(snap2, m)
			tape = append(tape, snap2)
			for j := 0; j < n; j++ {
				s := 0.0
				for i := 0; i < n; i++ {
					s += m[i*n+j]
				}
				colSums[j] = s + SinkhornEpsilon
				for i := 0; i < n; i++ {
					m[i*n+j] /= colSums[j]
				}
			}
		}

		g := make([]float64, stride)
		for i := range g {
			g[i] = float64(dOut[b*stride+i])
		}
		cur := make([]float64, stride)
		for it := iters - 1; it >= 0; it-- {
			// Undo the column normalisation.
			pre := tape[2*it+1]
			copy(cur, pre)
			for j := 0; j < n; j++ {
				s := 0.0
				for i := 0; i < n; i++ {
					s += cur[i*n+j]
				}
				den := s + SinkhornEpsilon
				coupled := 0.0
				for i := 0; i < n; i++ {
					coupled += g[i*n+j] * (cur[i*n+j] / den)
				}
				for i := 0; i < n; i++ {
					g[i*n+j] = (g[i*n+j] - coupled) / den
				}
			}
			// Undo the row normalisation.
			pre = tape[2*it]
			copy(cur, pre)
			for i := 0; i < n; i++ {
				s := 0.0
				for j := 0; j < n; j++ {
					s += cur[i*n+j]
				}
				den := s + SinkhornEpsilon
				coupled := 0.0
				for j := 0; j < n; j++ {
					coupled += g[i*n+j] * (cur[i*n+j] / den)
				}
				for j := 0; j < n; j++ {
					g[i*n+j] = (g[i*n+j] - coupled) / den
				}
			}
		}

		// Through exp and the max-subtraction.
		shift := 0.0
		for i := 0; i < stride; i++ {
			shift += g[i] * expM[i]
		}
		out := dLogits[b*stride : (b+1)*stride]
		for i := 0; i < stride; i++ {
			v := g[i] * expM[i]
			if i == argmax {
				v -= shift
			}
			out[i] = float32(v)
		}
	}
	return dLogits
}
