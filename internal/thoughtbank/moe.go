package thoughtbank

import (
	"fmt"
	"math"
	"sort"

	"overgo/internal/hostmath"
)

// DeepSeek-style shared-routed MoE feed-forward: a router, top-k routed experts,
// and shared experts that are always active. Ported from adaptive's
// go/extmodel/hyper_connection_block.go + moe_routing.go at b8fef3cc2.

// sharedRoutedMoEWeights is one MoE layer.
type sharedRoutedMoEWeights struct {
	DModel, DFF int
	NExperts    int
	NShared     int
	TopK        int
	WGate       []float32   // [n_experts, d_model]
	ExpertW12   [][]float32 // n_experts x [2*d_ff, d_model]
	ExpertW3    [][]float32 // n_experts x [d_model, d_ff]
	SharedW12   [][]float32 // n_shared x [2*d_ff, d_model]
	SharedW3    [][]float32 // n_shared x [d_model, d_ff]
}

// swiGLUClamped is the allocating clamped-SwiGLU over the hostmath owner.
func swiGLUClamped(fused []float32, rows, hidden int) []float32 {
	out := make([]float32, rows*hidden)
	hostmath.SwiGLUClampedInto(out, fused, rows, hidden)
	return out
}

func swiGLUExpert(x, w12, w3 []float32, rows, d, dff int) []float32 {
	// Both expert projections are independent per output unit, so each splits by
	// row over the flat (row, unit) output space.
	fused := make([]float32, rows*2*dff)
	wide := 2 * dff
	hostmath.ParallelRangeF64(rows*wide, d, func(lo, hi int) {
		for i := lo; i < hi; i++ {
			t, o := i/wide, i%wide
			fused[i] = float32(dot(w12[o*d:(o+1)*d], x[t*d:(t+1)*d]))
		}
	})

	hidden := swiGLUClamped(fused, rows, dff)
	out := make([]float32, rows*d)
	hostmath.ParallelRangeF64(rows*d, dff, func(lo, hi int) {
		for i := lo; i < hi; i++ {
			t, o := i/d, i%d
			out[i] = float32(dot(w3[o*dff:(o+1)*dff], hidden[t*dff:(t+1)*dff]))
		}
	})

	return out
}

// SharedRoutedMoEForward returns the layer output and its load-balance auxiliary.
//
// Shared experts are added at FULL weight, not folded into the routed convex
// combination: they are a dense residual path every token takes, which is why
// the routed weights sum to one on their own.
func SharedRoutedMoEForward(x []float32, rows int, w *sharedRoutedMoEWeights) ([]float32, float64, error) {
	d := w.DModel
	if len(x) != rows*d {
		return nil, 0, fmt.Errorf("moe: input has %d values, want %d", len(x), rows*d)
	}
	if len(w.ExpertW12) != w.NExperts || len(w.ExpertW3) != w.NExperts {
		return nil, 0, fmt.Errorf("moe: %d expert weight pairs, want %d", len(w.ExpertW12), w.NExperts)
	}

	out := make([]float32, rows*d)
	for s := 0; s < w.NShared; s++ {
		sh := swiGLUExpert(x, w.SharedW12[s], w.SharedW3[s], rows, d, w.DFF)
		for i := range out {
			out[i] += sh[i]
		}
	}
	// More than one shared expert is averaged (reference sum(...)/n_shared); a
	// single shared expert is used as-is.
	if w.NShared > 1 {
		inv := float32(1.0 / float64(w.NShared))
		for i := range out {
			out[i] *= inv
		}
	}

	routings := RouteMoETopK(x, w.WGate, rows, d, w.NExperts, w.TopK)
	// Expert-major dispatch: each expert runs once over the tokens routed to it.
	for e := 0; e < w.NExperts; e++ {
		var idx []int
		var coef []float32
		for t, r := range routings {
			for i, ex := range r.Experts {
				if ex == e {
					idx = append(idx, t)
					coef = append(coef, r.Weights[i])
				}
			}
		}
		if len(idx) == 0 {
			continue
		}
		gathered := make([]float32, len(idx)*d)
		for i, t := range idx {
			copy(gathered[i*d:(i+1)*d], x[t*d:(t+1)*d])
		}
		ey := swiGLUExpert(gathered, w.ExpertW12[e], w.ExpertW3[e], len(idx), d, w.DFF)
		for i, t := range idx {
			c := float64(coef[i])
			for j := 0; j < d; j++ {
				out[t*d+j] += float32(c * float64(ey[i*d+j]))
			}
		}
	}
	return out, MoEBalanceLoss(routings, w.NExperts, w.TopK), nil
}

// MoERouteEpsilon guards the renormalisation denominator. Softplus is strictly
// positive so the sum cannot truly be zero; this protects against underflow when
// every selected affinity is denormal.
const MoERouteEpsilon = 1e-8

// moERouting is one token's routing decision, in descending affinity order.
type moERouting struct {
	Experts []int
	Weights []float32
}

// RouteMoETopK scores rows tokens with sqrt(softplus(h W_gate)) and keeps the
// top-k per token, renormalising WITHIN the selected set (a convex combination).
// Ties break by lower expert index, matching torch.topk over a stable sort.
func RouteMoETopK(hidden, gate []float32, rows, d, nExperts, k int) []moERouting {
	if k > nExperts {
		k = nExperts
	}
	out := make([]moERouting, rows)
	scores := make([]float64, nExperts)
	order := make([]int, nExperts)
	for r := 0; r < rows; r++ {
		h := hidden[r*d : (r+1)*d]
		for e := 0; e < nExperts; e++ {
			scores[e] = math.Sqrt(hostmath.Softplus(dot(gate[e*d:(e+1)*d], h)))
			order[e] = e
		}
		sort.SliceStable(order, func(a, b int) bool {
			return scores[order[a]] > scores[order[b]]
		})

		sel := make([]int, k)
		wts := make([]float32, k)
		var sum float64
		for i := 0; i < k; i++ {
			sel[i] = order[i]
			sum += scores[order[i]]
		}
		inv := 1.0 / (sum + MoERouteEpsilon)
		for i := 0; i < k; i++ {
			wts[i] = float32(scores[order[i]] * inv)
		}
		out[r] = moERouting{Experts: sel, Weights: wts}
	}
	return out
}

// MoEBalanceLoss is the sequence-wise load-balancing auxiliary: mean squared
// deviation of per-expert usage from uniform. Usage counts SELECTIONS, not
// weights, so a low-weight choice still counts.
func MoEBalanceLoss(routings []moERouting, nExperts, k int) float64 {
	if len(routings) == 0 || nExperts == 0 {
		return 0
	}
	usage := make([]float64, nExperts)
	for _, r := range routings {
		for _, e := range r.Experts {
			usage[e]++
		}
	}
	expected := float64(k) / float64(nExperts)
	var acc float64
	for _, u := range usage {
		dev := u/float64(len(routings)) - expected
		acc += dev * dev
	}
	return acc / float64(nExperts)
}
