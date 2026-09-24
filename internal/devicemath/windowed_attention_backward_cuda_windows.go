//go:build windows

package devicemath

import (
	"math"
)

// windowedCausalSoftmaxGQA builds the per-head softmax weights p[nh*seq*seq] the
// device MHA backward consumes, restricted to the sliding-window causal band:
// query qi attends keys [lo, qi] with lo=max(0, qi+1-window); window<=0 is the
// full causal prefix. Scoring is scale=1 in f64 (caller folds 1/sqrt(hd) into
// q), GQA kv=h/group, matching hostmath.WindowedCausalAttention(Backward).
// Entries outside the band stay zero, so the dense device backward masks
// gradients exactly (masked p => zero dV/dK/dQ contribution).
func windowedCausalSoftmaxGQA(q, k []float32, seq, nh, nkv, hd, window int) []float32 {
	group := nh / nkv
	p := make([]float32, nh*seq*seq)
	row := make([]float64, seq)
	for h := range nh {
		kv := h / group
		for qi := range seq {
			lo := 0
			if window > 0 && qi+1 > window {
				lo = qi + 1 - window
			}
			nk := qi + 1 - lo
			mx := math.Inf(-1)
			for m := range nk {
				var dot float64
				for x := range hd {
					dot += float64(q[(qi*nh+h)*hd+x]) * float64(k[((lo+m)*nkv+kv)*hd+x])
				}
				row[m] = dot
				mx = max(mx, dot)
			}
			var sum float64
			for m := range nk {
				row[m] = math.Exp(row[m] - mx)
				sum += row[m]
			}
			for m := range nk {
				p[h*seq*seq+qi*seq+(lo+m)] = float32(row[m] / sum)
			}
		}
	}
	return p
}
