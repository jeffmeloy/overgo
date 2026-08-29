//go:build windows

package devicemath

import (
	"fmt"
	"math"

	"overgo/internal/cuda/device"
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

// WindowedCausalAttentionBackwardDevice computes dQ/dK/dV of sliding-window
// causal multi-head (GQA) attention on the device -- the VJP counterpart of
// hostmath.WindowedCausalAttention and the device analogue of
// hostmath.WindowedCausalAttentionBackward. It materializes the windowed causal
// softmax band (window<=0 => full causal prefix, bit-identical to the pre-existing
// full-causal device path) and runs the verified device MultiHeadAttentionBackward.
// The window enters only through the zeroed p band, so no attention kernel changes
// and masked keys contribute no gradient. Score scale is 1 (caller folds
// 1/sqrt(headDim) into q, matching the forward). Q is [seq, heads*headDim]; K/V
// are [seq, kvHeads*headDim]; dOut is [seq, heads*headDim].
func WindowedCausalAttentionBackwardDevice(worker *device.Worker, q, k, v, dOut []float32, seq, heads, kvHeads, headDim, window int) (dQ, dK, dV []float32, err error) {
	if seq <= 0 || heads <= 0 || kvHeads <= 0 || headDim <= 0 || heads%kvHeads != 0 ||
		len(q) != seq*heads*headDim || len(k) != seq*kvHeads*headDim ||
		len(v) != seq*kvHeads*headDim || len(dOut) != seq*heads*headDim {
		return nil, nil, nil, fmt.Errorf("WindowedCausalAttentionBackwardDevice: shape mismatch (seq=%d heads=%d kvHeads=%d headDim=%d)", seq, heads, kvHeads, headDim)
	}
	p := windowedCausalSoftmaxGQA(q, k, seq, heads, kvHeads, headDim, window)
	return MultiHeadAttentionBackward(worker, q, k, v, p, dOut, seq, heads, kvHeads, headDim, 1.0)
}
