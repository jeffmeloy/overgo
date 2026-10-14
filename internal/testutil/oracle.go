package testutil

import "math"

// ClampNormalizedF32InPlace bounds every sample to the signed unit interval [-1, 1] in place.
//   - interval = media.SignedUnitBounds32
//   - the bounds are written out because tests of media and of its dependencies import testutil, so testutil importing media is a cycle.
func ClampNormalizedF32InPlace(values []float32) {
	for index, value := range values {
		values[index] = max(-1, min(value, 1))
	}
}

// WindowedCausalAttentionBackward writes the VJP of hostmath.WindowedCausalAttention into dq, dk and dv.
//   - window <= 0 => the full causal span, and the VJP equals hostmath.CausalAttentionBackward
//   - score scale = 1 because the caller folds 1/sqrt(headDim) into q, as the forward does
//   - dq, dk and dv are cleared, then filled
//   - the query heads of one GQA group accumulate into their shared key and value rows.
func WindowedCausalAttentionBackward(dq, dk, dv, q, k, v, dOut []float32, seq, heads, kvHeads, headDim, window int) {
	clear(dq)
	clear(dk)
	clear(dv)
	probs := make([]float64, seq)
	dP := make([]float64, seq)
	group := heads / kvHeads
	for h := range heads {
		kv := h / group
		for qi := range seq {
			lo := 0
			if window > 0 && qi+1 > window {
				lo = qi + 1 - window
			}
			nk := qi + 1 - lo
			qRow := q[(qi*heads+h)*headDim : (qi*heads+h+1)*headDim]
			dout := dOut[(qi*heads+h)*headDim : (qi*heads+h+1)*headDim]
			dqRow := dq[(qi*heads+h)*headDim : (qi*heads+h+1)*headDim]
			// recompute softmax probs over keys lo..qi
			mx := math.Inf(-1)
			for m := range nk {
				kRow := k[((lo+m)*kvHeads+kv)*headDim : ((lo+m)*kvHeads+kv+1)*headDim]
				var dot float64
				for x := range headDim {
					dot += float64(qRow[x]) * float64(kRow[x])
				}
				probs[m] = dot
				mx = max(mx, dot)
			}
			var sum float64
			for m := range nk {
				probs[m] = math.Exp(probs[m] - mx)
				sum += probs[m]
			}
			inv := 1 / sum
			var dot float64
			for m := range nk {
				probs[m] *= inv
				ki := lo + m
				vRow := v[(ki*kvHeads+kv)*headDim : (ki*kvHeads+kv+1)*headDim]
				dvRow := dv[(ki*kvHeads+kv)*headDim : (ki*kvHeads+kv+1)*headDim]
				var dpm float64
				for x := range headDim {
					dpm += float64(dout[x]) * float64(vRow[x])
					dvRow[x] += float32(probs[m]) * dout[x]
				}
				dP[m] = dpm
				dot += probs[m] * dpm
			}
			for m := range nk {
				g := probs[m] * (dP[m] - dot)
				ki := lo + m
				kRow := k[(ki*kvHeads+kv)*headDim : (ki*kvHeads+kv+1)*headDim]
				dkRow := dk[(ki*kvHeads+kv)*headDim : (ki*kvHeads+kv+1)*headDim]
				for x := range headDim {
					dqRow[x] += float32(g * float64(kRow[x]))
					dkRow[x] += float32(g * float64(qRow[x]))
				}
			}
		}
	}
}
