// E4B (gemma3n) training-leg VJPs: the backward operators the gemma3n arch
// needs beyond the eight already-verified hostmath VJPs. All are neutral
// (reusable by any arch): the windowed-causal attention VJP generalizes
// CausalAttentionBackward, and GELU-tanh / softcap / activation-sparsity /
// embed-input-scale are element- or row-local. Each is finite-difference
// gated in e4b_backward_fd_test.go against the exact forward it differentiates.
package hostmath

import "math"

// WindowedCausalAttentionBackward: VJP of WindowedCausalAttention. window<=0 is
// the full causal span, so this reduces exactly to CausalAttentionBackward.
// Score scale is 1 (caller folds 1/sqrt(headDim) into q, matching the forward).
// dq/dk/dv are cleared then filled; GQA groups accumulate into shared kv rows.
func WindowedCausalAttentionBackward(dq, dk, dv, q, k, v, dOut []float32, seq, heads, kvHeads, headDim, window int) {
	clear(dq)
	clear(dk)
	clear(dv)
	probs := make([]float64, seq)
	dP := make([]float64, seq)
	group := heads / kvHeads
	for h := 0; h < heads; h++ {
		kv := h / group
		for qi := 0; qi < seq; qi++ {
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
			for m := 0; m < nk; m++ {
				kRow := k[((lo+m)*kvHeads+kv)*headDim : ((lo+m)*kvHeads+kv+1)*headDim]
				var dot float64
				for x := 0; x < headDim; x++ {
					dot += float64(qRow[x]) * float64(kRow[x])
				}
				probs[m] = dot
				if dot > mx {
					mx = dot
				}
			}
			var sum float64
			for m := 0; m < nk; m++ {
				probs[m] = math.Exp(probs[m] - mx)
				sum += probs[m]
			}
			inv := 1.0 / sum
			var dot float64
			for m := 0; m < nk; m++ {
				probs[m] *= inv
				ki := lo + m
				vRow := v[(ki*kvHeads+kv)*headDim : (ki*kvHeads+kv+1)*headDim]
				dvRow := dv[(ki*kvHeads+kv)*headDim : (ki*kvHeads+kv+1)*headDim]
				var dpm float64
				for x := 0; x < headDim; x++ {
					dpm += float64(dout[x]) * float64(vRow[x])
					dvRow[x] += float32(probs[m]) * dout[x]
				}
				dP[m] = dpm
				dot += probs[m] * dpm
			}
			for m := 0; m < nk; m++ {
				g := probs[m] * (dP[m] - dot)
				ki := lo + m
				kRow := k[(ki*kvHeads+kv)*headDim : (ki*kvHeads+kv+1)*headDim]
				dkRow := dk[(ki*kvHeads+kv)*headDim : (ki*kvHeads+kv+1)*headDim]
				for x := 0; x < headDim; x++ {
					dqRow[x] += float32(g * float64(kRow[x]))
					dkRow[x] += float32(g * float64(qRow[x]))
				}
			}
		}
	}
}

// GELUTanhBackward: VJP of GELUTanh, dst = dy * GELUTanhPrime(x). dst may alias
// dy (each output depends only on its own input).
func GELUTanhBackward(dst, x, dy []float32) {
	for i := range x {
		dst[i] = float32(float64(dy[i]) * GELUTanhPrime(float64(x[i])))
	}
}

// SoftcapBackward: VJP of the logit soft-cap y = cap*tanh(z/cap) (the gemma
// final-logit and attention-logit cap). Local derivative dy/dz = 1-tanh(z/cap)^2.
// Given the PRE-cap values and the gradient w.r.t. the capped output, scales in
// place into the gradient w.r.t. the pre-cap input. dOut may equal dst.
func SoftcapBackward(dst, dOut, preCap []float32, cap float64) {
	for i := range preCap {
		t := math.Tanh(float64(preCap[i]) / cap)
		dst[i] = float32(float64(dOut[i]) * (1 - t*t))
	}
}

// ActivationSparsityBackward: VJP of SparseGateInto. dst_i = relu(g_i - cutoff)
// with cutoff = mean + stdMult*sampleStd, so cutoff couples every element in a
// row. With m_i=1 iff g_i>cutoff, S=sum_i dA_i*m_i, and dcutoff/dg_j =
// 1/width + stdMult*(g_j-mean)/((width-1)*std):
//
//	dGate_j = dA_j*m_j - S*dcutoff/dg_j.
//
// dGate is cleared then filled. std==0 rows (constant input) contribute only
// the mean term.
func ActivationSparsityBackward(dGate, gate, dA []float32, rows, width int, stdMult float64) {
	clear(dGate)
	for r := 0; r < rows; r++ {
		base := r * width
		g := gate[base : base+width]
		da := dA[base : base+width]
		var sum float64
		for _, gv := range g {
			sum += float64(gv)
		}
		mean := sum / float64(width)
		var sq float64
		for _, gv := range g {
			d := float64(gv) - mean
			sq += d * d
		}
		std := math.Sqrt(sq / float64(width-1))
		cutoff := mean + stdMult*std
		var S float64
		for i, gv := range g {
			if float64(gv) > cutoff {
				S += float64(da[i])
			}
		}
		invW := 1.0 / float64(width)
		for i, gv := range g {
			var dcut float64
			if std > 0 {
				dcut = invW + stdMult*(float64(gv)-mean)/(float64(width-1)*std)
			} else {
				dcut = invW
			}
			var v float64
			if float64(gv) > cutoff {
				v = float64(da[i])
			}
			dGate[base+i] = float32(v - S*dcut)
		}
	}
}

// EmbedInputScaleBackward: VJP of the input-embedding scaling h = scale*E[token]
// (the gemma sqrt(d) input-embed scale). Accumulates the per-token upstream
// gradient into the shared embedding-table rows: dEmbed[token] += scale*dHidden.
// dEmbed must be pre-cleared (or hold prior accumulation); tokens index rows of
// width d. Rows repeated across tokens accumulate, matching a tied embed table.
func EmbedInputScaleBackward(dEmbed, dHidden []float32, tokens []int, d int, scale float64) {
	for t, tok := range tokens {
		src := dHidden[t*d : (t+1)*d]
		dst := dEmbed[tok*d : (tok+1)*d]
		for i := 0; i < d; i++ {
			dst[i] += float32(scale * float64(src[i]))
		}
	}
}
