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
			inv := 1.0 / sum
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
	for r := range rows {
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

// MatchMagnitudeBackward: VJP of the gemma3n per-row match-magnitude rescale
// out_j = sqrt(||target||^2/||input||^2) * input_j, which couples every input
// AND target element in a row. With si=sum input^2, st=sum target^2,
// scale=sqrt(st/si) and D=sum_j dOut_j*input_j:
//
//	dInput_k  = scale*(dOut_k - input_k*D/si)
//	dTarget_k = scale*target_k*D/st
//
// dInput/dTarget are set (each row overwritten). si==0 rows are identity
// (out=input): dInput=dOut, dTarget=0, matching the forward's skip branch;
// st==0 rows have scale=0 so both grads are zero. Layout is row-major
// [rows,width] (token-major, feature-fastest), matching the forward.
func MatchMagnitudeBackward(dInput, dTarget, input, target, dOut []float32, rows, width int) {
	for r := range rows {
		base := r * width
		in := input[base : base+width]
		tg := target[base : base+width]
		dO := dOut[base : base+width]
		di := dInput[base : base+width]
		dt := dTarget[base : base+width]
		var si, st, dot float64
		for j := range width {
			si += float64(in[j]) * float64(in[j])
			st += float64(tg[j]) * float64(tg[j])
			dot += float64(dO[j]) * float64(in[j])
		}
		if si == 0 {
			for j := range width {
				di[j] = dO[j]
				dt[j] = 0
			}
			continue
		}
		scale := math.Sqrt(st / si)
		for j := range width {
			di[j] = float32(scale * (float64(dO[j]) - float64(in[j])*dot/si))
		}
		if st == 0 {
			for j := range width {
				dt[j] = 0
			}
			continue
		}
		for j := range width {
			dt[j] = float32(scale * float64(tg[j]) * dot / st)
		}
	}
}

// TanhBackward: VJP of the element-wise tanh (the gemma3n AltUp-router and
// modality-coefficient nonlinearity): dst = dy*(1-tanh(x)^2). dst may alias dy.
func TanhBackward(dst, x, dy []float32) {
	for i := range x {
		t := math.Tanh(float64(x[i]))
		dst[i] = float32(float64(dy[i]) * (1 - t*t))
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
		for i := range d {
			dst[i] += float32(scale * float64(src[i]))
		}
	}
}
