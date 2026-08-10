package thoughtbank

import (
	"math"

	"overgo/internal/hostmath"
)

// AttentionSinkSoftmaxBackward differentiates AttentionSinkSoftmaxInto.
//
// The forward, per row and head, with M = max_k l_k:
//
//	e_i = exp(l_i - M)
//	D   = exp(sink) + sum_k e_k        <- sink DELIBERATELY UN-SHIFTED
//	p_i = e_i / D
//
// The un-shifted sink is the subtlety. In an ordinary softmax the max
// subtraction cancels between numerator and denominator, so M contributes
// nothing to the gradient. Here exp(sink) sits in the denominator at full scale
// while the logit terms are scaled by exp(-M), so the sink's SHARE of the
// denominator depends on M, hence on the largest logit.
//
// With s = exp(sink)/D that share and C = sum_i g_i p_i:
//
//	dL/dl_j  = p_j * (g_j - C) - [j == argmax] * s * C
//	dL/dsink = -s * C
//
// The bracketed term is the max-shift correction; it vanishes when s = 0, which
// is why it is absent from textbook softmax derivatives. It lands on the
// forward's argmax, and only the LOGITS define that argmax (deriving it from the
// post-normalisation float32 p can put the correction on the wrong entry).
func AttentionSinkSoftmaxBackward(dOut, out, logits, sink []float32, rows, heads, n int) (dLogits, dSink []float32) {
	dLogits = make([]float32, rows*heads*n)
	dSink = make([]float32, heads)
	if len(dOut) != rows*heads*n || len(out) != rows*heads*n ||
		len(logits) != rows*heads*n || len(sink) != heads {
		return dLogits, dSink
	}

	for r := 0; r < rows; r++ {
		for h := 0; h < heads; h++ {
			base := (r*heads + h) * n
			p := out[base : base+n]
			g := dOut[base : base+n]
			dst := dLogits[base : base+n]

			// C = sum_i g_i p_i, the coupling every softmax derivative carries.
			c := 0.0
			for i := range p {
				c += float64(g[i]) * float64(p[i])
			}

			// s = exp(sink)/D recovered from the OUTPUT: sum_i p_i = 1 - s
			// exactly, because the sink holds the rest of the mass. No logits, no
			// second max, so nothing here can disagree with the forward.
			total := 0.0
			for i := range p {
				total += float64(p[i])
			}
			s := 1.0 - total
			if s < 0 {
				s = 0 // rounding only; the sink's share cannot be negative
			}

			// argmax over the LOGITS with the same strict > the forward uses.
			lg := logits[base : base+n]
			argmax, best := 0, math.Inf(-1)
			for i, v := range lg {
				if f := float64(v); f > best {
					best, argmax = f, i
				}
			}

			for i := range p {
				v := float64(p[i]) * (float64(g[i]) - c)
				if i == argmax {
					v -= s * c
				}
				dst[i] = float32(v)
			}
			dSink[h] += float32(-s * c)
		}
	}
	return dLogits, dSink
}

// softmaxVJPInto writes the softmax VJP dz = p * (da - sum_k p_k da_k) into dz.
// The coupling is returned for callers that want it. This is the shared
// softmax-over-a-block coupling reused by the two KV-compression backwards.
func softmaxVJPInto(dz, p, da []float64) (coupling float64) {
	for k := range p {
		coupling += p[k] * da[k]
	}
	for j := range p {
		dz[j] = p[j] * (da[j] - coupling)
	}
	return coupling
}

// rmsNormBackwardAndWeightGrad is the allocating (dx, dW) form over the hostmath
// owner: dx set (not accumulated), dW = sum_rows dy*xhat.
func rmsNormBackwardAndWeightGrad(x, weight, dy []float32, rows, d int, eps float64) (dx, dW []float32) {
	dx = make([]float32, rows*d)
	dW = make([]float32, d)
	hostmath.RMSNormBackward(dx, dW, x, weight, dy, rows, d, eps, false)
	return dx, dW
}

// linearBackward is the allocating (dx, dW[, db]) form over the hostmath owner
// for y = W x with W [outDim, inDim] row-major.
func linearBackward(x, w, dy []float32, rows, inDim, outDim int, wantBias bool) (dx, dW, db []float32) {
	dx = make([]float32, rows*inDim)
	dW = make([]float32, outDim*inDim)
	if wantBias {
		db = make([]float32, outDim)
	}
	hostmath.LinearBackward(dx, dW, db, x, w, dy, rows, inDim, outDim, false)
	return dx, dW, db
}
