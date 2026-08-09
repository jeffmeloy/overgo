// Package hostmath is the neutral home for host (CPU, float32-storage,
// float64-accumulation) reference math shared by capability packages. One
// owner per primitive; nothing here knows any model family — callers supply
// every dimension and weight, all derived from artifacts.
package hostmath

import "math"

// RMSNormInto: out = x/sqrt(mean(x^2)+eps) * weight per row; a nil weight is
// unit scale. out may alias x.
func RMSNormInto(out, x, weight []float32, rows, d int, eps float64) {
	for r := 0; r < rows; r++ {
		row := x[r*d : (r+1)*d]
		var ss float64
		for _, v := range row {
			ss += float64(v) * float64(v)
		}
		inv := 1.0 / math.Sqrt(ss/float64(d)+eps)
		o := out[r*d : (r+1)*d]
		if weight == nil {
			for i, v := range row {
				o[i] = float32(float64(v) * inv)
			}
			continue
		}
		for i, v := range row {
			o[i] = float32(float64(v) * inv * float64(weight[i]))
		}
	}
}

// RopeInvFreq: the default rotary inverse-frequency ladder 1/theta^(2i/d).
func RopeInvFreq(theta float64, headDim int) []float64 {
	out := make([]float64, headDim/2)
	for i := range out {
		out[i] = 1 / math.Pow(theta, float64(2*i)/float64(headDim))
	}
	return out
}

// ApplyRotaryHalf rotates half-split pairs (i, i+d/2) of one head row by
// pos*invFreq[i] (in place).
func ApplyRotaryHalf(x []float32, invFreq []float64, pos int) {
	h := len(x) / 2
	for i := 0; i < h; i++ {
		a := float64(pos) * invFreq[i]
		c, s := math.Cos(a), math.Sin(a)
		x1, x2 := float64(x[i]), float64(x[i+h])
		x[i] = float32(x1*c - x2*s)
		x[i+h] = float32(x2*c + x1*s)
	}
}

// ApplyRotaryInterleaved rotates interleaved pairs (2i, 2i+1) of one head
// row by pos*invFreq[i] (in place) — the GGML "normal" rope layout; the
// split-half HF layout is ApplyRotaryHalf.
func ApplyRotaryInterleaved(x []float32, invFreq []float64, pos int) {
	h := len(x) / 2
	for i := 0; i < h; i++ {
		a := float64(pos) * invFreq[i]
		c, s := math.Cos(a), math.Sin(a)
		x1, x2 := float64(x[2*i]), float64(x[2*i+1])
		x[2*i] = float32(x1*c - x2*s)
		x[2*i+1] = float32(x1*s + x2*c)
	}
}

// SoftmaxInPlace: max-subtracted softmax; f32 storage, f64 sum.
func SoftmaxInPlace(row []float32) {
	if len(row) == 0 {
		return
	}
	mx := row[0]
	for _, value := range row[1:] {
		if value > mx {
			mx = value
		}
	}
	var sum float64
	for i, value := range row {
		exp := math.Exp(float64(value) - float64(mx))
		row[i] = float32(exp)
		sum += exp
	}
	inv := 1 / sum
	for i := range row {
		row[i] = float32(float64(row[i]) * inv)
	}
}

// Softplus: log(1+e^x).
func Softplus(x float64) float64 { return math.Log1p(math.Exp(x)) }

// Linear: dst[rows,out] = x[rows,in] · w[out,in]^T (row-major weight).
// Fans out over the output dimension — each worker owns disjoint dst
// columns, so the split is race-free.
func Linear(dst, x, w []float32, rows, inDim, outDim int) {
	parallelRange(outDim, func(oStart, oEnd int) {
		for r := 0; r < rows; r++ {
			xRow := x[r*inDim : (r+1)*inDim]
			dRow := dst[r*outDim : (r+1)*outDim]
			for o := oStart; o < oEnd; o++ {
				wRow := w[o*inDim : (o+1)*inDim]
				var sum float32
				for c := range wRow {
					sum += wRow[c] * xRow[c]
				}
				dRow[o] = sum
			}
		}
	})
}

// AddBias adds bias element-wise; a nil bias is the no-bias variant.
func AddBias(v, bias []float32) {
	if bias == nil {
		return
	}
	for k := range v {
		v[k] += bias[k]
	}
}

// SiLUInPlace: x*sigmoid(x) element-wise.
func SiLUInPlace(v []float32) {
	for k := range v {
		x := float64(v[k])
		v[k] = float32(x / (1 + math.Exp(-x)))
	}
}

// SiLUGate: dst = silu(gate) * up element-wise — the gated-MLP inner product.
func SiLUGate(dst, gate, up []float32) {
	for i := range dst {
		g := float64(gate[i])
		dst[i] = float32(g / (1 + math.Exp(-g)) * float64(up[i]))
	}
}

// GELUTanhInPlace: the tanh-approximation GELU
// 0.5*x*(1+tanh(sqrt(2/pi)*(x+0.044715*x^3))) element-wise, f64 math.
func GELUTanhInPlace(v []float32) {
	c := math.Sqrt(2 / math.Pi)
	for k := range v {
		x := float64(v[k])
		v[k] = float32(0.5 * x * (1 + math.Tanh(c*(x+0.044715*x*x*x))))
	}
}

// MaskedBidirectionalAttention: softmax(q·k^T)·v per head over the FULL key
// span (set attention — no causal order) with score scale 1; any scale is
// folded into q by the caller. keyMask hides masked keys from every query
// (nil = all visible; must admit at least one key). Scores and softmax
// reduce in f64. Layout: q/out [querySeq][heads][headDim] flat,
// k/v [keySeq][kvHeads][headDim]; query head h reads kv head h/(heads/kvHeads).
func MaskedBidirectionalAttention(out, q, k, v []float32, querySeq, keySeq, heads, kvHeads, headDim int, keyMask []bool) {
	if keyMask != nil && len(keyMask) != keySeq {
		panic("hostmath: attention key mask length != keySeq")
	}
	clear(out)
	parallelRange(heads, func(hStart, hEnd int) {
		scores := make([]float64, keySeq)
		group := heads / kvHeads
		for h := hStart; h < hEnd; h++ {
			kv := h / group
			for qi := 0; qi < querySeq; qi++ {
				qRow := q[(qi*heads+h)*headDim : (qi*heads+h+1)*headDim]
				maxScore := math.Inf(-1)
				for ki := 0; ki < keySeq; ki++ {
					if keyMask != nil && !keyMask[ki] {
						scores[ki] = math.Inf(-1)
						continue
					}
					kRow := k[(ki*kvHeads+kv)*headDim : (ki*kvHeads+kv+1)*headDim]
					var dot float64
					for d := 0; d < headDim; d++ {
						dot += float64(qRow[d]) * float64(kRow[d])
					}
					scores[ki] = dot
					maxScore = max(maxScore, dot)
				}
				var sum float64
				for ki := 0; ki < keySeq; ki++ {
					if math.IsInf(scores[ki], -1) {
						scores[ki] = 0
						continue
					}
					scores[ki] = math.Exp(scores[ki] - maxScore)
					sum += scores[ki]
				}
				outRow := out[(qi*heads+h)*headDim : (qi*heads+h+1)*headDim]
				for ki := 0; ki < keySeq; ki++ {
					if scores[ki] == 0 {
						continue
					}
					weight := scores[ki] / sum
					vRow := v[(ki*kvHeads+kv)*headDim : (ki*kvHeads+kv+1)*headDim]
					for d := 0; d < headDim; d++ {
						outRow[d] += float32(weight * float64(vRow[d]))
					}
				}
			}
		}
	})
}

// CausalAttention: softmax(q·k^T)·v per head over a causal span with score
// scale 1 — any scale (fixed or learned) is folded into q by the caller.
// Layout: q/out [seq][heads][headDim] flat, k/v [seq][kvHeads][headDim];
// query head h reads kv head h/(heads/kvHeads) (grouped-query attention;
// kvHeads == heads is the per-head case). Scores are f64 dots stored f32.
func CausalAttention(out, q, k, v []float32, seq, heads, kvHeads, headDim int) {
	clear(out)
	parallelRange(heads, func(hStart, hEnd int) {
		causalAttentionHeads(out, q, k, v, seq, heads, kvHeads, headDim, hStart, hEnd)
	})
}

// CausalAttentionStep: one cached-decode query at position cachedRows-1
// attending the cachedRows keys/values accumulated so far — the incremental
// form of CausalAttention. Same per-row math (f64 dots stored f32,
// SoftmaxInPlace, f32 value accumulation), so a stepped decode is
// bit-identical to the full-sequence core's row at that position. Layout:
// q/out [heads][headDim] flat; kCache/vCache [cachedRows][kvHeads][headDim].
func CausalAttentionStep(out, q, kCache, vCache []float32, cachedRows, heads, kvHeads, headDim int) {
	clear(out)
	group := heads / kvHeads
	scores := make([]float32, cachedRows)
	for h := 0; h < heads; h++ {
		kv := h / group
		qRow := q[h*headDim : (h+1)*headDim]
		for ki := range scores {
			kRow := kCache[(ki*kvHeads+kv)*headDim : (ki*kvHeads+kv+1)*headDim]
			var dot float64
			for d := 0; d < headDim; d++ {
				dot += float64(qRow[d]) * float64(kRow[d])
			}
			scores[ki] = float32(dot)
		}
		SoftmaxInPlace(scores)
		outRow := out[h*headDim : (h+1)*headDim]
		for ki, weight := range scores {
			vRow := vCache[(ki*kvHeads+kv)*headDim : (ki*kvHeads+kv+1)*headDim]
			for d := 0; d < headDim; d++ {
				outRow[d] += weight * vRow[d]
			}
		}
	}
}

// causalAttentionHeads runs the head range [hStart,hEnd) — each worker owns
// disjoint out rows per head, so the split is race-free.
func causalAttentionHeads(out, q, k, v []float32, seq, heads, kvHeads, headDim, hStart, hEnd int) {
	scores := make([]float32, seq)
	group := heads / kvHeads
	for h := hStart; h < hEnd; h++ {
		kv := h / group
		for qi := 0; qi < seq; qi++ {
			qRow := q[(qi*heads+h)*headDim : (qi*heads+h+1)*headDim]
			probs := scores[:qi+1]
			for ki := range probs {
				kRow := k[(ki*kvHeads+kv)*headDim : (ki*kvHeads+kv+1)*headDim]
				var dot float64
				for d := 0; d < headDim; d++ {
					dot += float64(qRow[d]) * float64(kRow[d])
				}
				probs[ki] = float32(dot)
			}
			SoftmaxInPlace(probs)
			outRow := out[(qi*heads+h)*headDim : (qi*heads+h+1)*headDim]
			for ki, weight := range probs {
				vRow := v[(ki*kvHeads+kv)*headDim : (ki*kvHeads+kv+1)*headDim]
				for d := 0; d < headDim; d++ {
					outRow[d] += weight * vRow[d]
				}
			}
		}
	}
}
