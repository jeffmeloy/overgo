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

// CausalAttention: softmax(q·k^T)·v per head over a causal span with score
// scale 1 — any scale (fixed or learned) is folded into q by the caller.
// Layout is [seq][heads][headDim] flat; scores are f64 dots stored f32.
func CausalAttention(out, q, k, v []float32, seq, heads, headDim int) {
	clear(out)
	parallelRange(heads, func(hStart, hEnd int) {
		causalAttentionHeads(out, q, k, v, seq, heads, headDim, hStart, hEnd)
	})
}

// causalAttentionHeads runs the head range [hStart,hEnd) — each worker owns
// disjoint out rows per head, so the split is race-free.
func causalAttentionHeads(out, q, k, v []float32, seq, heads, headDim, hStart, hEnd int) {
	scores := make([]float32, seq)
	for h := hStart; h < hEnd; h++ {
		for qi := 0; qi < seq; qi++ {
			qRow := q[(qi*heads+h)*headDim : (qi*heads+h+1)*headDim]
			probs := scores[:qi+1]
			for ki := range probs {
				kRow := k[(ki*heads+h)*headDim : (ki*heads+h+1)*headDim]
				var dot float64
				for d := 0; d < headDim; d++ {
					dot += float64(qRow[d]) * float64(kRow[d])
				}
				probs[ki] = float32(dot)
			}
			SoftmaxInPlace(probs)
			outRow := out[(qi*heads+h)*headDim : (qi*heads+h+1)*headDim]
			for ki, weight := range probs {
				vRow := v[(ki*heads+h)*headDim : (ki*heads+h+1)*headDim]
				for d := 0; d < headDim; d++ {
					outRow[d] += weight * vRow[d]
				}
			}
		}
	}
}
