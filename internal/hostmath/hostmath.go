// Package hostmath is the neutral home for host (CPU, float32-storage,
// float64-accumulation) reference math shared by capability packages. One
// owner per primitive; nothing here knows any model family — callers supply
// every dimension and weight, all derived from artifacts.
package hostmath

import "math"

// Transpose2D materializes the transpose of a row-major matrix.
func Transpose2D(x []float32, rows, columns int) []float32 {
	out := make([]float32, len(x))
	for row := 0; row < rows; row++ {
		for column := 0; column < columns; column++ {
			out[column*rows+row] = x[row*columns+column]
		}
	}
	return out
}

// GradientSlot returns the named accumulation buffer, allocating it on first touch.
func GradientSlot(gradients map[string][]float32, name string, size int) []float32 {
	if gradients == nil {
		return nil
	}
	if slot, ok := gradients[name]; ok {
		return slot
	}
	slot := make([]float32, size)
	gradients[name] = slot
	return slot
}

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

// LayerNormInto: classic LayerNorm per row — mean-subtracted, BIASED
// variance (divide by d), optional affine. weight and bias come together or
// not at all (nil/nil is the no-affine variant). out may alias x. f64 stats.
func LayerNormInto(out, x, weight, bias []float32, rows, d int, eps float64) {
	affine := weight != nil || bias != nil
	if affine && (len(weight) != d || len(bias) != d) {
		panic("hostmath: LayerNormInto affine vectors do not match width")
	}
	for r := 0; r < rows; r++ {
		row := x[r*d : (r+1)*d]
		var mean float64
		for _, v := range row {
			mean += float64(v)
		}
		mean /= float64(d)
		var variance float64
		for _, v := range row {
			dv := float64(v) - mean
			variance += dv * dv
		}
		variance /= float64(d)
		inv := 1.0 / math.Sqrt(variance+eps)
		o := out[r*d : (r+1)*d]
		for j := 0; j < d; j++ {
			n := (float64(row[j]) - mean) * inv
			if affine {
				n = n*float64(weight[j]) + float64(bias[j])
			}
			o[j] = float32(n)
		}
	}
}

// LayerNormF32AffineInto rounds normalization before the affine step.
func LayerNormF32AffineInto(out, x, weight, bias []float32, rows, d int, eps float32) {
	parallelRangeCost(rows, 2*d, macF64, func(start, end int) {
		LayerNormInto(out[start*d:end*d], x[start*d:end*d], nil, nil, end-start, d, float64(eps))
		for row := start; row < end; row++ {
			values := out[row*d : (row+1)*d]
			for column, value := range values {
				values[column] = value*weight[column] + bias[column]
			}
		}
	})
}

// GELUErf: the EXACT (erf) GELU 0.5*x*(1+erf(x/sqrt(2))) — distinct from the
// tanh approximation above.
func GELUErf(x float64) float64 { return 0.5 * x * (1 + math.Erf(x/math.Sqrt2)) }

// ShiftFlowSigma applies rational flow-time shifting.
func ShiftFlowSigma(sigma, shift float64) float64 {
	return shift * sigma / (1 + (shift-1)*sigma)
}

// GELUErfPrime: derivative of GELUErf,
// 0.5(1+erf(x/sqrt2)) + x*exp(-x^2/2)/sqrt(2pi) — the scalar VJP factor for
// the erf GELU, for callers mixing it into a larger reduction per element.
func GELUErfPrime(x float64) float64 {
	return 0.5*(1+math.Erf(x/math.Sqrt2)) + x*math.Exp(-x*x/2)/math.Sqrt(2*math.Pi)
}

// GELUErfInPlace applies GELUErf element-wise, f64 math.
func GELUErfInPlace(v []float32) {
	for k := range v {
		v[k] = float32(GELUErf(float64(v[k])))
	}
}

// RopeInvFreq: the default rotary inverse-frequency ladder 1/theta^(2i/d). d is
// the rotary width (== head_dim for full rope; < head_dim for partial rope).
func RopeInvFreq(theta float64, ropeDim int) []float64 {
	out := make([]float64, ropeDim/2)
	for i := range out {
		out[i] = 1 / math.Pow(theta, float64(2*i)/float64(ropeDim))
	}
	return out
}

// RopeWidth resolves the rotary width applied per head: ropeDim when it is a
// valid partial factor in (0, headDim], else the full headDim (full-rope
// default). Single owner of the serving partial-rope convention: only the first
// RopeWidth dims of each head row are rotated (NeoX split-half), the remainder
// pass through unrotated — matching reference.ropeNeoX with rotary<width.
func RopeWidth(ropeDim, headDim int) int {
	if ropeDim <= 0 || ropeDim > headDim {
		return headDim
	}
	return ropeDim
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
	// rows*inDim MACs per output column.
	parallelRangeCost(outDim, rows*inDim, macF32, func(oStart, oEnd int) {
		linearCols(dst, x, w, rows, inDim, outDim, oStart, oEnd)
	})
}

// LinearBF16: Linear with native BF16 weight storage and F32 accumulation.
func LinearBF16(dst, x []float32, w []uint16, rows, inDim, outDim int) {
	parallelRangeCost(outDim, rows*inDim, macF32, func(oStart, oEnd int) {
		for r := 0; r < rows; r++ {
			xRow := x[r*inDim : (r+1)*inDim]
			dRow := dst[r*outDim : (r+1)*outDim]
			for o := oStart; o < oEnd; o++ {
				wRow := w[o*inDim : (o+1)*inDim]
				var sum float32
				for c, value := range wRow {
					sum += math.Float32frombits(uint32(value)<<16) * xRow[c]
				}
				dRow[o] = sum
			}
		}
	})
}

// LinearBF16BackwardInput applies the input VJP for a native-BF16 linear.
func LinearBF16BackwardInput(dx, dy []float32, w []uint16, rows, inDim, outDim int) {
	parallelRangeCost(inDim, rows*outDim, macF32, func(cStart, cEnd int) {
		for r := 0; r < rows; r++ {
			dyRow := dy[r*outDim : (r+1)*outDim]
			dxRow := dx[r*inDim : (r+1)*inDim]
			for c := cStart; c < cEnd; c++ {
				var sum float32
				for o, gradient := range dyRow {
					sum += gradient * math.Float32frombits(uint32(w[o*inDim+c])<<16)
				}
				dxRow[c] = sum
			}
		}
	})
}

// linearCols: output columns [oStart,oEnd) of Linear — the serial kernel the
// dispatch calibration times.
func linearCols(dst, x, w []float32, rows, inDim, outDim, oStart, oEnd int) {
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

// GELUTanhInPlace: tanh-GELU; F64 arithmetic.
func GELUTanhInPlace(v []float32) {
	for k := range v {
		v[k] = float32(GELUTanh(float64(v[k])))
	}
}

// Tanh-GELU scale.
var geluTanhSqrt2OverPi = math.Sqrt(2 / math.Pi)

// GELUTanhCubicCoefficient: tanh-GELU approximation coefficient.
const GELUTanhCubicCoefficient = 0.044715

// GELUTanh: smooth tanh-GELU; no serving dtype round-trip.
func GELUTanh(x float64) float64 {
	return 0.5 * x * (1 + math.Tanh(geluTanhSqrt2OverPi*(x+GELUTanhCubicCoefficient*x*x*x)))
}

// GELUTanhPrime: GELUTanh derivative.
func GELUTanhPrime(x float64) float64 {
	u := geluTanhSqrt2OverPi * (x + GELUTanhCubicCoefficient*x*x*x)
	t := math.Tanh(u)
	dudx := geluTanhSqrt2OverPi * (1 + 3*GELUTanhCubicCoefficient*x*x)
	return 0.5*(1+t) + 0.5*x*(1-t*t)*dudx
}

// SparseGateInto: the gemma3n activation-sparsity gate. Per row of width
// `width`, cutoff = mean + stdMult*sampleStd (sample std uses width-1), and
// dst_i = relu(gate_i - cutoff). stdMult<=0 with a nil-effect cutoff reduces
// to the identity relu; the caller supplies the model's std multiplier. The
// cutoff depends on every element in the row, so this is not element-wise.
func SparseGateInto(dst, gate []float32, rows, width int, stdMult float64) {
	for r := 0; r < rows; r++ {
		base := r * width
		row := gate[base : base+width]
		var sum float64
		for _, g := range row {
			sum += float64(g)
		}
		mean := sum / float64(width)
		var sq float64
		for _, g := range row {
			d := float64(g) - mean
			sq += d * d
		}
		std := math.Sqrt(sq / float64(width-1))
		cutoff := mean + stdMult*std
		for i, g := range row {
			d := float64(g) - cutoff
			if d > 0 {
				dst[base+i] = float32(d)
			} else {
				dst[base+i] = 0
			}
		}
	}
}

// WindowedCausalAttention: causal attention where each query qi attends only
// keys in [max(0,qi-window+1), qi]; window<=0 is the full causal span, making
// this a strict generalization of CausalAttention. Score scale is 1 (the
// caller folds any 1/sqrt(headDim) into q, matching CausalAttention). Layout:
// q/out [seq][heads][headDim], k/v [seq][kvHeads][headDim]; query head h reads
// kv head h/(heads/kvHeads). Softmax reduces in f64.
func WindowedCausalAttention(out, q, k, v []float32, seq, heads, kvHeads, headDim, window int) {
	clear(out)
	group := heads / kvHeads
	scores := make([]float32, seq)
	for h := 0; h < heads; h++ {
		kv := h / group
		for qi := 0; qi < seq; qi++ {
			lo := 0
			if window > 0 && qi+1 > window {
				lo = qi + 1 - window
			}
			qRow := q[(qi*heads+h)*headDim : (qi*heads+h+1)*headDim]
			probs := scores[:qi+1-lo]
			for m := range probs {
				kRow := k[((lo+m)*kvHeads+kv)*headDim : ((lo+m)*kvHeads+kv+1)*headDim]
				var dot float64
				for d := 0; d < headDim; d++ {
					dot += float64(qRow[d]) * float64(kRow[d])
				}
				probs[m] = float32(dot)
			}
			SoftmaxInPlace(probs)
			outRow := out[(qi*heads+h)*headDim : (qi*heads+h+1)*headDim]
			for m, weight := range probs {
				vRow := v[((lo+m)*kvHeads+kv)*headDim : ((lo+m)*kvHeads+kv+1)*headDim]
				for d := 0; d < headDim; d++ {
					outRow[d] += weight * vRow[d]
				}
			}
		}
	}
}

// MaskedBidirectionalAttention: softmax(q·k^T)·v per head over the FULL key
// span (set attention — no causal order) with score scale 1; any scale is
// folded into q by the caller. keyMask hides masked keys from every query
// (nil = all visible; must admit at least one key). Scores and softmax
// reduce in f64. Layout: q/out [querySeq][heads][headDim] flat,
// k/v [keySeq][kvHeads][headDim]; query head h reads kv head h/(heads/kvHeads).
func MaskedBidirectionalAttention(out, q, k, v []float32, querySeq, keySeq, heads, kvHeads, headDim int, keyMask []bool) {
	ScaledMaskedBidirectionalAttention(out, q, k, v, querySeq, keySeq, heads, kvHeads, headDim, 1, keyMask)
}

// ScaledMaskedBidirectionalAttention: MaskedBidirectionalAttention with an
// explicit score scale — softmax(scale·q·k^T)·v. scale 1 is bit-identical to
// the unscaled variant (the f64 score multiplies by exactly 1).
func ScaledMaskedBidirectionalAttention(out, q, k, v []float32, querySeq, keySeq, heads, kvHeads, headDim int, scale float64, keyMask []bool) {
	if keyMask != nil && len(keyMask) != keySeq {
		panic("hostmath: attention key mask length != keySeq")
	}
	clear(out)
	// Per head: querySeq*keySeq (score dot + value mix) pairs at 2*headDim
	// MACs each; softmax exp is lower-order (keySeq vs keySeq*headDim).
	parallelRangeCost(heads, 2*querySeq*keySeq*headDim, macF64, func(hStart, hEnd int) {
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
					dot *= scale
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
	// Per head: causal span sum_qi(qi+1) = seq*(seq+1)/2 pairs at 2*headDim
	// MACs each (score dot + value mix).
	parallelRangeCost(heads, seq*(seq+1)*headDim, macF64, func(hStart, hEnd int) {
		causalAttentionHeads(out, q, k, v, seq, heads, kvHeads, headDim, hStart, hEnd)
	})
}

// CausalAttentionStep: one cached-decode query at position cachedRows-1
// attending the cachedRows keys/values accumulated so far — the incremental
// form of CausalAttention. Layout: q/out [heads][headDim] flat;
// kCache/vCache [cachedRows][kvHeads][headDim].
func CausalAttentionStep(out, q, kCache, vCache []float32, cachedRows, heads, kvHeads, headDim int) {
	CausalAttentionSteps(out, q, kCache, vCache, cachedRows-1, 1, heads, kvHeads, headDim)
}

// CausalAttentionSteps: T new cached-decode queries at positions
// rows0..rows0+T-1, each attending the cache prefix through its own row —
// the batched form of CausalAttentionStep. Same per-row math (f64 dots
// stored f32, SoftmaxInPlace, f32 value accumulation), so a stepped decode
// is bit-identical to the full-sequence core's row at that position.
// Layout: q/out [T][heads][headDim] flat; kCache/vCache hold rows0+T rows
// [kvHeads][headDim]. Parallel over heads — each worker owns disjoint out
// slices per row.
func CausalAttentionSteps(out, q, kCache, vCache []float32, rows0, T, heads, kvHeads, headDim int) {
	clear(out)
	group := heads / kvHeads
	// Per head: sum_t (rows0+t+1) key rows at 2*headDim MACs each (score
	// dot + value mix).
	unit := 2 * headDim * (T*rows0 + T*(T+1)/2)
	parallelRangeCost(heads, unit, macF64, func(hStart, hEnd int) {
		scores := make([]float32, rows0+T)
		for h := hStart; h < hEnd; h++ {
			kv := h / group
			for t := 0; t < T; t++ {
				probs := scores[:rows0+t+1]
				qRow := q[(t*heads+h)*headDim : (t*heads+h+1)*headDim]
				for ki := range probs {
					kRow := kCache[(ki*kvHeads+kv)*headDim : (ki*kvHeads+kv+1)*headDim]
					var dot float64
					for d := 0; d < headDim; d++ {
						dot += float64(qRow[d]) * float64(kRow[d])
					}
					probs[ki] = float32(dot)
				}
				SoftmaxInPlace(probs)
				outRow := out[(t*heads+h)*headDim : (t*heads+h+1)*headDim]
				for ki, weight := range probs {
					vRow := vCache[(ki*kvHeads+kv)*headDim : (ki*kvHeads+kv+1)*headDim]
					for d := 0; d < headDim; d++ {
						outRow[d] += weight * vRow[d]
					}
				}
			}
		}
	})
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
