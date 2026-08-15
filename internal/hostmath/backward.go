// Backward (VJP) counterparts of the forward primitives, ported from the
// verified reference math: f32 storage, f64 accumulation, recompute-based
// attention backward (no stored probabilities).
package hostmath

import "math"

// RMSNormBackward: given dy at the norm output, writes dx (set, or += when
// addDX) and accumulates the scale gradient into dscale (nil skips it). A
// nil weight is the unit-scale variant.
func RMSNormBackward(dx, dscale, x, weight, dy []float32, rows, d int, eps float64, addDX bool) {
	affine := len(weight) != 0
	for r := 0; r < rows; r++ {
		xr := x[r*d : (r+1)*d]
		dyr := dy[r*d : (r+1)*d]
		dxr := dx[r*d : (r+1)*d]
		var ss float64
		for _, v := range xr {
			ss += float64(v) * float64(v)
		}
		inv := 1.0 / math.Sqrt(ss/float64(d)+eps)
		var dotGX float64
		for i := 0; i < d; i++ {
			gi := float64(dyr[i])
			if affine {
				gi *= float64(weight[i])
			}
			dotGX += gi * float64(xr[i])
			if dscale != nil {
				dscale[i] += float32(float64(dyr[i]) * float64(xr[i]) * inv)
			}
		}
		coef := inv * inv * inv / float64(d) * dotGX
		for i := 0; i < d; i++ {
			gi := float64(dyr[i])
			if affine {
				gi *= float64(weight[i])
			}
			value := float32(inv*gi - coef*float64(xr[i]))
			if addDX {
				dxr[i] += value
			} else {
				dxr[i] = value
			}
		}
	}
}

// LayerNormBackward: VJP of LayerNormInto (classic LN, biased variance).
// Given dy at the norm output, writes dx (set, or += when addDX) and
// accumulates dW += dy*xhat and dB += dy when non-nil. A nil weight is the
// no-affine variant (dW/dB must then be nil). f64 stats, matching forward.
func LayerNormBackward(dx, dW, dB, x, weight, dy []float32, rows, d int, eps float64, addDX bool) {
	affine := len(weight) != 0
	for r := 0; r < rows; r++ {
		xr := x[r*d : (r+1)*d]
		dyr := dy[r*d : (r+1)*d]
		dxr := dx[r*d : (r+1)*d]
		var mean float64
		for _, v := range xr {
			mean += float64(v)
		}
		mean /= float64(d)
		var variance float64
		for _, v := range xr {
			dv := float64(v) - mean
			variance += dv * dv
		}
		inv := 1.0 / math.Sqrt(variance/float64(d)+eps)

		var meanG, meanGXHat float64
		for i := 0; i < d; i++ {
			xhat := (float64(xr[i]) - mean) * inv
			g := float64(dyr[i])
			if affine {
				g *= float64(weight[i])
			}
			meanG += g
			meanGXHat += g * xhat
			if dW != nil {
				dW[i] += float32(float64(dyr[i]) * xhat)
			}
			if dB != nil {
				dB[i] += dyr[i]
			}
		}
		meanG /= float64(d)
		meanGXHat /= float64(d)
		for i := 0; i < d; i++ {
			xhat := (float64(xr[i]) - mean) * inv
			g := float64(dyr[i])
			if affine {
				g *= float64(weight[i])
			}
			value := float32(inv * (g - meanG - xhat*meanGXHat))
			if addDX {
				dxr[i] += value
			} else {
				dxr[i] = value
			}
		}
	}
}

// GELUErfBackward: dst = dy * gelu'(x) for the EXACT erf GELU;
// gelu'(x) = 0.5*(1+erf(x/sqrt2)) + x*exp(-x^2/2)/sqrt(2*pi). dst may alias dy.
func GELUErfBackward(dst, x, dy []float32) {
	invSqrt2Pi := 1 / math.Sqrt(2*math.Pi)
	for i := range x {
		xv := float64(x[i])
		g := 0.5*(1+math.Erf(xv/math.Sqrt2)) + xv*invSqrt2Pi*math.Exp(-0.5*xv*xv)
		dst[i] = float32(float64(dy[i]) * g)
	}
}

// RotaryHalfBackward: the VJP of ApplyRotaryHalf — rotation by the negated
// angle, in place on one head row's gradient.
func RotaryHalfBackward(dx []float32, invFreq []float64, pos int) {
	h := len(dx) / 2
	for i := 0; i < h; i++ {
		a := float64(pos) * invFreq[i]
		c, s := math.Cos(a), math.Sin(a)
		d1, d2 := float64(dx[i]), float64(dx[i+h])
		dx[i] = float32(c*d1 + s*d2)
		dx[i+h] = float32(-s*d1 + c*d2)
	}
}

// RotaryInterleavedBackward: the VJP of ApplyRotaryInterleaved — rotation by
// the negated angle, in place on one head row's gradient.
func RotaryInterleavedBackward(dx []float32, invFreq []float64, pos int) {
	h := len(dx) / 2
	for i := 0; i < h; i++ {
		a := float64(pos) * invFreq[i]
		c, s := math.Cos(a), math.Sin(a)
		d1, d2 := float64(dx[2*i]), float64(dx[2*i+1])
		dx[2*i] = float32(c*d1 + s*d2)
		dx[2*i+1] = float32(-s*d1 + c*d2)
	}
}

// SiLUBackward: dst = dy * silu'(x).
func SiLUBackward(dst, x, dy []float32) {
	for i, v := range x {
		s := 1.0 / (1.0 + math.Exp(-float64(v)))
		dst[i] = float32(float64(dy[i]) * s * (1.0 + float64(v)*(1.0-s)))
	}
}

// SiLUGateBackward: VJP of SiLUGate — dGate = dy*up*silu'(gate),
// dUp = dy*silu(gate). Both set.
func SiLUGateBackward(dGate, dUp, gate, up, dy []float32) {
	for i := range gate {
		g := float64(gate[i])
		s := 1.0 / (1.0 + math.Exp(-g))
		silu := g * s
		dGate[i] = float32(float64(dy[i]) * float64(up[i]) * s * (1.0 + g*(1.0-s)))
		dUp[i] = float32(float64(dy[i]) * silu)
	}
}

// SoftmaxCrossEntropy: mean cross-entropy over rows of logits [rows,classes]
// against integer targets; writes the mean-loss gradient
// (softmax - onehot)/rows into dLogits and returns the loss. f64 throughout.
func SoftmaxCrossEntropy(dLogits, logits []float32, targets []int, rows, classes int) float64 {
	var loss float64
	invRows := 1.0 / float64(rows)
	for r := 0; r < rows; r++ {
		row := logits[r*classes : (r+1)*classes]
		mx := float64(row[0])
		for _, v := range row[1:] {
			if float64(v) > mx {
				mx = float64(v)
			}
		}
		var sum float64
		for _, v := range row {
			sum += math.Exp(float64(v) - mx)
		}
		logSum := math.Log(sum) + mx
		target := targets[r]
		loss += (logSum - float64(row[target])) * invRows
		dRow := dLogits[r*classes : (r+1)*classes]
		for c := 0; c < classes; c++ {
			p := math.Exp(float64(row[c]) - logSum)
			if c == target {
				p -= 1
			}
			dRow[c] = float32(p * invRows)
		}
	}
	return loss
}

// LinearBackward: forward was dst = x·w^T with w row-major [out,in]. Writes
// dx (set, or += when addDX), and accumulates dW += dy^T⊗x and dB += dy
// when non-nil.
func LinearBackward(dx, dW, dB, x, w, dy []float32, rows, inDim, outDim int, addDX bool) {
	for r := 0; r < rows; r++ {
		xRow := x[r*inDim : (r+1)*inDim]
		dyRow := dy[r*outDim : (r+1)*outDim]
		var dxRow []float32
		if dx != nil {
			dxRow = dx[r*inDim : (r+1)*inDim]
			if !addDX {
				clear(dxRow)
			}
		}
		for o := 0; o < outDim; o++ {
			g := dyRow[o]
			if dB != nil {
				dB[o] += g
			}
			if g == 0 {
				continue
			}
			if dW != nil {
				dWRow := dW[o*inDim : (o+1)*inDim]
				for c := 0; c < inDim; c++ {
					dWRow[c] += g * xRow[c]
				}
			}
			if dxRow != nil {
				wRow := w[o*inDim : (o+1)*inDim]
				for c := 0; c < inDim; c++ {
					dxRow[c] += g * wRow[c]
				}
			}
		}
	}
}

// CausalAttentionBackward: VJP of CausalAttention (score scale 1, causal,
// grouped-query KV — dk/dv accumulate across the query heads sharing each kv
// head, so the head loop stays serial). Probabilities are recomputed
// row-by-row; the softmax VJP is ds = p*(dP - p·dP). Layout matches the
// forward: q [seq][heads][headDim], k/v [seq][kvHeads][headDim].
func CausalAttentionBackward(dq, dk, dv, q, k, v, dOut []float32, seq, heads, kvHeads, headDim int) {
	clear(dq)
	clear(dk)
	clear(dv)
	probs := make([]float64, seq)
	dP := make([]float64, seq)
	group := heads / kvHeads
	for h := 0; h < heads; h++ {
		kv := h / group
		for qi := 0; qi < seq; qi++ {
			nk := qi + 1
			qRow := q[(qi*heads+h)*headDim : (qi*heads+h+1)*headDim]
			dout := dOut[(qi*heads+h)*headDim : (qi*heads+h+1)*headDim]
			dqRow := dq[(qi*heads+h)*headDim : (qi*heads+h+1)*headDim]
			mx := math.Inf(-1)
			for m := 0; m < nk; m++ {
				kRow := k[(m*kvHeads+kv)*headDim : (m*kvHeads+kv+1)*headDim]
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
				vRow := v[(m*kvHeads+kv)*headDim : (m*kvHeads+kv+1)*headDim]
				dvRow := dv[(m*kvHeads+kv)*headDim : (m*kvHeads+kv+1)*headDim]
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
				kRow := k[(m*kvHeads+kv)*headDim : (m*kvHeads+kv+1)*headDim]
				dkRow := dk[(m*kvHeads+kv)*headDim : (m*kvHeads+kv+1)*headDim]
				for x := 0; x < headDim; x++ {
					dqRow[x] += float32(g * float64(kRow[x]))
					dkRow[x] += float32(g * float64(qRow[x]))
				}
			}
		}
	}
}
