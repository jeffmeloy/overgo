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

// SiLUBackward: dst = dy * silu'(x).
func SiLUBackward(dst, x, dy []float32) {
	for i, v := range x {
		s := 1.0 / (1.0 + math.Exp(-float64(v)))
		dst[i] = float32(float64(dy[i]) * s * (1.0 + float64(v)*(1.0-s)))
	}
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
			wRow := w[o*inDim : (o+1)*inDim]
			if dW != nil {
				dWRow := dW[o*inDim : (o+1)*inDim]
				for c := 0; c < inDim; c++ {
					dWRow[c] += g * xRow[c]
				}
			}
			if dxRow != nil {
				for c := 0; c < inDim; c++ {
					dxRow[c] += g * wRow[c]
				}
			}
		}
	}
}

// CausalAttentionBackward: VJP of CausalAttention (score scale 1, causal,
// per-head KV). Probabilities are recomputed row-by-row; the softmax VJP is
// ds = p*(dP - p·dP). Layout matches the forward: [seq][heads][headDim].
func CausalAttentionBackward(dq, dk, dv, q, k, v, dOut []float32, seq, heads, headDim int) {
	clear(dq)
	clear(dk)
	clear(dv)
	probs := make([]float64, seq)
	dP := make([]float64, seq)
	for h := 0; h < heads; h++ {
		for qi := 0; qi < seq; qi++ {
			nk := qi + 1
			qRow := q[(qi*heads+h)*headDim : (qi*heads+h+1)*headDim]
			dout := dOut[(qi*heads+h)*headDim : (qi*heads+h+1)*headDim]
			dqRow := dq[(qi*heads+h)*headDim : (qi*heads+h+1)*headDim]
			mx := math.Inf(-1)
			for m := 0; m < nk; m++ {
				kRow := k[(m*heads+h)*headDim : (m*heads+h+1)*headDim]
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
				vRow := v[(m*heads+h)*headDim : (m*heads+h+1)*headDim]
				dvRow := dv[(m*heads+h)*headDim : (m*heads+h+1)*headDim]
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
				kRow := k[(m*heads+h)*headDim : (m*heads+h+1)*headDim]
				dkRow := dk[(m*heads+h)*headDim : (m*heads+h+1)*headDim]
				for x := 0; x < headDim; x++ {
					dqRow[x] += float32(g * float64(kRow[x]))
					dkRow[x] += float32(g * float64(qRow[x]))
				}
			}
		}
	}
}
