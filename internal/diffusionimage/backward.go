package diffusionimage

import (
	"fmt"
	"math"

	"overgo/internal/hostmath"
	"overgo/internal/tensor/dtype"
)

// Grads: parameter gradients keyed by (prefix-stripped) tensor name.
// Targets are zero-initialized once and accumulated into.
type Grads map[string][]float32

func (g Grads) addScalar(name string, v float32) {
	hostmath.GradientSlot(g, name, 1)[0] += v
}

func (g Grads) addF64(name string, values []float64) {
	dst := hostmath.GradientSlot(g, name, len(values))
	for i, v := range values {
		dst[i] += float32(v)
	}
}

// forwardTrace: retained activations for the full-model VJP.
type forwardTrace struct {
	input          []float32
	batch          int
	imageH, imageW int
	encLevels      []levelTrace
	encDowns       []resizeTrace
	decLevels      []levelTrace
	decUps         []resizeTrace
	middleInputs   [][]float32
	middleResidual []float32
	finalInput     []float32
	middleC        int
	middleH        int
	middleW        int
	finalH, finalW int
}

type levelTrace struct {
	input              []float32
	blockInputs        [][]float32
	channels, h, width int
}

type resizeTrace struct {
	input, residual         []float32
	channels, nextChannels  int
	h, width, residualLevel int
}

type resBlockTrace struct {
	input, norm1, activated1, hidden1 []float32
	norm2, activated2, branch, output []float32
}

// conv2dValidStrideBackward: VJP of conv2dValidStride; accumulates dW/dB
// into grads under name.weight/.bias, returns dx.
func conv2dValidStrideBackward(grads Grads, name string, x, weight, dOut []float32, b, cin, cout, h, w, kernel, stride int) []float32 {
	oh := (h-kernel)/stride + 1
	ow := (w-kernel)/stride + 1
	kernelElems := kernel * kernel
	dxf := make([]float64, b*cin*h*w)
	dWf := make([]float64, cout*cin*kernelElems)
	dBf := make([]float64, cout)
	for bi := range b {
		for co := range cout {
			wco := co * cin * kernelElems
			for oy := range oh {
				for ox := range ow {
					g := float64(dOut[(bi*cout+co)*oh*ow+oy*ow+ox])
					dBf[co] += g
					for ci := range cin {
						xb := (bi*cin + ci) * h * w
						wc := wco + ci*kernelElems
						iy, ix := oy*stride, ox*stride
						for ky := range kernel {
							for kx := range kernel {
								xi := xb + (iy+ky)*w + ix + kx
								wi := wc + ky*kernel + kx
								dWf[wi] += g * float64(x[xi])
								dxf[xi] += g * float64(weight[wi])
							}
						}
					}
				}
			}
		}
	}
	grads.addF64(name+".weight", dWf)
	grads.addF64(name+".bias", dBf)
	return dtype.Float64SliceToFloat32(dxf)
}

// convTranspose2dStrideBackward: VJP of convTranspose2dStride
// (weight [cin, cout, k, k]).
func convTranspose2dStrideBackward(grads Grads, name string, x, weight, dOut []float32, b, cin, cout, h, w, kernel, stride int) []float32 {
	oh := (h-1)*stride + kernel
	ow := (w-1)*stride + kernel
	kernelElems := kernel * kernel
	dxf := make([]float64, b*cin*h*w)
	dWf := make([]float64, cin*cout*kernelElems)
	dBf := make([]float64, cout)
	for bi := range b {
		for ci := range cin {
			for iy := range h {
				for ix := range w {
					xi := (bi*cin+ci)*h*w + iy*w + ix
					xv := float64(x[xi])
					wci := ci * cout * kernelElems
					for co := range cout {
						wc := wci + co*kernelElems
						ob := (bi*cout + co) * oh * ow
						oy, ox := iy*stride, ix*stride
						for ky := range kernel {
							for kx := range kernel {
								g := float64(dOut[ob+(oy+ky)*ow+ox+kx])
								wi := wc + ky*kernel + kx
								dWf[wi] += xv * g
								dxf[xi] += float64(weight[wi]) * g
							}
						}
					}
				}
			}
		}
	}
	for bi := range b {
		for co := range cout {
			for _, v := range dOut[(bi*cout+co)*oh*ow : (bi*cout+co+1)*oh*ow] {
				dBf[co] += float64(v)
			}
		}
	}
	grads.addF64(name+".weight", dWf)
	grads.addF64(name+".bias", dBf)
	return dtype.Float64SliceToFloat32(dxf)
}

// conv2dSame3x3Backward: VJP of conv2dSame3x3; parallel over input channels
// (dx indexed by (b,cin), dW by (cout,cin,tap): worker writes are disjoint).
func conv2dSame3x3Backward(grads Grads, name string, x, weight, dOut []float32, b, cin, cout, h, w int) []float32 {
	kernelElems := conv3x3Kernel * conv3x3Kernel
	plane := h * w
	dxf := make([]float64, b*cin*plane)
	dWf := make([]float64, cout*cin*kernelElems)
	dBf := make([]float64, cout)
	for bi := range b {
		for co := range cout {
			gb := (bi*cout + co) * plane
			var biasSum float64
			for _, g := range dOut[gb : gb+plane] {
				biasSum += float64(g)
			}
			dBf[co] += biasSum
		}
	}
	hostmath.ParallelRangeF64(cin, 2*b*cout*plane*kernelElems, func(ciLo, ciHi int) {
		for ci := ciLo; ci < ciHi; ci++ {
			for bi := range b {
				xb := (bi*cin + ci) * plane
				for co := range cout {
					gb := (bi*cout + co) * plane
					wc := co*cin*kernelElems + ci*kernelElems
					for di := -conv3x3Radius; di <= conv3x3Radius; di++ {
						iLo, iHi := 0, h
						if di < 0 {
							iLo = -di
						} else if di > 0 {
							iHi = h - di
						}
						for dj := -conv3x3Radius; dj <= conv3x3Radius; dj++ {
							wi := wc + (di+conv3x3Radius)*conv3x3Kernel + (dj + conv3x3Radius)
							wv := float64(weight[wi])
							jLo, jHi := 0, w
							if dj < 0 {
								jLo = -dj
							} else if dj > 0 {
								jHi = w - dj
							}
							if jLo >= jHi {
								continue
							}
							span := jHi - jLo
							var weightAcc float64
							for i := iLo; i < iHi; i++ {
								gRow := dOut[gb+i*w+jLo : gb+i*w+jHi]
								off := xb + (i+di)*w + jLo + dj
								xRow := x[off : off+span]
								dxRow := dxf[off : off+span]
								for k, gv := range gRow {
									g := float64(gv)
									weightAcc += g * float64(xRow[k])
									dxRow[k] += g * wv
								}
							}
							dWf[wi] += weightAcc
						}
					}
				}
			}
		}
	})
	grads.addF64(name+".weight", dWf)
	grads.addF64(name+".bias", dBf)
	return dtype.Float64SliceToFloat32(dxf)
}

func avgPool2xBackward(dOut []float32, b, c, h, w int) []float32 {
	oh, ow := h/2, w/2
	dx := make([]float32, b*c*h*w)
	for bi := range b {
		for ci := range c {
			inBase := (bi*c + ci) * h * w
			outBase := (bi*c + ci) * oh * ow
			for i := range oh {
				for j := range ow {
					g := dOut[outBase+i*ow+j] * 0.25
					p := inBase + 2*i*w + 2*j
					dx[p] += g
					dx[p+1] += g
					dx[p+w] += g
					dx[p+w+1] += g
				}
			}
		}
	}
	return dx
}

func upsampleNearest2xBackward(dOut []float32, b, c, h, w int) []float32 {
	oh, ow := 2*h, 2*w
	dx := make([]float32, b*c*h*w)
	for bi := range b {
		for ci := range c {
			ob := (bi*c + ci) * oh * ow
			ib := (bi*c + ci) * h * w
			for i := range oh {
				for j := range ow {
					dx[ib+(i/2)*w+(j/2)] += dOut[ob+i*ow+j]
				}
			}
		}
	}
	return dx
}

// groupNormBackward: VJP of groupNormInto (f64 stats); accumulates dW/dB.
func groupNormBackward(dx, dWeight, dBias, x, weight, dOut []float32, n, c, h, w, groups int, eps float64) {
	clear(dx)
	channelsPerGroup := c / groups
	hw := h * w
	groupCount := channelsPerGroup * hw
	groupElems := float64(groupCount)
	xhat := make([]float64, groupCount)
	gradNorm := make([]float64, groupCount)
	for ni := range n {
		for g := range groups {
			var sum, sumsq float64
			for ci := g * channelsPerGroup; ci < (g+1)*channelsPerGroup; ci++ {
				base := (ni*c + ci) * hw
				for i := range hw {
					v := float64(x[base+i])
					sum += v
					sumsq += v * v
				}
			}
			mean := sum / groupElems
			inv := 1 / math.Sqrt(sumsq/groupElems-mean*mean+eps)
			var gradSum, gradXHatSum float64
			pos := 0
			for ci := g * channelsPerGroup; ci < (g+1)*channelsPerGroup; ci++ {
				base := (ni*c + ci) * hw
				wc := float64(weight[ci])
				for i := range hw {
					xh := (float64(x[base+i]) - mean) * inv
					gy := float64(dOut[base+i])
					gn := gy * wc
					xhat[pos], gradNorm[pos] = xh, gn
					gradSum += gn
					gradXHatSum += gn * xh
					if dWeight != nil {
						dWeight[ci] += float32(gy * xh)
					}
					if dBias != nil {
						dBias[ci] += float32(gy)
					}
					pos++
				}
			}
			pos = 0
			for ci := g * channelsPerGroup; ci < (g+1)*channelsPerGroup; ci++ {
				base := (ni*c + ci) * hw
				for i := range hw {
					dx[base+i] = float32((groupElems*gradNorm[pos] - gradSum - xhat[pos]*gradXHatSum) * inv / groupElems)
					pos++
				}
			}
		}
	}
}

// xatgluBackward: VJP of xatgluGateInto into dProjected; returns dAlpha.
func xatgluBackward(dProjected, projected, dOut []float32, alpha float64, rows, outDim int) float32 {
	stride := 2 * outDim
	alphaScale := 1 + 2*alpha
	var dAlpha float64
	for r := range rows {
		gatePath := projected[r*stride : r*stride+outDim]
		valuePath := projected[r*stride+outDim : r*stride+stride]
		dGate := dProjected[r*stride : r*stride+outDim]
		dValue := dProjected[r*stride+outDim : r*stride+stride]
		gradOut := dOut[r*outDim : r*outDim+outDim]
		for j := range outDim {
			g := float64(gatePath[j])
			gate := (math.Atan(g) + xatgluHalfPi) * xatgluInvPi
			dy := float64(gradOut[j])
			dGate[j] = float32(dy * float64(valuePath[j]) * alphaScale * xatgluInvPi / (1 + g*g))
			dValue[j] = float32(dy * (gate*alphaScale - alpha))
			dAlpha += dy * float64(valuePath[j]) * (2*gate - 1)
		}
	}
	return float32(dAlpha)
}

// cpFactorContractBackward: VJP of cpFactorContractInto.
func cpFactorContractBackward(dA, dB, aFactor, bFactor, dOut []float32, tokens, heads, rank, headDim int) {
	clear(dA)
	clear(dB)
	invRank := 1 / float64(rank)
	for t := range tokens {
		aBase := t * heads * rank
		bBase := t * rank * headDim
		for h := range heads {
			aRow := aFactor[aBase+h*rank : aBase+(h+1)*rank]
			dARow := dA[aBase+h*rank : aBase+(h+1)*rank]
			gradOut := dOut[(t*heads+h)*headDim : (t*heads+h+1)*headDim]
			for r := range rank {
				bRow := bFactor[bBase+r*headDim : bBase+(r+1)*headDim]
				dBRow := dB[bBase+r*headDim : bBase+(r+1)*headDim]
				var sumA float64
				for d := range headDim {
					grad := float64(gradOut[d]) * invRank
					sumA += grad * float64(bRow[d])
					dBRow[d] += float32(grad * float64(aRow[r]))
				}
				dARow[r] = float32(sumA)
			}
		}
	}
}

// applyRopeBackward: VJP of applyRope — same tables, transposed rotation.
func applyRopeBackward(dx, cos, sin []float32, seq, nHeads, headDim int) {
	half := headDim / 2
	for p := range seq {
		cp := cos[p*headDim : (p+1)*headDim]
		sp := sin[p*headDim : (p+1)*headDim]
		for h := range nHeads {
			vec := dx[(p*nHeads+h)*headDim : (p*nHeads+h+1)*headDim]
			for i := range half {
				j := i + half
				a, b := vec[i], vec[j]
				vec[i] = a*cp[i] + b*sp[i]
				vec[j] = b*cp[j] - a*sp[j]
			}
		}
	}
}

// bidirectionalAttentionBackward: VJP of the full-span SDPA with score
// scale; probabilities recomputed per query row (f64), parallel over heads.
func bidirectionalAttentionBackward(dq, dk, dv, q, k, v, dOut []float32, seq, heads, headDim int, scale float64) {
	clear(dq)
	clear(dk)
	clear(dv)
	hostmath.ParallelRangeF64(heads, 4*seq*seq*headDim, func(hLo, hHi int) {
		probs := make([]float64, seq)
		dP := make([]float64, seq)
		for h := hLo; h < hHi; h++ {
			for qi := range seq {
				qRow := q[(qi*heads+h)*headDim : (qi*heads+h+1)*headDim]
				dout := dOut[(qi*heads+h)*headDim : (qi*heads+h+1)*headDim]
				dqRow := dq[(qi*heads+h)*headDim : (qi*heads+h+1)*headDim]
				mx := math.Inf(-1)
				for m := range seq {
					kRow := k[(m*heads+h)*headDim : (m*heads+h+1)*headDim]
					var dot float64
					for x := range headDim {
						dot += float64(qRow[x]) * float64(kRow[x])
					}
					probs[m] = dot * scale
					if probs[m] > mx {
						mx = probs[m]
					}
				}
				var sum float64
				for m := range seq {
					probs[m] = math.Exp(probs[m] - mx)
					sum += probs[m]
				}
				inv := 1.0 / sum
				var pdotdP float64
				for m := range seq {
					probs[m] *= inv
					vRow := v[(m*heads+h)*headDim : (m*heads+h+1)*headDim]
					dvRow := dv[(m*heads+h)*headDim : (m*heads+h+1)*headDim]
					var dpm float64
					for x := range headDim {
						dpm += float64(dout[x]) * float64(vRow[x])
						dvRow[x] += float32(probs[m] * float64(dout[x]))
					}
					dP[m] = dpm
					pdotdP += probs[m] * dpm
				}
				for m := range seq {
					g := probs[m] * (dP[m] - pdotdP) * scale
					if g == 0 {
						continue
					}
					kRow := k[(m*heads+h)*headDim : (m*heads+h+1)*headDim]
					dkRow := dk[(m*heads+h)*headDim : (m*heads+h+1)*headDim]
					for x := range headDim {
						dqRow[x] += float32(g * float64(kRow[x]))
						dkRow[x] += float32(g * float64(qRow[x]))
					}
				}
			}
		}
	})
}

// linearGrad: hostmath.LinearBackward into named grad targets; returns dx.
func linearGrad(grads Grads, weightName, biasName string, x, w, dy []float32, rows, inDim, outDim int) []float32 {
	dx := make([]float32, rows*inDim)
	var dB []float32
	if biasName != "" {
		dB = hostmath.GradientSlot(grads, biasName, outDim)
	}
	hostmath.LinearBackward(dx, hostmath.GradientSlot(grads, weightName, outDim*inDim), dB, x, w, dy, rows, inDim, outDim, false)
	return dx
}

// layerNormGrad: hostmath.LayerNormBackward into named grad targets.
func layerNormGrad(grads Grads, weightName, biasName string, x, weight, dy []float32, rows, d int, eps float64) []float32 {
	dx := make([]float32, rows*d)
	hostmath.LayerNormBackward(dx, hostmath.GradientSlot(grads, weightName, d), hostmath.GradientSlot(grads, biasName, d), x, weight, dy, rows, d, eps, false)
	return dx
}

// tpaBackward: VJP of tpaForward; recomputes the forward tape.
func tpaBackward(grads Grads, prefix string, x []float32, w tpaWeights, dOut []float32, seq, nEmbd, nHead, headDim, qRank, kvRank int, ropeTheta float64) []float32 {
	Aq := make([]float32, seq*nHead*qRank)
	Ak := make([]float32, seq*nHead*kvRank)
	Av := make([]float32, seq*nHead*kvRank)
	Bq := make([]float32, seq*qRank*headDim)
	Bk := make([]float32, seq*kvRank*headDim)
	Bv := make([]float32, seq*kvRank*headDim)
	hostmath.LinearF64(Aq, x, w.WAq, nil, seq, nEmbd, nHead*qRank)
	hostmath.LinearF64(Ak, x, w.WAk, nil, seq, nEmbd, nHead*kvRank)
	hostmath.LinearF64(Av, x, w.WAv, nil, seq, nEmbd, nHead*kvRank)
	hostmath.LinearF64(Bq, x, w.WBq, nil, seq, nEmbd, qRank*headDim)
	hostmath.LinearF64(Bk, x, w.WBk, nil, seq, nEmbd, kvRank*headDim)
	hostmath.LinearF64(Bv, x, w.WBv, nil, seq, nEmbd, kvRank*headDim)
	q := make([]float32, seq*nHead*headDim)
	k := make([]float32, seq*nHead*headDim)
	v := make([]float32, seq*nHead*headDim)
	cpFactorContractInto(q, Aq, Bq, seq, nHead, qRank, headDim)
	cpFactorContractInto(k, Ak, Bk, seq, nHead, kvRank, headDim)
	cpFactorContractInto(v, Av, Bv, seq, nHead, kvRank, headDim)
	cos, sin := buildRopeTables(seq, headDim, ropeTheta)
	applyRope(q, cos, sin, seq, nHead, headDim)
	applyRope(k, cos, sin, seq, nHead, headDim)

	scale := 1 / math.Sqrt(float64(headDim))
	scaledQ := make([]float32, len(q))
	for i, qv := range q {
		scaledQ[i] = qv * float32(scale)
	}
	attn := make([]float32, seq*nHead*headDim)
	hostmath.MaskedBidirectionalAttention(attn, scaledQ, k, v, seq, seq, nHead, nHead, headDim, nil)
	projected := make([]float32, 2*seq*nEmbd)
	hostmath.LinearF64(projected, attn, w.Woproj, w.Boproj, seq, nHead*headDim, 2*nEmbd)

	dProjected := make([]float32, len(projected))
	dAlphaO := xatgluBackward(dProjected, projected, dOut, w.AlphaO, seq, nEmbd)
	dAttn := linearGrad(grads, prefix+".attn.o_proj.proj.weight", prefix+".attn.o_proj.proj.bias", attn, w.Woproj, dProjected, seq, nHead*headDim, 2*nEmbd)
	dq := make([]float32, len(q))
	dk := make([]float32, len(k))
	dv := make([]float32, len(v))
	bidirectionalAttentionBackward(dq, dk, dv, q, k, v, dAttn, seq, nHead, headDim, scale)
	applyRopeBackward(dq, cos, sin, seq, nHead, headDim)
	applyRopeBackward(dk, cos, sin, seq, nHead, headDim)

	dAq, dBq := make([]float32, len(Aq)), make([]float32, len(Bq))
	dAk, dBk := make([]float32, len(Ak)), make([]float32, len(Bk))
	dAv, dBv := make([]float32, len(Av)), make([]float32, len(Bv))
	cpFactorContractBackward(dAq, dBq, Aq, Bq, dq, seq, nHead, qRank, headDim)
	cpFactorContractBackward(dAk, dBk, Ak, Bk, dk, seq, nHead, kvRank, headDim)
	cpFactorContractBackward(dAv, dBv, Av, Bv, dv, seq, nHead, kvRank, headDim)

	dX := make([]float32, len(x))
	qkv := prefix + ".attn.c_qkv."
	for _, part := range []struct {
		name   string
		w, dy  []float32
		outDim int
	}{
		{qkv + "W_A_q.weight", w.WAq, dAq, nHead * qRank},
		{qkv + "W_A_k.weight", w.WAk, dAk, nHead * kvRank},
		{qkv + "W_A_v.weight", w.WAv, dAv, nHead * kvRank},
		{qkv + "W_B_q.weight", w.WBq, dBq, qRank * headDim},
		{qkv + "W_B_k.weight", w.WBk, dBk, kvRank * headDim},
		{qkv + "W_B_v.weight", w.WBv, dBv, kvRank * headDim},
	} {
		hostmath.LinearBackward(dX, hostmath.GradientSlot(grads, part.name, part.outDim*nEmbd), nil, x, part.w, part.dy, seq, nEmbd, part.outDim, true)
	}
	grads.addScalar(prefix+".attn.o_proj.alpha", dAlphaO)
	return dX
}

func (blk *attnBlock) backward(m *Model, x, dOutImage []float32, batch, channels, h, width int, grads Grads) []float32 {
	cfg := &m.Cfg
	rows := batch * h * width
	trace := blk.forwardTrace(m, x, batch, channels, h, width, true)
	dOut := imageNCHWToTokens(dOutImage, batch, channels, h, width)
	var dMLPScale float64
	for i, v := range dOut {
		dMLPScale += float64(v) * float64(trace.mlpOutput[i])
	}
	dMlpOut := make([]float32, len(dOut))
	for i, v := range dOut {
		dMlpOut[i] = v * blk.mlpScale
	}
	prefix := blk.name
	dMlpHidden := linearGrad(grads, prefix+".mlp.1.weight", "", trace.mlpHidden, blk.mlpOutput, dMlpOut, rows, blk.mlpHidden, channels)
	dMlpProjected := make([]float32, len(trace.mlpProjected))
	dMLPAlpha := xatgluBackward(dMlpProjected, trace.mlpProjected, dMlpHidden, float64(blk.mlpAlpha), rows, blk.mlpHidden)
	dNorm2 := linearGrad(grads, prefix+".mlp.0.proj.weight", "", trace.norm2, blk.mlpProjection, dMlpProjected, rows, channels, blk.mlpProjected)
	dResidFromNorm2 := layerNormGrad(grads, prefix+".norm2.weight", prefix+".norm2.bias", trace.residual, blk.norm2Weight, dNorm2, rows, channels, cfg.NormEps)
	dResid := append([]float32(nil), dOut...)
	for i := range dResid {
		dResid[i] += dResidFromNorm2[i]
	}
	var dAttnScale float64
	for i, v := range dResid {
		dAttnScale += float64(v) * float64(trace.attention[i])
	}
	dAttn := make([]float32, len(dResid))
	for i, v := range dResid {
		dAttn[i] = v * blk.attentionScale
	}
	dTPA := tpaBackward(grads, prefix, trace.norm1, blk.attention, dAttn, rows, channels, cfg.Heads, channels/cfg.Heads, cfg.QRank, cfg.KVRank, cfg.RopeTheta)
	dTokensFromNorm1 := layerNormGrad(grads, prefix+".norm1.weight", prefix+".norm1.bias", x, blk.norm1Weight, dTPA, rows, channels, cfg.NormEps)
	dX := dResid
	for i := range dX {
		dX[i] += dTokensFromNorm1[i]
	}
	grads.addScalar(prefix+".learned_residual_scale_attn", float32(dAttnScale))
	grads.addScalar(prefix+".mlp.0.alpha", dMLPAlpha)
	grads.addScalar(prefix+".learned_residual_scale_mlp", float32(dMLPScale))
	return dX
}

func (blk *resBlock) backward(m *Model, x, dOut []float32, batch, channels, h, width int, grads Grads) []float32 {
	cfg := &m.Cfg
	prefix := blk.name
	trace := blk.forwardTrace(m, x, batch, channels, h, width)
	var dScale float64
	dH2 := make([]float32, len(dOut))
	for i, v := range dOut {
		dScale += float64(v) * float64(trace.branch[i])
		dH2[i] = v * blk.residualScale
	}
	dA2 := conv2dSame3x3Backward(grads, prefix+".conv2", trace.activated2, blk.conv2Weight, dH2, batch, channels, channels, h, width)
	dN2 := make([]float32, len(trace.norm2))
	hostmath.SiLUBackward(dN2, trace.norm2, dA2)
	dH1 := make([]float32, len(trace.hidden1))
	groupNormBackward(dH1, hostmath.GradientSlot(grads, prefix+".norm2.weight", channels), hostmath.GradientSlot(grads, prefix+".norm2.bias", channels), trace.hidden1, blk.norm2Weight, dN2, batch, channels, h, width, cfg.Groups, cfg.NormEps)
	dA1 := conv2dSame3x3Backward(grads, prefix+".conv1", trace.activated1, blk.conv1Weight, dH1, batch, channels, channels, h, width)
	dN1 := make([]float32, len(trace.norm1))
	hostmath.SiLUBackward(dN1, trace.norm1, dA1)
	dxGN := make([]float32, len(x))
	groupNormBackward(dxGN, hostmath.GradientSlot(grads, prefix+".norm1.weight", channels), hostmath.GradientSlot(grads, prefix+".norm1.bias", channels), x, blk.norm1Weight, dN1, batch, channels, h, width, cfg.Groups, cfg.NormEps)
	dx := append([]float32(nil), dOut...)
	for i := range dx {
		dx[i] += dxGN[i]
	}
	grads.addScalar(prefix+".learned_residual_scale", float32(dScale))
	return dx
}

func (blk *resBlock) forwardTrace(m *Model, x []float32, batch, channels, h, width int) resBlockTrace {
	trace := resBlockTrace{input: x}
	trace.norm1 = make([]float32, len(x))
	groupNormInto(trace.norm1, x, blk.norm1Weight, blk.norm1Bias, batch, channels, h, width, m.Cfg.Groups, m.Cfg.NormEps)
	trace.activated1 = append([]float32(nil), trace.norm1...)
	hostmath.SiLUInPlace(trace.activated1)
	trace.hidden1 = conv2dSame3x3(trace.activated1, blk.conv1Weight, blk.conv1Bias, batch, channels, channels, h, width)
	trace.norm2 = make([]float32, len(trace.hidden1))
	groupNormInto(trace.norm2, trace.hidden1, blk.norm2Weight, blk.norm2Bias, batch, channels, h, width, m.Cfg.Groups, m.Cfg.NormEps)
	trace.activated2 = append([]float32(nil), trace.norm2...)
	hostmath.SiLUInPlace(trace.activated2)
	trace.branch = conv2dSame3x3(trace.activated2, blk.conv2Weight, blk.conv2Bias, batch, channels, channels, h, width)
	trace.output = make([]float32, len(x))
	for index := range trace.output {
		trace.output[index] = x[index] + blk.residualScale*trace.branch[index]
	}
	return trace
}

func (lv *level) backward(m *Model, grads Grads, trace levelTrace, dOut []float32, batch int) ([]float32, error) {
	if len(trace.blockInputs) != len(lv.blocks) {
		return nil, fmt.Errorf("diffusionimage: %s trace inputs=%d blocks=%d", lv.name, len(trace.blockInputs), len(lv.blocks))
	}
	d := append([]float32(nil), dOut...)
	for i := len(lv.blocks) - 1; i >= 0; i-- {
		d = lv.blocks[i].backward(m, trace.blockInputs[i], d, batch, trace.channels, trace.h, trace.width, grads)
	}
	return d, nil
}

func accumulate(base, delta []float32) []float32 {
	if len(delta) == 0 {
		return base
	}
	if len(base) == 0 {
		return delta
	}
	for i := range base {
		base[i] += delta[i]
	}
	return base
}

// backwardFromTrace: full-model VJP mirroring the forward, including both
// uses of the shared learned_middle_residual_scale (decoder skip add and
// middle subtraction).
func (m *Model) backwardFromTrace(trace forwardTrace, dOut []float32, grads Grads) ([]float32, error) {
	const pointwise = 1
	cfg := &m.Cfg
	midScale := m.middleResidual

	d := convTranspose2dStrideBackward(grads, m.final.name, trace.finalInput, m.final.weight, dOut,
		trace.batch, cfg.BaseChannels, cfg.InChannels, trace.finalH, trace.finalW, cfg.PatchSize, cfg.PatchSize)

	encSkipGrads := make([][]float32, cfg.NumLevels)
	for l := cfg.NumLevels - 1; l >= 0; l-- {
		var err error
		if d, err = m.decoders[l].backward(m, grads, trace.decLevels[l], d, trace.batch); err != nil {
			return nil, err
		}
		if l > 0 {
			upTrace := trace.decUps[l-1]
			skipGrad := make([]float32, len(upTrace.residual))
			var dMidScale float64
			for i, v := range d {
				dMidScale += float64(v) * float64(upTrace.residual[i])
				skipGrad[i] = v * midScale
			}
			encSkipGrads[upTrace.residualLevel] = accumulate(encSkipGrads[upTrace.residualLevel], skipGrad)
			dConv := upsampleNearest2xBackward(d, trace.batch, upTrace.nextChannels, upTrace.h, upTrace.width)
			transition := m.decoders[l-1].transition
			d = conv2dValidStrideBackward(grads, transition.name, upTrace.input, transition.weight, dConv,
				trace.batch, upTrace.channels, upTrace.nextChannels, upTrace.h, upTrace.width, pointwise, pointwise)
			grads.addScalar("learned_middle_residual_scale", float32(dMidScale))
		}
	}

	dMiddleResidual := make([]float32, len(trace.middleResidual))
	var dMidScale float64
	for i, v := range d {
		dMidScale -= float64(v) * float64(trace.middleResidual[i])
		dMiddleResidual[i] = -v * midScale
	}
	grads.addScalar("learned_middle_residual_scale", float32(dMidScale))
	for i := cfg.MidBlocks - 1; i >= 0; i-- {
		d = m.middle[i].backward(m, trace.middleInputs[i], d, trace.batch, trace.middleC, trace.middleH, trace.middleW, grads)
	}
	d = accumulate(d, dMiddleResidual)

	for l := cfg.NumLevels - 1; l >= 0; l-- {
		var err error
		if d, err = m.encoders[l].backward(m, grads, trace.encLevels[l], d, trace.batch); err != nil {
			return nil, err
		}
		d = accumulate(d, encSkipGrads[l])
		if l > 0 {
			downTrace := trace.encDowns[l-1]
			dPool := avgPool2xBackward(d, trace.batch, downTrace.nextChannels, downTrace.h, downTrace.width)
			transition := m.encoders[l-1].transition
			d = conv2dValidStrideBackward(grads, transition.name, downTrace.input, transition.weight, dPool,
				trace.batch, downTrace.channels, downTrace.nextChannels, downTrace.h, downTrace.width, pointwise, pointwise)
		}
	}

	dx := conv2dValidStrideBackward(grads, m.patch.name, trace.input, m.patch.weight, d,
		trace.batch, cfg.InChannels, cfg.BaseChannels, trace.imageH, trace.imageW, cfg.PatchSize, cfg.PatchSize)
	return dx, nil
}
