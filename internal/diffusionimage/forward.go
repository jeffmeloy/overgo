package diffusionimage

import (
	"fmt"
	"math"
	"math/rand"

	"overgo/internal/hostmath"
)

// conv2dValidStride: NCHW Conv2d, square kernel, no padding — patch
// embedding (k=stride=patch) and 1x1 level transitions.
func conv2dValidStride(x, weight, bias []float32, b, cin, cout, h, w, kernel, stride int) ([]float32, int, int) {
	oh := (h-kernel)/stride + 1
	ow := (w-kernel)/stride + 1
	out := make([]float32, b*cout*oh*ow)
	kernelElems := kernel * kernel
	hostmath.ParallelRangeF64(b*cout, oh*ow*cin*kernelElems, func(lo, hi int) {
		for owner := lo; owner < hi; owner++ {
			bi, co := owner/cout, owner%cout
			wco := co * cin * kernelElems
			for oy := 0; oy < oh; oy++ {
				for ox := 0; ox < ow; ox++ {
					var acc float64
					if bias != nil {
						acc = float64(bias[co])
					}
					for ci := 0; ci < cin; ci++ {
						xb := (bi*cin + ci) * h * w
						wc := wco + ci*kernelElems
						iy, ix := oy*stride, ox*stride
						for ky := 0; ky < kernel; ky++ {
							for kx := 0; kx < kernel; kx++ {
								acc += float64(x[xb+(iy+ky)*w+ix+kx]) * float64(weight[wc+ky*kernel+kx])
							}
						}
					}
					out[(bi*cout+co)*oh*ow+oy*ow+ox] = float32(acc)
				}
			}
		}
	})
	return out, oh, ow
}

// convTranspose2dStride: NCHW ConvTranspose2d, square kernel, no padding;
// weight is torch layout [cin, cout, k, k]. Final unpatchify projection.
func convTranspose2dStride(x, weight, bias []float32, b, cin, cout, h, w, kernel, stride int) ([]float32, int, int) {
	oh := (h-1)*stride + kernel
	ow := (w-1)*stride + kernel
	out := make([]float32, b*cout*oh*ow)
	kernelElems := kernel * kernel
	// Split on (batch, out-channel): each owner accumulates its own plane.
	hostmath.ParallelRangeF64(b*cout, h*w*cin*kernelElems, func(lo, hi int) {
		for owner := lo; owner < hi; owner++ {
			bi, co := owner/cout, owner%cout
			ob := (bi*cout + co) * oh * ow
			for ci := 0; ci < cin; ci++ {
				wc := (ci*cout + co) * kernelElems
				for iy := 0; iy < h; iy++ {
					for ix := 0; ix < w; ix++ {
						xv := float64(x[(bi*cin+ci)*h*w+iy*w+ix])
						oy, ox := iy*stride, ix*stride
						for ky := 0; ky < kernel; ky++ {
							for kx := 0; kx < kernel; kx++ {
								out[ob+(oy+ky)*ow+ox+kx] += float32(xv * float64(weight[wc+ky*kernel+kx]))
							}
						}
					}
				}
			}
			if bias != nil {
				row := out[ob : ob+oh*ow]
				for i := range row {
					row[i] += bias[co]
				}
			}
		}
	})
	return out, oh, ow
}

const conv3x3Kernel, conv3x3Radius = 3, 1

// conv2dSame3x3: NCHW 3x3 stride-1 same-padding, f64 plane accumulator;
// parallel over (batch, out-channel) planes.
func conv2dSame3x3(x, weight, bias []float32, b, cin, cout, h, w int) []float32 {
	out := make([]float32, b*cout*h*w)
	kernelElems := conv3x3Kernel * conv3x3Kernel
	plane := h * w
	hostmath.ParallelRangeF64(b*cout, plane*cin*kernelElems, func(lo, hi int) {
		acc := make([]float64, plane)
		for owner := lo; owner < hi; owner++ {
			bi, co := owner/cout, owner%cout
			fill := 0.0
			if bias != nil {
				fill = float64(bias[co])
			}
			for i := range acc {
				acc[i] = fill
			}
			wco := co * cin * kernelElems
			for ci := 0; ci < cin; ci++ {
				xb := (bi*cin + ci) * plane
				wc := wco + ci*kernelElems
				for di := -conv3x3Radius; di <= conv3x3Radius; di++ {
					iLo, iHi := 0, h
					if di < 0 {
						iLo = -di
					} else if di > 0 {
						iHi = h - di
					}
					for dj := -conv3x3Radius; dj <= conv3x3Radius; dj++ {
						wv := float64(weight[wc+(di+conv3x3Radius)*conv3x3Kernel+(dj+conv3x3Radius)])
						jLo, jHi := 0, w
						if dj < 0 {
							jLo = -dj
						} else if dj > 0 {
							jHi = w - dj
						}
						if jLo >= jHi {
							continue
						}
						for i := iLo; i < iHi; i++ {
							dst := acc[i*w+jLo : i*w+jHi]
							src := x[xb+(i+di)*w+jLo+dj : xb+(i+di)*w+jHi+dj]
							for k, v := range src {
								dst[k] += float64(v) * wv
							}
						}
					}
				}
			}
			ob := (bi*cout + co) * plane
			for i, v := range acc {
				out[ob+i] = float32(v)
			}
		}
	})
	return out
}

// avgPool2x: 2x2 stride-2 average pool.
func avgPool2x(x []float32, b, c, h, w int) []float32 {
	oh, ow := h/2, w/2
	out := make([]float32, b*c*oh*ow)
	for bi := 0; bi < b; bi++ {
		for ci := 0; ci < c; ci++ {
			inBase := (bi*c + ci) * h * w
			outBase := (bi*c + ci) * oh * ow
			for i := 0; i < oh; i++ {
				for j := 0; j < ow; j++ {
					p := inBase + 2*i*w + 2*j
					out[outBase+i*ow+j] = (x[p] + x[p+1] + x[p+w] + x[p+w+1]) * 0.25
				}
			}
		}
	}
	return out
}

// upsampleNearest2x: nearest-neighbor 2x upsample.
func upsampleNearest2x(x []float32, b, c, h, w int) []float32 {
	oh, ow := 2*h, 2*w
	out := make([]float32, b*c*oh*ow)
	for bi := 0; bi < b; bi++ {
		for ci := 0; ci < c; ci++ {
			ib := (bi*c + ci) * h * w
			ob := (bi*c + ci) * oh * ow
			for i := 0; i < oh; i++ {
				for j := 0; j < ow; j++ {
					out[ob+i*ow+j] = x[ib+(i/2)*w+(j/2)]
				}
			}
		}
	}
	return out
}

// groupNormInto: torch GroupNorm over NCHW (f64 stats, biased variance).
func groupNormInto(out, x, weight, bias []float32, n, c, h, w, groups int, eps float64) {
	channelsPerGroup := c / groups
	hw := h * w
	groupElems := float64(channelsPerGroup * hw)
	for ni := 0; ni < n; ni++ {
		for g := 0; g < groups; g++ {
			var sum, sumsq float64
			for ci := g * channelsPerGroup; ci < (g+1)*channelsPerGroup; ci++ {
				base := (ni*c + ci) * hw
				for i := 0; i < hw; i++ {
					v := float64(x[base+i])
					sum += v
					sumsq += v * v
				}
			}
			mean := sum / groupElems
			inv := 1 / math.Sqrt(sumsq/groupElems-mean*mean+eps)
			for ci := g * channelsPerGroup; ci < (g+1)*channelsPerGroup; ci++ {
				wc, bc := float64(weight[ci]), float64(bias[ci])
				base := (ni*c + ci) * hw
				for i := 0; i < hw; i++ {
					out[base+i] = float32((float64(x[base+i])-mean)*inv*wc + bc)
				}
			}
		}
	}
}

const (
	xatgluHalfPi = math.Pi / 2
	xatgluInvPi  = 1 / math.Pi
)

// xatgluGateInto: arctan-gated GLU with learned alpha range expansion —
// dst = (gate*(1+2a)-a) * value over [gate|value] projected rows.
func xatgluGateInto(dst, projected []float32, alpha float64, rows, outDim int) {
	stride := 2 * outDim
	for r := 0; r < rows; r++ {
		gatePath := projected[r*stride : r*stride+outDim]
		valuePath := projected[r*stride+outDim : r*stride+stride]
		out := dst[r*outDim : r*outDim+outDim]
		for j := 0; j < outDim; j++ {
			gate := (math.Atan(float64(gatePath[j])) + xatgluHalfPi) * xatgluInvPi
			out[j] = float32((gate*(1+2*alpha) - alpha) * float64(valuePath[j]))
		}
	}
}

// cpFactorContractInto: per-token rank contraction q = (A x B)/rank.
func cpFactorContractInto(dst, aFactor, bFactor []float32, tokens, heads, rank, headDim int) {
	invRank := 1 / float64(rank)
	for t := 0; t < tokens; t++ {
		aBase := t * heads * rank
		bBase := t * rank * headDim
		for h := 0; h < heads; h++ {
			aRow := aFactor[aBase+h*rank : aBase+(h+1)*rank]
			out := dst[(t*heads+h)*headDim : (t*heads+h+1)*headDim]
			for d := 0; d < headDim; d++ {
				var sum float64
				for r := 0; r < rank; r++ {
					sum += float64(aRow[r]) * float64(bFactor[bBase+r*headDim+d])
				}
				out[d] = float32(sum * invRank)
			}
		}
	}
}

// buildRopeTables: half-split cos/sin at NEGATED sin — the vendor rotation
// (y1=x1*cos+x2*sin; y2=-x1*sin+x2*cos) expressed through the shared
// rotate-half apply below.
func buildRopeTables(seq, headDim int, theta float64) (cos, sin []float32) {
	half := headDim / 2
	cos = make([]float32, seq*headDim)
	sin = make([]float32, seq*headDim)
	for i := 0; i < half; i++ {
		inv := 1 / math.Pow(theta, float64(2*i)/float64(headDim))
		for p := 0; p < seq; p++ {
			c, s := float32(math.Cos(float64(p)*inv)), float32(-math.Sin(float64(p)*inv))
			base := p * headDim
			cos[base+i], cos[base+half+i] = c, c
			sin[base+i], sin[base+half+i] = s, s
		}
	}
	return cos, sin
}

// applyRope: rotate-half in place over [seq, heads, headDim].
func applyRope(x, cos, sin []float32, seq, nHeads, headDim int) {
	half := headDim / 2
	for p := 0; p < seq; p++ {
		cp := cos[p*headDim : (p+1)*headDim]
		sp := sin[p*headDim : (p+1)*headDim]
		for h := 0; h < nHeads; h++ {
			vec := x[(p*nHeads+h)*headDim : (p*nHeads+h+1)*headDim]
			for i := 0; i < half; i++ {
				j := i + half
				a, b := vec[i], vec[j]
				vec[i] = a*cp[i] - b*sp[i]
				vec[j] = b*cp[j] + a*sp[j]
			}
		}
	}
}

// tpaForward: CP-factored q/k/v, vendor RoPE, full bidirectional SDPA,
// xATGLU output projection. out is [seq, nEmbd] tokens.
func tpaForward(out, x []float32, w tpaWeights, seq, nEmbd, nHead, headDim, qRank, kvRank int, ropeTheta float64) {
	q := make([]float32, seq*nHead*headDim)
	k := make([]float32, seq*nHead*headDim)
	v := make([]float32, seq*nHead*headDim)
	maxRank := max(qRank, kvRank)
	aFactor := make([]float32, seq*nHead*maxRank)
	bFactor := make([]float32, seq*maxRank*headDim)
	contract := func(dst, wa, wb []float32, rank int) {
		a := aFactor[:seq*nHead*rank]
		bf := bFactor[:seq*rank*headDim]
		hostmath.LinearF64(a, x, wa, nil, seq, nEmbd, nHead*rank)
		hostmath.LinearF64(bf, x, wb, nil, seq, nEmbd, rank*headDim)
		cpFactorContractInto(dst, a, bf, seq, nHead, rank, headDim)
	}
	contract(q, w.WAq, w.WBq, qRank)
	contract(k, w.WAk, w.WBk, kvRank)
	contract(v, w.WAv, w.WBv, kvRank)

	cos, sin := buildRopeTables(seq, headDim, ropeTheta)
	applyRope(q, cos, sin, seq, nHead, headDim)
	applyRope(k, cos, sin, seq, nHead, headDim)

	// hostmath's bidirectional core takes score scale folded into q.
	scale := float32(1 / math.Sqrt(float64(headDim)))
	for i := range q {
		q[i] *= scale
	}
	attn := make([]float32, seq*nHead*headDim)
	hostmath.MaskedBidirectionalAttention(attn, q, k, v, seq, seq, nHead, nHead, headDim, nil)

	projected := make([]float32, 2*seq*nEmbd)
	hostmath.LinearF64(projected, attn, w.Woproj, w.Boproj, seq, nHead*headDim, 2*nEmbd)
	xatgluGateInto(out, projected, w.AlphaO, seq, nEmbd)
}

// tokensToImageNCHW: vendor raw reshape (permute(1,2,0).reshape(b,c,h,w));
// golden-locked, do not "fix".
func tokensToImageNCHW(tokens []float32, b, c, h, w int) []float32 {
	seq := h * w
	out := make([]float32, len(tokens))
	pos := 0
	for tkn := 0; tkn < seq; tkn++ {
		for ch := 0; ch < c; ch++ {
			for bi := 0; bi < b; bi++ {
				out[pos] = tokens[(bi*seq+tkn)*c+ch]
				pos++
			}
		}
	}
	return out
}

func imageNCHWToTokens(image []float32, b, c, h, w int) []float32 {
	seq := h * w
	out := make([]float32, len(image))
	pos := 0
	for tkn := 0; tkn < seq; tkn++ {
		for ch := 0; ch < c; ch++ {
			for bi := 0; bi < b; bi++ {
				out[(bi*seq+tkn)*c+ch] = image[pos]
				pos++
			}
		}
	}
	return out
}

// addScaledInto: out = base + scale*branch.
func addScaledInto(out, base, branch []float32, scale float32) {
	for i := range out {
		out[i] = base[i] + scale*branch[i]
	}
}

func (blk *resBlock) forward(m *Model, x []float32, b, c, h, w int) []float32 {
	cfg := &m.Cfg
	buf := make([]float32, len(x))
	groupNormInto(buf, x, blk.norm1Weight, blk.norm1Bias, b, c, h, w, cfg.Groups, cfg.NormEps)
	hostmath.SiLUInPlace(buf)
	h1 := conv2dSame3x3(buf, blk.conv1Weight, blk.conv1Bias, b, c, c, h, w)
	groupNormInto(buf, h1, blk.norm2Weight, blk.norm2Bias, b, c, h, w, cfg.Groups, cfg.NormEps)
	hostmath.SiLUInPlace(buf)
	out := conv2dSame3x3(buf, blk.conv2Weight, blk.conv2Bias, b, c, c, h, w)
	addScaledInto(out, x, out, blk.residualScale)
	return out
}

// attnTrace: retained transformer-block activations for the VJP.
type attnTrace struct {
	norm1, attention, residual, norm2          []float32
	mlpProjected, mlpHidden, mlpOutput, output []float32
}

func (blk *attnBlock) forward(m *Model, x []float32, b, c, h, w int) []float32 {
	trace := blk.forwardTrace(m, x, b, c, h, w, false)
	return tokensToImageNCHW(trace.output, b, c, h, w)
}

// forwardTrace runs the block on the vendor token view ([b*h*w, c] raw
// reinterpretation of the NCHW slab; see tokensToImageNCHW).
func (blk *attnBlock) forwardTrace(m *Model, x []float32, b, c, h, w int, retain bool) attnTrace {
	cfg := &m.Cfg
	rows := b * h * w
	tokens := x
	norm1 := make([]float32, len(tokens))
	residual := make([]float32, len(tokens))
	norm2 := make([]float32, len(tokens))
	out := make([]float32, len(tokens))
	var attention, mlpOutput []float32
	hostmath.LayerNormInto(norm1, tokens, blk.norm1Weight, blk.norm1Bias, rows, c, cfg.NormEps)
	attentionOut := residual
	if retain {
		attention = make([]float32, len(tokens))
		attentionOut = attention
	}
	tpaForward(attentionOut, norm1, blk.attention, rows, c, cfg.Heads, c/cfg.Heads, cfg.QRank, cfg.KVRank, cfg.RopeTheta)
	addScaledInto(residual, tokens, attentionOut, blk.attentionScale)
	hostmath.LayerNormInto(norm2, residual, blk.norm2Weight, blk.norm2Bias, rows, c, cfg.NormEps)
	mlpProjected := make([]float32, rows*blk.mlpProjected)
	mlpHidden := make([]float32, rows*blk.mlpHidden)
	hostmath.LinearF64(mlpProjected, norm2, blk.mlpProjection, nil, rows, c, blk.mlpProjected)
	xatgluGateInto(mlpHidden, mlpProjected, float64(blk.mlpAlpha), rows, blk.mlpHidden)
	mlpTarget := out
	if retain {
		mlpOutput = make([]float32, len(tokens))
		mlpTarget = mlpOutput
	}
	hostmath.LinearF64(mlpTarget, mlpHidden, blk.mlpOutput, nil, rows, blk.mlpHidden, c)
	addScaledInto(out, residual, mlpTarget, blk.mlpScale)
	return attnTrace{
		norm1: norm1, attention: attention, residual: residual, norm2: norm2,
		mlpProjected: mlpProjected, mlpHidden: mlpHidden, mlpOutput: mlpOutput, output: out,
	}
}

// Forward: NCHW image to same-shape flow velocity.
func (m *Model) Forward(x []float32, b, imgH, imgW int) ([]float32, error) {
	out, _, err := m.forward(x, b, imgH, imgW, false)
	return out, err
}

func (m *Model) forward(x []float32, b, imgH, imgW int, retain bool) ([]float32, forwardTrace, error) {
	const pointwise = 1
	var trace forwardTrace
	cfg := &m.Cfg
	if len(x) != b*cfg.InChannels*imgH*imgW {
		return nil, trace, fmt.Errorf("diffusionimage forward: input len %d != %d", len(x), b*cfg.InChannels*imgH*imgW)
	}
	cur, h, width := conv2dValidStride(x, m.patch.weight, m.patch.bias, b, cfg.InChannels, cfg.BaseChannels, imgH, imgW, cfg.PatchSize, cfg.PatchSize)

	channels := cfg.BaseChannels
	currRes := cur
	residuals := make([][]float32, 0, cfg.NumLevels-1)

	for l := range m.encoders {
		encoder := &m.encoders[l]
		levelTrace := levelTrace{input: cur, channels: channels, h: h, width: width}
		for _, blk := range encoder.blocks {
			if retain {
				levelTrace.blockInputs = append(levelTrace.blockInputs, cur)
			}
			cur = blk.forward(m, cur, b, channels, h, width)
		}
		if retain {
			trace.encLevels = append(trace.encLevels, levelTrace)
		}
		if l < len(m.encoders)-1 {
			residuals = append(residuals, currRes)
			nextChannels := channels * 2
			if retain {
				trace.encDowns = append(trace.encDowns, resizeTrace{input: cur, channels: channels, nextChannels: nextChannels, h: h, width: width})
			}
			cur, _, _ = conv2dValidStride(cur, encoder.transition.weight, encoder.transition.bias, b, channels, nextChannels, h, width, pointwise, pointwise)
			cur = avgPool2x(cur, b, nextChannels, h, width)
			channels, h, width = nextChannels, h/2, width/2
			currRes = cur
		}
	}

	trace.middleC, trace.middleH, trace.middleW = channels, h, width
	for _, blk := range m.middle {
		if retain {
			trace.middleInputs = append(trace.middleInputs, cur)
		}
		cur = blk.forward(m, cur, b, channels, h, width)
	}
	if retain {
		trace.middleResidual = currRes
	}
	midScale := m.middleResidual
	for i := range cur {
		cur[i] -= currRes[i] * midScale
	}

	for l := range m.decoders {
		decoder := &m.decoders[l]
		levelTrace := levelTrace{input: cur, channels: channels, h: h, width: width}
		for _, blk := range decoder.blocks {
			if retain {
				levelTrace.blockInputs = append(levelTrace.blockInputs, cur)
			}
			cur = blk.forward(m, cur, b, channels, h, width)
		}
		if retain {
			trace.decLevels = append(trace.decLevels, levelTrace)
		}
		if l < len(m.decoders)-1 {
			nextChannels := channels / 2
			residual := residuals[len(residuals)-1]
			residuals = residuals[:len(residuals)-1]
			if retain {
				trace.decUps = append(trace.decUps, resizeTrace{input: cur, residual: residual, residualLevel: len(m.decoders) - 2 - l, channels: channels, nextChannels: nextChannels, h: h, width: width})
			}
			cur, _, _ = conv2dValidStride(cur, decoder.transition.weight, decoder.transition.bias, b, channels, nextChannels, h, width, pointwise, pointwise)
			cur = upsampleNearest2x(cur, b, nextChannels, h, width)
			channels, h, width = nextChannels, h*2, width*2
			for i := range cur {
				cur[i] += residual[i] * midScale
			}
		}
	}

	if retain {
		trace.input = x
		trace.batch = b
		trace.imageH, trace.imageW = imgH, imgW
		trace.finalInput, trace.finalH, trace.finalW = cur, h, width
	}
	out, oh, ow := convTranspose2dStride(cur, m.final.weight, m.final.bias, b, cfg.BaseChannels, cfg.InChannels, h, width, cfg.PatchSize, cfg.PatchSize)
	if oh != imgH || ow != imgW {
		return nil, trace, fmt.Errorf("diffusionimage forward: output %dx%d != %dx%d", oh, ow, imgH, imgW)
	}
	return out, trace, nil
}

// Sample: seeded Gaussian start, Euler-integrated autonomous flow — the
// reference generate profile's execution (SampleContext; the model ignores
// time conditioning, vendor forward(x, t=None) discards t).
func (m *Model) Sample(b, imgH, imgW, steps int, seed int64) ([]float32, error) {
	if steps <= 0 {
		return nil, fmt.Errorf("diffusionimage sample: steps %d", steps)
	}
	rng := rand.New(rand.NewSource(seed))
	x := make([]float32, b*m.Cfg.InChannels*imgH*imgW)
	for i := range x {
		x[i] = float32(rng.NormFloat64())
	}
	dt := float32(1) / float32(steps)
	for s := 0; s < steps; s++ {
		v, err := m.Forward(x, b, imgH, imgW)
		if err != nil {
			return nil, err
		}
		for i := range x {
			x[i] += v[i] * dt
		}
	}
	return x, nil
}
