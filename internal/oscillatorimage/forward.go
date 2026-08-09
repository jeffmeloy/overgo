package oscillatorimage

import (
	"fmt"
	"math"
	"math/rand"
)

// kuramotoVelocityRowInto: one uncoupled-group velocity row. The coupling
// entry is quantized to f32 after scaling, matching the reference numerics.
func kuramotoVelocityRowInto(vel, theta, omega, coupling []float32, n int, scale float64, zeroDiagonal bool, sinT, cosT []float64) {
	for j := 0; j < n; j++ {
		sinT[j], cosT[j] = math.Sincos(float64(theta[j]))
	}
	for i := 0; i < n; i++ {
		krow := coupling[i*n : (i+1)*n]
		var ws, wc float64
		for j := 0; j < n; j++ {
			if zeroDiagonal && i == j {
				continue
			}
			k := float64(float32(float64(krow[j]) * scale))
			ws += k * sinT[j]
			wc += k * cosT[j]
		}
		vel[i] = float32(float64(omega[i]) + cosT[i]*ws - sinT[i]*wc)
	}
}

// conditionalKuramotoForwardInto: main + condition group velocities plus the
// per-batch class drive coupling main oscillators to condition phases.
func conditionalKuramotoForwardInto(out, state, omega, omegaCond, kMat, kCondMat, drive []float32, b, n, nCond int, kScale, kCondScale, kDriveScale float64, sinT, cosT []float64) {
	tot := n + nCond
	sinMain, sinCond := sinT[:n], sinT[n:]
	cosMain, cosCond := cosT[:n], cosT[n:]
	for bi := 0; bi < b; bi++ {
		stateRow, outRow := state[bi*tot:(bi+1)*tot], out[bi*tot:(bi+1)*tot]
		kuramotoVelocityRowInto(outRow[:n], stateRow[:n], omega, kMat, n, kScale, true, sinMain, cosMain)
		kuramotoVelocityRowInto(outRow[n:], stateRow[n:], omegaCond, kCondMat, nCond, kCondScale, true, sinCond, cosCond)
		for i := 0; i < n; i++ {
			var ds, dc float64
			base := bi*n*nCond + i*nCond
			for m := 0; m < nCond; m++ {
				dv := float64(drive[base+m]) * kDriveScale
				ds += dv * sinCond[m]
				dc += dv * cosCond[m]
			}
			outRow[i] = float32(float64(outRow[i]) + cosMain[i]*ds - sinMain[i]*dc)
		}
	}
}

// readoutTransform: relativize phase rows; encode sin or [sin,cos].
func readoutTransform(phases []float32, b, n, stride, offset int, relativization, encoding string) []float32 {
	outWidth := n
	if encoding == "sin_cos" {
		outWidth = 2 * n
	}
	out := make([]float32, b*outWidth)
	for bi := 0; bi < b; bi++ {
		row := phases[bi*stride+offset:][:n]
		var mean float64
		if relativization == "mean_relative" {
			for _, value := range row {
				mean += float64(value)
			}
			mean /= float64(n)
		}
		ref := float32(0)
		if relativization == "ref_oscillator" {
			ref = row[0]
		}
		for j, value := range row {
			if relativization == "mean_relative" {
				value = float32(float64(value) - mean)
			} else {
				value -= ref
			}
			switch encoding {
			case "sin":
				out[bi*outWidth+j] = float32(math.Sin(float64(value)))
			case "sin_cos":
				s, c := math.Sincos(float64(value))
				out[bi*outWidth+j], out[bi*outWidth+n+j] = float32(s), float32(c)
			default:
				out[bi*outWidth+j] = value
			}
		}
	}
	return out
}

// conv2dSame3x3: NCHW 3x3 stride-1 same-padding; f64 accumulation.
func conv2dSame3x3(x, weight, bias []float32, b, cin, cout, h, w int) []float32 {
	out := make([]float32, b*cout*h*w)
	plane := h * w
	for bi := 0; bi < b; bi++ {
		for co := 0; co < cout; co++ {
			ob := (bi*cout + co) * plane
			fill := 0.0
			if bias != nil {
				fill = float64(bias[co])
			}
			for i := 0; i < h; i++ {
				for j := 0; j < w; j++ {
					acc := fill
					for ci := 0; ci < cin; ci++ {
						xb := (bi*cin + ci) * plane
						wc := (co*cin + ci) * convTaps
						for di := -1; di <= 1; di++ {
							si := i + di
							if si < 0 || si >= h {
								continue
							}
							for dj := -1; dj <= 1; dj++ {
								sj := j + dj
								if sj < 0 || sj >= w {
									continue
								}
								acc += float64(x[xb+si*w+sj]) * float64(weight[wc+(di+1)*convKernel+(dj+1)])
							}
						}
					}
					out[ob+i*w+j] = float32(acc)
				}
			}
		}
	}
	return out
}

// upsampleNearest2x: nearest-neighbor 2x NCHW upsample.
func upsampleNearest2x(x []float32, b, c, h, w int) []float32 {
	oh, ow := upsample*h, upsample*w
	out := make([]float32, b*c*oh*ow)
	for bi := 0; bi < b; bi++ {
		for ci := 0; ci < c; ci++ {
			ib := (bi*c + ci) * h * w
			ob := (bi*c + ci) * oh * ow
			for i := 0; i < oh; i++ {
				for j := 0; j < ow; j++ {
					out[ob+i*ow+j] = x[ib+(i/upsample)*w+(j/upsample)]
				}
			}
		}
	}
	return out
}

func leakyReLU(x []float32, slope float64) {
	s := float32(slope)
	for i, v := range x {
		if v < 0 {
			x[i] = s * v
		}
	}
}

// resizeConvBlock: upsample 2x -> conv3x3 -> leaky -> conv3x3 -> leaky.
func resizeConvBlock(x, w1, b1, w2, b2 []float32, b, cin, cout, h, w int, slope float64) []float32 {
	up := upsampleNearest2x(x, b, cin, h, w)
	oh, ow := upsample*h, upsample*w
	c1 := conv2dSame3x3(up, w1, b1, b, cin, cout, oh, ow)
	leakyReLU(c1, slope)
	c2 := conv2dSame3x3(c1, w2, b2, b, cout, cout, oh, ow)
	leakyReLU(c2, slope)
	return c2
}

// decoderTape: forward intermediates the backward recomputes from.
type decoderTape struct {
	rows   [][]float32
	dims   [][3]int
	output []float32
}

func decoderForwardTrace(features []float32, blocks []DecoderBlock, toOutW, toOutB []float32, b, inChannels, inH, inW, outChannels int, slope float64, tanhOut bool) ([]float32, decoderTape) {
	tape := decoderTape{rows: make([][]float32, len(blocks)+1), dims: make([][3]int, len(blocks))}
	tape.rows[0] = features
	cin, h, w := inChannels, inH, inW
	for index, block := range blocks {
		tape.dims[index] = [3]int{cin, h, w}
		tape.rows[index+1] = resizeConvBlock(tape.rows[index], block.W1, block.B1, block.W2, block.B2, b, cin, block.Cout, h, w, slope)
		cin, h, w = block.Cout, upsample*h, upsample*w
	}
	tape.output = conv2dSame3x3(tape.rows[len(blocks)], toOutW, toOutB, b, cin, outChannels, h, w)
	if tanhOut {
		for i, v := range tape.output {
			tape.output[i] = float32(math.Tanh(float64(v)))
		}
	}
	return tape.output, tape
}

// generateImage: Euler-integrate the coupled phases, read out, decode.
func generateImage(initialState, omega, omegaCond, kMat, kCondMat, drive []float32,
	blocks []DecoderBlock, toOutW, toOutB []float32,
	b, n, nCond, numSteps int, dt, kScale, kCondScale, kDriveScale float64,
	inChannels, inH, inW, outChannels int, slope float64, relativization, encoding string, tanhOut bool) []float32 {
	tot := n + nCond
	state := make([]float32, len(initialState))
	copy(state, initialState)
	vel := make([]float32, len(state))
	trig := make([]float64, 2*tot)
	for s := 0; s < numSteps; s++ {
		conditionalKuramotoForwardInto(vel, state, omega, omegaCond, kMat, kCondMat, drive, b, n, nCond, kScale, kCondScale, kDriveScale, trig[:tot], trig[tot:])
		for i := range state {
			state[i] = float32(float64(state[i]) + dt*float64(vel[i]))
		}
	}
	features := readoutTransform(state, b, n, tot, 0, relativization, encoding)
	output, _ := decoderForwardTrace(features, blocks, toOutW, toOutB, b, inChannels, inH, inW, outChannels, slope, tanhOut)
	return output
}

// Generate samples one seeded image for classID. Initial phases are uniform
// in [-pi,pi) from the artifact's own Go-rand sampling convention; output is
// flat [out_ch, OutH, OutW], tanh-bounded when configured.
func (m *Model) Generate(classID int, seed int64) ([]float32, error) {
	cfg := m.Cfg
	if classID < 0 || classID >= cfg.NClasses {
		return nil, fmt.Errorf("oscillatorimage: class %d out of [0,%d)", classID, cfg.NClasses)
	}
	tot := cfg.N + cfg.NCond
	rng := rand.New(rand.NewSource(seed))
	init := make([]float32, tot)
	for i := range init {
		init[i] = float32((rng.Float64()*2 - 1) * math.Pi)
	}
	drive := m.Drive[classID*cfg.N*cfg.NCond : (classID+1)*cfg.N*cfg.NCond]
	return generateImage(init, m.Omega, m.OmegaCond, m.K, m.KCond, drive,
		m.Blocks, m.ToOutW, m.ToOutB,
		1, cfg.N, cfg.NCond, cfg.NumSteps, cfg.Dt, cfg.KScale, cfg.KCondScale, cfg.KDriveScale,
		cfg.InChannels, cfg.InH, cfg.InW, cfg.OutChannels, m.Slope, cfg.Relativization, cfg.Encoding, cfg.TanhOut), nil
}
