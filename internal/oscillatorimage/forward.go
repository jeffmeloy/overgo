package oscillatorimage

import (
	"context"
	"fmt"
	"math"
	"math/rand"

	"overgo/internal/latentimage"
)

// kuramotoVelocityRowInto: one uncoupled-group velocity row. The coupling
// entry is quantized to f32 after scaling, matching the reference numerics.
func kuramotoVelocityRowInto(vel, theta, omega, coupling []float32, n int, scale float64, zeroDiagonal bool, sinT, cosT []float64) {
	for j := range n {
		sinT[j], cosT[j] = math.Sincos(float64(theta[j]))
	}
	for i := range n {
		krow := coupling[i*n : (i+1)*n]
		var ws, wc float64
		for j := range n {
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
	for bi := range b {
		stateRow, outRow := state[bi*tot:(bi+1)*tot], out[bi*tot:(bi+1)*tot]
		kuramotoVelocityRowInto(outRow[:n], stateRow[:n], omega, kMat, n, kScale, true, sinMain, cosMain)
		kuramotoVelocityRowInto(outRow[n:], stateRow[n:], omegaCond, kCondMat, nCond, kCondScale, true, sinCond, cosCond)
		for i := range n {
			var ds, dc float64
			base := bi*n*nCond + i*nCond
			for m := range nCond {
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
	for bi := range b {
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
	for bi := range b {
		for co := range cout {
			ob := (bi*cout + co) * plane
			fill := 0.0
			if bias != nil {
				fill = float64(bias[co])
			}
			for i := range h {
				for j := range w {
					acc := fill
					for ci := range cin {
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
	for bi := range b {
		for ci := range c {
			ib := (bi*c + ci) * h * w
			ob := (bi*c + ci) * oh * ow
			for i := range oh {
				for j := range ow {
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

type phasePlan struct {
	state []float32
	drive []float32
}

func (m *Model) prepare(ctx context.Context, request Request) (phasePlan, error) {
	if err := ctx.Err(); err != nil {
		return phasePlan{}, err
	}
	if m == nil {
		return phasePlan{}, fmt.Errorf("oscillatorimage: model is unavailable")
	}
	cfg := m.Cfg
	if request.Class < 0 || request.Class >= cfg.NClasses {
		return phasePlan{}, fmt.Errorf("oscillatorimage: class %d out of [0,%d)", request.Class, cfg.NClasses)
	}
	rng := rand.New(rand.NewSource(request.Seed))
	state := make([]float32, cfg.N+cfg.NCond)
	for index := range state {
		state[index] = float32((rng.Float64()*2 - 1) * math.Pi)
	}
	start := request.Class * cfg.N * cfg.NCond
	return phasePlan{state: state, drive: m.Drive[start : start+cfg.N*cfg.NCond]}, nil
}

func (m *Model) integrate(ctx context.Context, plan phasePlan) ([]float32, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	cfg := m.Cfg
	tot := cfg.N + cfg.NCond
	// Batch is carried by the plan's state length (b samples of tot phases);
	// prepare() builds b=1, but the staged path stays batch-general so a
	// multi-sample plan integrates every sample (parity: un0 generator golden).
	b := len(plan.state) / tot
	state := append([]float32(nil), plan.state...)
	vel := make([]float32, len(state))
	trig := make([]float64, 2*tot)
	for range cfg.NumSteps {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		conditionalKuramotoForwardInto(
			vel, state, m.Omega, m.OmegaCond, m.K, m.KCond, plan.drive,
			b, cfg.N, cfg.NCond, cfg.KScale, cfg.KCondScale, cfg.KDriveScale, trig[:tot], trig[tot:],
		)
		for i := range state {
			state[i] = float32(float64(state[i]) + cfg.Dt*float64(vel[i]))
		}
	}
	return readoutTransform(state, b, cfg.N, tot, 0, cfg.Relativization, cfg.Encoding), nil
}

func (m *Model) decodePlanar(features []float32) (planarImage, error) {
	cfg := m.Cfg
	// Batch derived from the readout width (b samples of in_ch*in_h*in_w).
	b := len(features) / (cfg.InChannels * cfg.InH * cfg.InW)
	pixels, _ := decoderForwardTrace(
		features, m.Blocks, m.ToOutW, m.ToOutB, b,
		cfg.InChannels, cfg.InH, cfg.InW, cfg.OutChannels, m.Slope, cfg.TanhOut,
	)
	return planarImage{Pixels: pixels, Channels: cfg.OutChannels, Height: cfg.OutH(), Width: cfg.OutW()}, nil
}

func (m *Model) decode(ctx context.Context, features []float32) (latentimage.EncodedImage, error) {
	if err := ctx.Err(); err != nil {
		return latentimage.EncodedImage{}, err
	}
	image, err := m.decodePlanar(features)
	if err != nil {
		return latentimage.EncodedImage{}, err
	}
	return latentimage.EncodePlanarPNG(image.Pixels, image.Height, image.Width)
}
