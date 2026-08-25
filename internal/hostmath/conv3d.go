// 3-D convolution primitives (Wan-family causal video codec stacks).
// Channel-major [c][T][H][W] layouts, PyTorch weight order [cOut][cIn][kT][kH][kW],
// f64 accumulation stored f32. Output channels are independent and each keeps
// its serial accumulation order, so the parallel channel split is
// bit-identical to the serial loop.
package hostmath

import (
	"fmt"
	"math"
)

// Conv3DShape: one convolution call's geometry. PadT is the symmetric-pad
// parameter; the causal form places all 2*PadT temporal padding on the left.
type Conv3DShape struct {
	CIn, COut                 int
	InT, InH, InW             int
	KT, KH, KW                int
	PadT, PadH, PadW          int
	StrideT, StrideH, StrideW int
}

// OutputDims: standard conv output extents; temporal uses the full 2*PadT.
func (s Conv3DShape) OutputDims() (t, h, w int, err error) {
	if s.CIn <= 0 || s.COut <= 0 || s.InT <= 0 || s.InH <= 0 || s.InW <= 0 || s.KT <= 0 || s.KH <= 0 || s.KW <= 0 {
		return 0, 0, 0, fmt.Errorf("conv3d: bad shape %+v", s)
	}
	if s.StrideT <= 0 || s.StrideH <= 0 || s.StrideW <= 0 {
		return 0, 0, 0, fmt.Errorf("conv3d: bad stride %+v", s)
	}
	t = (s.InT+2*s.PadT-s.KT)/s.StrideT + 1
	h = (s.InH+2*s.PadH-s.KH)/s.StrideH + 1
	w = (s.InW+2*s.PadW-s.KW)/s.StrideW + 1
	if t <= 0 || h <= 0 || w <= 0 {
		return 0, 0, 0, fmt.Errorf("conv3d: nonpositive output for %+v", s)
	}
	return t, h, w, nil
}

// CausalConv3DInto: causal 3-D convolution. All 2*PadT temporal padding sits
// on the left; cache holds cacheT prior input frames [cIn][cacheT][H][W] that
// replace the rightmost cacheT pad frames (the streaming invariant). cache
// nil with cacheT=0 is the cold start (pure zero padding). bias may be nil.
func CausalConv3DInto(out, x, cache, w, bias []float32, cacheT int, shape Conv3DShape) error {
	outT, outH, outW, err := shape.OutputDims()
	if err != nil {
		return err
	}
	if cacheT < 0 || cacheT > 2*shape.PadT {
		return fmt.Errorf("conv3d cache: cacheT=%d outside [0,%d]", cacheT, 2*shape.PadT)
	}
	if len(cache) != shape.CIn*cacheT*shape.InH*shape.InW {
		return fmt.Errorf("conv3d cache: cache len=%d want %d", len(cache), shape.CIn*cacheT*shape.InH*shape.InW)
	}
	if len(x) != shape.CIn*shape.InT*shape.InH*shape.InW {
		return fmt.Errorf("conv3d: x len=%d want %d", len(x), shape.CIn*shape.InT*shape.InH*shape.InW)
	}
	if len(w) != shape.COut*shape.CIn*shape.KT*shape.KH*shape.KW {
		return fmt.Errorf("conv3d: weight len=%d want %d", len(w), shape.COut*shape.CIn*shape.KT*shape.KH*shape.KW)
	}
	if bias != nil && len(bias) != shape.COut {
		return fmt.Errorf("conv3d: bias len=%d want %d", len(bias), shape.COut)
	}
	if len(out) != shape.COut*outT*outH*outW {
		return fmt.Errorf("conv3d: out len=%d want %d", len(out), shape.COut*outT*outH*outW)
	}
	// outT*outH*outW*cIn*kT*kH*kW MACs per output channel.
	parallelRangeCost(shape.COut, outT*outH*outW*shape.CIn*shape.KT*shape.KH*shape.KW, macF64, func(coLo, coHi int) {
		causalConv3DChannels(out, x, cache, w, bias, cacheT, shape, outT, outH, outW, coLo, coHi)
	})
	return nil
}

// causalConv3DChannels: output channels [coLo,coHi) — the serial kernel the
// dispatch calibration times. Stable (ci,kt,kh,kw) accumulation order.
func causalConv3DChannels(out, x, cache, w, bias []float32, cacheT int, s Conv3DShape, outT, outH, outW, coLo, coHi int) {
	for co := coLo; co < coHi; co++ {
		for ot := 0; ot < outT; ot++ {
			for oh := 0; oh < outH; oh++ {
				for ow := 0; ow < outW; ow++ {
					acc := float64(0)
					if bias != nil {
						acc = float64(bias[co])
					}
					for ci := 0; ci < s.CIn; ci++ {
						for kt := 0; kt < s.KT; kt++ {
							paddedT := ot*s.StrideT + kt - (2*s.PadT - cacheT)
							if paddedT < 0 || paddedT >= cacheT+s.InT {
								continue
							}
							src := x
							it := paddedT - cacheT
							frames := s.InT
							if paddedT < cacheT {
								src, it, frames = cache, paddedT, cacheT
							}
							for kh := 0; kh < s.KH; kh++ {
								ih := oh*s.StrideH + kh - s.PadH
								if ih < 0 || ih >= s.InH {
									continue
								}
								for kw := 0; kw < s.KW; kw++ {
									iw := ow*s.StrideW + kw - s.PadW
									if iw < 0 || iw >= s.InW {
										continue
									}
									xi := ((ci*frames+it)*s.InH+ih)*s.InW + iw
									wi := (((co*s.CIn+ci)*s.KT+kt)*s.KH+kh)*s.KW + kw
									acc += float64(src[xi]) * float64(w[wi])
								}
							}
						}
					}
					out[((co*outT+ot)*outH+oh)*outW+ow] = float32(acc)
				}
			}
		}
	}
}

// SquareKernelSide: derives the side of a square 2-D kernel from a flat
// weight length and its channel-pair count.
func SquareKernelSide(weightElements, channelPairs int) (int, error) {
	if weightElements <= 0 || channelPairs <= 0 || weightElements%channelPairs != 0 {
		return 0, fmt.Errorf("square kernel: weight=%d channel pairs=%d", weightElements, channelPairs)
	}
	area := weightElements / channelPairs
	side := int(math.Sqrt(float64(area)))
	if side*side != area {
		return 0, fmt.Errorf("square kernel: area=%d is not square", area)
	}
	return side, nil
}

// ResizeConv2DInto: nearest-neighbor 2x spatial upsample fused with a
// same-padded square Conv2d, applied per frame of a channel-major
// [cIn][t][h][w] volume (the Wan resample stack Upsample(2x)+Conv2d). The
// kernel side derives from the weight length. bias may be nil.
func ResizeConv2DInto(out, x, w, bias []float32, cIn, cOut, t, h, wd int) error {
	if cIn <= 0 || cOut <= 0 || t <= 0 || h <= 0 || wd <= 0 {
		return fmt.Errorf("resize conv2d: bad shape cIn=%d cOut=%d t=%d h=%d w=%d", cIn, cOut, t, h, wd)
	}
	kernel, err := SquareKernelSide(len(w), cOut*cIn)
	if err != nil {
		return fmt.Errorf("resize conv2d: %w", err)
	}
	const scale = 2
	oh, ow := scale*h, scale*wd
	if len(x) != cIn*t*h*wd || len(out) != cOut*t*oh*ow {
		return fmt.Errorf("resize conv2d: bad lengths out=%d x=%d", len(out), len(x))
	}
	if bias != nil && len(bias) != cOut {
		return fmt.Errorf("resize conv2d: bias len=%d want %d", len(bias), cOut)
	}
	plane := oh * ow
	// cIn*kernel^2*plane MACs per (channel, frame) owner.
	parallelRangeCost(cOut*t, cIn*kernel*kernel*plane, macF64, func(lo, hi int) {
		acc := make([]float64, plane)
		for owner := lo; owner < hi; owner++ {
			co, ti := owner/t, owner%t
			fill := float64(0)
			if bias != nil {
				fill = float64(bias[co])
			}
			for i := range acc {
				acc[i] = fill
			}
			for ci := 0; ci < cIn; ci++ {
				weightBase := (co*cIn + ci) * kernel * kernel
				for ky := 0; ky < kernel; ky++ {
					dy := ky - kernel/2
					for kx := 0; kx < kernel; kx++ {
						dx := kx - kernel/2
						wv := float64(w[weightBase+ky*kernel+kx])
						for y := 0; y < oh; y++ {
							sy := y + dy
							if sy < 0 || sy >= oh {
								continue
							}
							inputRow := ((ci*t+ti)*h + sy/scale) * wd
							for xOut := 0; xOut < ow; xOut++ {
								sx := xOut + dx
								if sx >= 0 && sx < ow {
									acc[y*ow+xOut] += float64(x[inputRow+sx/scale]) * wv
								}
							}
						}
					}
				}
			}
			base := (co*t + ti) * plane
			for i, value := range acc {
				out[base+i] = float32(value)
			}
		}
	})
	return nil
}
