package patchtower

import (
	"bytes"
	"encoding/json"
	"fmt"
	"image"
	_ "image/jpeg"
	_ "image/png"
	"math"
	"os"
	"path/filepath"
)

// Preprocess order: aligned smart resize, torch-style uint8 antialias bicubic
// sample, normalize, patchify into merged-block token order. Ported from
// adaptive_new extmodel media.go / hybrid_recurrent_decoder.go.

// maxAspectRatioDefault: transformers qwen2_vl smart_resize MAX_RATIO; absent
// from preprocessor_config.json so the reference processor default governs.
const maxAspectRatioDefault = 200

const imageSampleCenter = 0.5

type PreprocessConfig struct {
	PatchSize         int       `json:"patch_size"`
	TemporalPatchSize int       `json:"temporal_patch_size"`
	MergeSize         int       `json:"merge_size"`
	ImageMean         []float64 `json:"image_mean"`
	ImageStd          []float64 `json:"image_std"`
	MaxAspectRatio    int       `json:"max_aspect_ratio"`
	Size              struct {
		ShortestEdge int `json:"shortest_edge"`
		LongestEdge  int `json:"longest_edge"`
	} `json:"size"`
}

func LoadPreprocessConfig(modelDir string) (*PreprocessConfig, error) {
	raw, err := os.ReadFile(filepath.Join(modelDir, "preprocessor_config.json"))
	if err != nil {
		return nil, err
	}
	var c PreprocessConfig
	if err := json.Unmarshal(raw, &c); err != nil {
		return nil, fmt.Errorf("patch tower preprocessor config: %w", err)
	}
	if c.PatchSize <= 0 || c.TemporalPatchSize <= 0 || c.MergeSize <= 0 ||
		len(c.ImageMean) != RGBChannels || len(c.ImageStd) != RGBChannels ||
		c.Size.ShortestEdge <= 0 || c.Size.LongestEdge <= 0 {
		return nil, fmt.Errorf("patch tower preprocessor config incomplete: %+v", c)
	}
	if c.MaxAspectRatio == 0 {
		c.MaxAspectRatio = maxAspectRatioDefault
	}
	if c.MaxAspectRatio < 0 {
		return nil, fmt.Errorf("patch tower preprocessor config invalid max_aspect_ratio: %+v", c)
	}
	return &c, nil
}

// DecodeImageBytesRGB: decodes encoded image bytes (PNG/JPEG) to row-major
// HWC uint8 RGB, dropping alpha.
func DecodeImageBytesRGB(data []byte) (rgb []uint8, h, w int, err error) {
	im, _, err := image.Decode(bytes.NewReader(data))
	if err != nil {
		return nil, 0, 0, err
	}
	b := im.Bounds()
	h, w = b.Dy(), b.Dx()
	rgb = make([]uint8, h*w*RGBChannels)
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			r, g, bl, _ := im.At(b.Min.X+x, b.Min.Y+y).RGBA()
			i := (y*w + x) * RGBChannels
			rgb[i], rgb[i+1], rgb[i+2] = uint8(r>>8), uint8(g>>8), uint8(bl>>8)
		}
	}
	return rgb, h, w, nil
}

// PreprocessImage: single frame -> merged-patch pixel values + grid.
func PreprocessImage(c *PreprocessConfig, rgb []uint8, h, w int) (pixelValues []float32, gridT, gridH, gridW int, err error) {
	if len(rgb) != h*w*RGBChannels {
		return nil, 0, 0, 0, fmt.Errorf("patch tower preprocess: frame len %d != %d", len(rgb), h*w*RGBChannels)
	}
	rh, rw, err := smartResizeAligned(h, w, c.PatchSize*c.MergeSize, c.Size.ShortestEdge, c.Size.LongestEdge, c.MaxAspectRatio)
	if err != nil {
		return nil, 0, 0, 0, fmt.Errorf("patch tower smart resize: %w", err)
	}
	normalized, err := resizeNormalizeRGBBicubic(rgb, h, w, rh, rw, c.ImageMean, c.ImageStd)
	if err != nil {
		return nil, 0, 0, 0, err
	}
	frames := [][]float32{normalized}
	for len(frames) < c.TemporalPatchSize {
		frames = append(frames, frames[len(frames)-1])
	}
	return patchifyMergedFrames(frames, rh, rw, c.PatchSize, c.TemporalPatchSize, c.MergeSize)
}

// smartResizeAligned preserves aspect ratio while satisfying a model-derived
// alignment factor and explicit pixel-area/aspect-ratio contract. RoundToEven
// matches Python's round, including exact half cases.
func smartResizeAligned(h, w, factor, minPixels, maxPixels, maxAspectRatio int) (int, int, error) {
	if h <= 0 || w <= 0 || factor <= 0 || minPixels <= 0 || maxPixels < minPixels || maxAspectRatio <= 0 {
		return 0, 0, fmt.Errorf("invalid aligned resize contract: image=%dx%d factor=%d pixels=[%d,%d] max_aspect=%d", h, w, factor, minPixels, maxPixels, maxAspectRatio)
	}
	hi, lo := h, w
	if hi < lo {
		hi, lo = lo, hi
	}
	if float64(hi)/float64(lo) > float64(maxAspectRatio) {
		return 0, 0, fmt.Errorf("aspect ratio exceeds %d (%dx%d)", maxAspectRatio, h, w)
	}
	f := float64(factor)
	hBar := max(factor, int(math.RoundToEven(float64(h)/f))*factor)
	wBar := max(factor, int(math.RoundToEven(float64(w)/f))*factor)
	switch {
	case hBar*wBar > maxPixels:
		beta := math.Sqrt(float64(h) * float64(w) / float64(maxPixels))
		hBar = max(factor, int(math.Floor(float64(h)/beta/f))*factor)
		wBar = max(factor, int(math.Floor(float64(w)/beta/f))*factor)
	case hBar*wBar < minPixels:
		beta := math.Sqrt(float64(minPixels) / (float64(h) * float64(w)))
		hBar = int(math.Ceil(float64(h)*beta/f)) * factor
		wBar = int(math.Ceil(float64(w)*beta/f)) * factor
	}
	return hBar, wBar, nil
}

// resizeBicubicAntialias resizes HWC-uint8 src via the torch uint8 antialias
// bicubic path: separable fp64 accumulation with uint8 round+clamp between axes.
func resizeBicubicAntialias(src []uint8, h, w, th, tw int) ([]uint8, error) {
	if len(src) != h*w*RGBChannels {
		return nil, fmt.Errorf("resize bicubic antialias: src len %d != %d", len(src), h*w*RGBChannels)
	}
	xmin, xk, xw := aaWeights(w, tw)
	mid := make([]uint8, h*tw*RGBChannels)
	for y := 0; y < h; y++ {
		for ox := 0; ox < tw; ox++ {
			x0, k, wt := xmin[ox], xk[ox], xw[ox]
			var r, g, b float64
			for j := 0; j < k; j++ {
				si := (y*w + (x0 + j)) * RGBChannels
				r += wt[j] * float64(src[si])
				g += wt[j] * float64(src[si+1])
				b += wt[j] * float64(src[si+2])
			}
			di := (y*tw + ox) * RGBChannels
			mid[di], mid[di+1], mid[di+2] = clampRoundU8(r), clampRoundU8(g), clampRoundU8(b)
		}
	}
	ymin, yk, yw := aaWeights(h, th)
	out := make([]uint8, th*tw*RGBChannels)
	for oy := 0; oy < th; oy++ {
		y0, k, wt := ymin[oy], yk[oy], yw[oy]
		for x := 0; x < tw; x++ {
			var r, g, b float64
			for j := 0; j < k; j++ {
				si := ((y0+j)*tw + x) * RGBChannels
				r += wt[j] * float64(mid[si])
				g += wt[j] * float64(mid[si+1])
				b += wt[j] * float64(mid[si+2])
			}
			di := (oy*tw + x) * RGBChannels
			out[di], out[di+1], out[di+2] = clampRoundU8(r), clampRoundU8(g), clampRoundU8(b)
		}
	}
	return out, nil
}

func cubicAA(t float64) float64 {
	const a = -0.75 // torch bicubic convolution parameter
	if t < 0 {
		t = -t
	}
	if t < 1.0 {
		return ((a+2.0)*t-(a+3.0))*t*t + 1.0
	}
	if t < 2.0 {
		return (((t-5.0)*t+8.0)*t - 4.0) * a
	}
	return 0.0
}

func aaWeights(inSize, outSize int) (xmin []int, ksize []int, weights [][]float64) {
	scale := float64(inSize) / float64(outSize)
	support := 2.0
	invscale := 1.0
	if scale >= 1.0 {
		support = 2.0 * scale
		invscale = 1.0 / scale
	}
	xmin = make([]int, outSize)
	ksize = make([]int, outSize)
	weights = make([][]float64, outSize)
	for i := 0; i < outSize; i++ {
		center := scale * (float64(i) + imageSampleCenter)
		lo := int(center - support + imageSampleCenter)
		if lo < 0 {
			lo = 0
		}
		hi := int(center + support + imageSampleCenter)
		if hi > inSize {
			hi = inSize
		}
		k := hi - lo
		w := make([]float64, k)
		var sum float64
		for x := 0; x < k; x++ {
			w[x] = cubicAA((float64(x+lo) - center + imageSampleCenter) * invscale)
			sum += w[x]
		}
		if sum != 0 {
			for x := range w {
				w[x] /= sum
			}
		}
		xmin[i], ksize[i], weights[i] = lo, k, w
	}
	return
}

func clampRoundU8(v float64) uint8 {
	v = math.Round(v)
	if v <= 0 {
		return 0
	}
	if v >= 255 {
		return 255
	}
	return uint8(v)
}

func resizeNormalizeRGBBicubic(rgb []uint8, h, w, outH, outW int, mean, std []float64) ([]float32, error) {
	if len(mean) != RGBChannels || len(std) != RGBChannels {
		return nil, fmt.Errorf("RGB normalization requires %d channel parameters", RGBChannels)
	}
	resized, err := resizeBicubicAntialias(rgb, h, w, outH, outW)
	if err != nil {
		return nil, err
	}
	plane := outH * outW
	normalized := make([]float32, RGBChannels*plane)
	for y := 0; y < outH; y++ {
		for x := 0; x < outW; x++ {
			for channel := 0; channel < RGBChannels; channel++ {
				if std[channel] <= 0 {
					return nil, fmt.Errorf("RGB normalization channel %d has non-positive scale", channel)
				}
				value := float64(resized[(y*outW+x)*RGBChannels+channel]) / 255.0
				normalized[channel*plane+y*outW+x] = float32((value - mean[channel]) / std[channel])
			}
		}
	}
	return normalized, nil
}

// patchifyMergedFrames: emits temporal/spatial merged patches from CHW frames.
func patchifyMergedFrames(frames [][]float32, rh, rw, patch, temporal, merge int) ([]float32, int, int, int, error) {
	if len(frames) == 0 || temporal <= 0 || len(frames)%temporal != 0 {
		return nil, 0, 0, 0, fmt.Errorf("patch tower patchify: %d frames not a positive multiple of temporal %d", len(frames), temporal)
	}
	plane := rh * rw
	for i, fr := range frames {
		if len(fr) != RGBChannels*plane {
			return nil, 0, 0, 0, fmt.Errorf("patch tower patchify: frame %d len %d != %d", i, len(fr), RGBChannels*plane)
		}
	}
	gridT, gridH, gridW := len(frames)/temporal, rh/patch, rw/patch
	hb, wb := gridH/merge, gridW/merge
	featDim := RGBChannels * temporal * patch * patch
	pixelValues := make([]float32, gridT*gridH*gridW*featDim)
	p := 0
	for gt := 0; gt < gridT; gt++ {
		for hbi := 0; hbi < hb; hbi++ {
			for wbi := 0; wbi < wb; wbi++ {
				for mh := 0; mh < merge; mh++ {
					for mw := 0; mw < merge; mw++ {
						f := 0
						for ch := 0; ch < RGBChannels; ch++ {
							yc := (hbi*merge + mh) * patch
							xc := (wbi*merge + mw) * patch
							base := ch*plane + yc*rw + xc
							for t := 0; t < temporal; t++ {
								fr := frames[gt*temporal+t]
								for ph := 0; ph < patch; ph++ {
									row := base + ph*rw
									for pw := 0; pw < patch; pw++ {
										pixelValues[p*featDim+f] = fr[row+pw]
										f++
									}
								}
							}
						}
						p++
					}
				}
			}
		}
	}
	return pixelValues, gridT, gridH, gridW, nil
}
