package patchtower

import (
	"bytes"
	"fmt"
	"image"
	_ "image/jpeg"
	_ "image/png"
	"path/filepath"

	"overgo/internal/jsonfile"
	"overgo/internal/media"
)

// Pipeline: aligned resize, antialias, normalize, merged patch order.

type PreprocessConfig struct {
	PatchSize         int       `json:"patch_size"`
	TemporalPatchSize int       `json:"temporal_patch_size"`
	MergeSize         int       `json:"merge_size"`
	ImageMean         []float64 `json:"image_mean"`
	ImageStd          []float64 `json:"image_std"`
	Size              struct {
		ShortestEdge int `json:"shortest_edge"`
		LongestEdge  int `json:"longest_edge"`
	} `json:"size"`
}

func LoadPreprocessConfig(modelDir string) (*PreprocessConfig, error) {
	var c PreprocessConfig
	if err := jsonfile.Decode(filepath.Join(modelDir, "preprocessor_config.json"), &c); err != nil {
		return nil, fmt.Errorf("patch tower preprocessor config: %w", err)
	}
	if c.PatchSize <= 0 || c.TemporalPatchSize <= 0 || c.MergeSize <= 0 ||
		len(c.ImageMean) != media.RGBChannels || len(c.ImageStd) != media.RGBChannels ||
		c.Size.ShortestEdge <= 0 || c.Size.LongestEdge <= 0 {
		return nil, fmt.Errorf("patch tower preprocessor config incomplete: %+v", c)
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
	rgb = make([]uint8, h*w*media.RGBChannels)
	for y := range h {
		for x := range w {
			r, g, bl, _ := im.At(b.Min.X+x, b.Min.Y+y).RGBA()
			i := (y*w + x) * media.RGBChannels
			rgb[i], rgb[i+1], rgb[i+2] = uint8(r>>8), uint8(g>>8), uint8(bl>>8)
		}
	}
	return rgb, h, w, nil
}

// PreprocessImage: single frame -> merged-patch pixel values + grid.
func PreprocessImage(c *PreprocessConfig, rgb []uint8, h, w int) (pixelValues []float32, gridT, gridH, gridW int, err error) {
	if len(rgb) != h*w*media.RGBChannels {
		return nil, 0, 0, 0, fmt.Errorf("patch tower preprocess: frame len %d != %d", len(rgb), h*w*media.RGBChannels)
	}
	rh, rw, err := media.ResizeAligned(h, w, c.PatchSize*c.MergeSize, c.Size.ShortestEdge, c.Size.LongestEdge)
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

// resizeBicubicAntialias resizes HWC-uint8 src via the torch uint8 antialias
// bicubic path: separable fp64 accumulation with uint8 round+clamp between axes.
func resizeBicubicAntialias(src []uint8, h, w, th, tw int) ([]uint8, error) {
	if len(src) != h*w*media.RGBChannels {
		return nil, fmt.Errorf("resize bicubic antialias: src len %d != %d", len(src), h*w*media.RGBChannels)
	}
	xmin, xk, xw := aaWeights(w, tw)
	mid := make([]uint8, h*tw*media.RGBChannels)
	for y := range h {
		for ox := range tw {
			x0, k, wt := xmin[ox], xk[ox], xw[ox]
			var r, g, b float64
			for j := range k {
				si := (y*w + (x0 + j)) * media.RGBChannels
				r += wt[j] * float64(src[si])
				g += wt[j] * float64(src[si+1])
				b += wt[j] * float64(src[si+2])
			}
			di := (y*tw + ox) * media.RGBChannels
			mid[di], mid[di+1], mid[di+2] = media.RoundedUint8(r), media.RoundedUint8(g), media.RoundedUint8(b)
		}
	}
	ymin, yk, yw := aaWeights(h, th)
	out := make([]uint8, th*tw*media.RGBChannels)
	for oy := range th {
		y0, k, wt := ymin[oy], yk[oy], yw[oy]
		for x := range tw {
			var r, g, b float64
			for j := range k {
				si := ((y0+j)*tw + x) * media.RGBChannels
				r += wt[j] * float64(mid[si])
				g += wt[j] * float64(mid[si+1])
				b += wt[j] * float64(mid[si+2])
			}
			di := (oy*tw + x) * media.RGBChannels
			out[di], out[di+1], out[di+2] = media.RoundedUint8(r), media.RoundedUint8(g), media.RoundedUint8(b)
		}
	}
	return out, nil
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
	for i := range outSize {
		center := scale * (float64(i) + media.RasterSampleCenter)
		lo := int(center - support + media.RasterSampleCenter)
		if lo < 0 {
			lo = 0
		}
		hi := int(center + support + media.RasterSampleCenter)
		if hi > inSize {
			hi = inSize
		}
		k := hi - lo
		w := make([]float64, k)
		var sum float64
		for x := range k {
			w[x] = media.CubicConvolutionWeight((float64(x+lo) - center + media.RasterSampleCenter) * invscale)
			sum += w[x]
		}
		if sum != 0 {
			for x := range w {
				w[x] /= sum
			}
		}
		xmin[i], ksize[i], weights[i] = lo, k, w
	}
	return xmin, ksize, weights
}

func resizeNormalizeRGBBicubic(rgb []uint8, h, w, outH, outW int, mean, std []float64) ([]float32, error) {
	if len(mean) != media.RGBChannels || len(std) != media.RGBChannels {
		return nil, fmt.Errorf("RGB normalization requires %d channel parameters", media.RGBChannels)
	}
	resized, err := resizeBicubicAntialias(rgb, h, w, outH, outW)
	if err != nil {
		return nil, err
	}
	plane := outH * outW
	normalized := make([]float32, media.RGBChannels*plane)
	for y := range outH {
		for x := range outW {
			for channel := 0; channel < media.RGBChannels; channel++ {
				if std[channel] <= 0 {
					return nil, fmt.Errorf("RGB normalization channel %d has non-positive scale", channel)
				}
				value := float64(resized[(y*outW+x)*media.RGBChannels+channel]) / 255.0
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
		if len(fr) != media.RGBChannels*plane {
			return nil, 0, 0, 0, fmt.Errorf("patch tower patchify: frame %d len %d != %d", i, len(fr), media.RGBChannels*plane)
		}
	}
	gridT, gridH, gridW := len(frames)/temporal, rh/patch, rw/patch
	hb, wb := gridH/merge, gridW/merge
	featDim := media.RGBChannels * temporal * patch * patch
	pixelValues := make([]float32, gridT*gridH*gridW*featDim)
	p := 0
	for gt := range gridT {
		for hbi := range hb {
			for wbi := range wb {
				for mh := range merge {
					for mw := range merge {
						f := 0
						for ch := 0; ch < media.RGBChannels; ch++ {
							yc := (hbi*merge + mh) * patch
							xc := (wbi*merge + mw) * patch
							base := ch*plane + yc*rw + xc
							for t := range temporal {
								fr := frames[gt*temporal+t]
								for ph := range patch {
									row := base + ph*rw
									for pw := range patch {
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
