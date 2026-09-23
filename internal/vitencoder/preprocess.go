package vitencoder

import (
	"errors"
	"fmt"
	"math"
	"slices"

	"overgo/internal/checked"
	"overgo/internal/jsonfile"
)

// The resize below is PIL's 8-bit bicubic resample, which a checkpoint's own
// image processor applies: coefficients from the a = -0.5 cubic, widened by
// the downscale factor so it antialiases, normalized per output pixel and
// converted to 22-bit fixed point; a horizontal pass then a vertical one,
// each rounding and clamping to 8 bits. Matching it exactly is what lets an
// embedding be compared with the reference bit for bit at the pixel stage.

const (
	// pilPrecisionBits is PIL's fixed-point coefficient precision for 8-bit
	// images: 32 bits less 8 for the value and 2 for headroom.
	pilPrecisionBits = 32 - 8 - 2
	// bicubicSupport is the cubic filter's radius in source pixels at scale 1.
	bicubicSupport = 2.0
	// bicubicA is the cubic coefficient PIL uses.
	bicubicA = -0.5
	channels = 3
)

// bicubic is the cubic convolution kernel; it is zero beyond its support.
func bicubic(x float64) (weight float64) {
	x = math.Abs(x)
	switch {
	case x < 1:
		weight = ((bicubicA+2)*x-(bicubicA+3))*x*x + 1
	case x < bicubicSupport:
		weight = (((x-5)*x+8)*x - 4) * bicubicA
	}
	return weight
}

// resampleWeights are one axis's fixed-point coefficients: for each output
// position the first source index and its weights.
type resampleWeights struct {
	start   []int
	weights [][]int64
}

func pilWeights(inSize, outSize int) resampleWeights {
	scale := float64(inSize) / float64(outSize)
	filterScale := max(scale, 1)
	support := bicubicSupport * filterScale
	result := resampleWeights{start: make([]int, outSize), weights: make([][]int64, outSize)}
	for out := range outSize {
		center := (float64(out) + 0.5) * scale
		first := max(int(center-support+0.5), 0)
		last := min(int(center+support+0.5), inSize)
		coefficients := make([]float64, last-first)
		var total float64
		for index := range coefficients {
			coefficients[index] = bicubic((float64(index+first) - center + 0.5) / filterScale)
			total += coefficients[index]
		}
		fixed := make([]int64, len(coefficients))
		for index, value := range coefficients {
			if total != 0 {
				value /= total
			}
			if value < 0 {
				fixed[index] = int64(-0.5 + value*(1<<pilPrecisionBits))
			} else {
				fixed[index] = int64(0.5 + value*(1<<pilPrecisionBits))
			}
		}
		result.start[out], result.weights[out] = first, fixed
	}
	return result
}

// clip8 drops the fixed-point fraction and clamps to a byte.
func clip8(sum int64) (clipped uint8) {
	if value := sum >> pilPrecisionBits; checked.NonNegativeInt64(value) {
		clipped = uint8(min(value, math.MaxUint8))
	}
	return clipped
}

// resizeBicubic resizes interleaved RGB bytes as PIL's bicubic resample does.
func resizeBicubic(source []uint8, height, width, outHeight, outWidth int) ([]uint8, error) {
	if !checked.PositiveInts(outHeight, outWidth) || !checked.Equal(len(source), height*width*channels) {
		return nil, errors.New("vitencoder: resize input does not match its declared size")
	}
	rounding := int64(1) << (pilPrecisionBits - 1)
	horizontal := pilWeights(width, outWidth)
	across := make([]uint8, height*outWidth*channels)
	for y := range height {
		for x := range outWidth {
			for c := range channels {
				sum := rounding
				for offset, weight := range horizontal.weights[x] {
					sum += int64(source[(y*width+horizontal.start[x]+offset)*channels+c]) * weight
				}
				across[(y*outWidth+x)*channels+c] = clip8(sum)
			}
		}
	}
	vertical := pilWeights(height, outHeight)
	resized := make([]uint8, outHeight*outWidth*channels)
	for y := range outHeight {
		for x := range outWidth {
			for c := range channels {
				sum := rounding
				for offset, weight := range vertical.weights[y] {
					sum += int64(across[((vertical.start[y]+offset)*outWidth+x)*channels+c]) * weight
				}
				resized[(y*outWidth+x)*channels+c] = clip8(sum)
			}
		}
	}
	return resized, nil
}

// pilBicubic is PIL's Resampling.BICUBIC, the one resample ported here.
const pilBicubic = 3

// processorConfig is the checkpoint's image processor configuration.
type processorConfig struct {
	Resize    bool `json:"do_resize"`
	Crop      bool `json:"do_center_crop"`
	Rescale   bool `json:"do_rescale"`
	Normalize bool `json:"do_normalize"`
	Resample  int  `json:"resample"`
	Size      struct {
		ShortestEdge int `json:"shortest_edge"`
	} `json:"size"`
	CropSize struct {
		Height int `json:"height"`
		Width  int `json:"width"`
	} `json:"crop_size"`
	RescaleFactor float64    `json:"rescale_factor"`
	Mean          [3]float32 `json:"image_mean"`
	Std           [3]float32 `json:"image_std"`
}

// preprocess is a checkpoint's image processor: resize so the shorter edge is
// shortestEdge, crop the centre square of crop, rescale and normalize each
// channel.
type preprocess struct {
	shortestEdge, crop int
	rescale            float32
	mean, std          [3]float32
}

// loadPreprocess reads the image processor configuration and refuses one that
// asks for a step or a resample this port does not perform.
func loadPreprocess(path string) (preprocess, error) {
	var config processorConfig
	if err := jsonfile.Decode(path, &config); err != nil {
		return preprocess{}, fmt.Errorf("vitencoder: image processor configuration: %w", err)
	}
	if !config.Resize || !config.Crop || !config.Rescale || !config.Normalize || config.Resample != pilBicubic ||
		!checked.PositiveInts(config.Size.ShortestEdge, config.CropSize.Height) || !checked.Equal(config.CropSize.Height, config.CropSize.Width) ||
		!checked.PositiveFinite64(config.RescaleFactor) ||
		slices.ContainsFunc(config.Std[:], func(std float32) bool { return !checked.PositiveFinite32(std) }) {
		return preprocess{}, errors.New("vitencoder: the image processor is not a bicubic resize, square centre crop, rescale and normalization")
	}
	return preprocess{
		shortestEdge: config.Size.ShortestEdge, crop: config.CropSize.Height,
		rescale: float32(config.RescaleFactor), mean: config.Mean, std: config.Std,
	}, nil
}

// resizedSize is the processor's output size before the crop, truncating the
// longer edge as the processor does.
func (p preprocess) resizedSize(height, width int) (int, int) {
	if checked.AtMostInt(height, width) {
		return p.shortestEdge, p.shortestEdge * width / height
	}
	return p.shortestEdge * height / width, p.shortestEdge
}

// pixels turns interleaved RGB bytes into the channel-first normalized pixels
// the encoder reads.
func (p preprocess) pixels(source []uint8, height, width int) ([]float32, error) {
	outHeight, outWidth := p.resizedSize(height, width)
	if !checked.AtLeastInt(outHeight, p.crop) || !checked.AtLeastInt(outWidth, p.crop) {
		return nil, errors.New("vitencoder: image is smaller than the crop")
	}
	resized, err := resizeBicubic(source, height, width, outHeight, outWidth)
	if err != nil {
		return nil, err
	}
	top, left := (outHeight-p.crop)/2, (outWidth-p.crop)/2
	pixels := make([]float32, channels*p.crop*p.crop)
	for c := range channels {
		for y := range p.crop {
			for x := range p.crop {
				value := float32(resized[((top+y)*outWidth+left+x)*channels+c]) * p.rescale
				pixels[(c*p.crop+y)*p.crop+x] = (value - p.mean[c]) / p.std[c]
			}
		}
	}
	return pixels, nil
}
