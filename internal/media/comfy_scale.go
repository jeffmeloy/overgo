package media

import (
	"errors"
	"fmt"
	"image"
	"image/color"
	"math"

	"overgo/internal/checked"
)

// ComfyUI's ImageScale at lanczos with a centre crop (comfy.utils.common_upscale), which the MOSS Ultra workflow's image
// to image sections scale their source by, reproduced to the byte: LoadImage reads a picture as 8-bit RGB over 255 in
// float32; the crop keeps the middle of the side the canvas's aspect cuts, its margin rounded half to even as Python
// rounds; lanczos turns the float32 picture back to 8 bits as 255 * x truncated, and Pillow resamples those 8-bit
// values with its LANCZOS filter in fixed point (Resample.c), the rows first and the columns after, a pass skipped where
// its side keeps its size; the scaled picture is its 8-bit values over 255.
const (
	// pillowPrecisionBits is the fixed-point precision of Pillow's 8-bit resampling coefficients (PRECISION_BITS).
	pillowPrecisionBits = 32 - 8 - 2
	// lanczosSupport is the half-width of Pillow's LANCZOS filter, three lobes of sinc.
	lanczosSupport = 3.0
	// UnitLevels is the largest 8-bit level, the divisor ComfyUI reads a picture over.
	UnitLevels = 255
)

// comfyLevels is the 8-bit level ComfyUI's lanczos makes of each level LoadImage read: float32(v) / 255, times 255 in
// float32, truncated.
var comfyLevels = func() (levels [UnitLevels + 1]uint8) {
	for value := range levels {
		unit := float32(value) / UnitLevels
		levels[value] = uint8(unit * UnitLevels)
	}
	return levels
}()

// ScaleLikeComfyUI is picture scaled to width by height as ComfyUI's ImageScale scales it at lanczos with a centre
// crop: its 8-bit RGB levels, a row's pixels after another's. A picture is read as LoadImage converts it to RGB, its
// alpha dropped; its EXIF orientation is not applied, nor JPEG decoded as Pillow decodes it, so a PNG without one is
// what this scales to ComfyUI's bytes.
func ScaleLikeComfyUI(picture image.Image, width, height int) ([]uint8, error) {
	bounds := picture.Bounds()
	inputWidth, inputHeight := bounds.Dx(), bounds.Dy()
	if !checked.PositiveInts(width, height, inputWidth, inputHeight) {
		return nil, fmt.Errorf("media: scale a %dx%d picture to %dx%d: sides must be positive", inputWidth, inputHeight, width, height)
	}
	left, top := centreCrop(inputWidth, inputHeight, width, height)
	croppedWidth, croppedHeight := inputWidth-2*left, inputHeight-2*top
	if croppedWidth <= 0 || croppedHeight <= 0 {
		return nil, errors.New("media: the centre crop leaves no picture")
	}
	levels := make([]uint8, croppedWidth*croppedHeight*RGBChannels)
	for y := range croppedHeight {
		for x := range croppedWidth {
			pixel := color.NRGBAModel.Convert(picture.At(bounds.Min.X+left+x, bounds.Min.Y+top+y)).(color.NRGBA)
			at := (y*croppedWidth + x) * RGBChannels
			levels[at], levels[at+1], levels[at+2] = comfyLevels[pixel.R], comfyLevels[pixel.G], comfyLevels[pixel.B]
		}
	}
	if croppedWidth != width {
		levels = pillowLanczosRows(levels, croppedWidth, croppedHeight, width)
	}
	if croppedHeight != height {
		levels = pillowLanczosColumns(levels, width, croppedHeight, height)
	}
	return levels, nil
}

// centreCrop is the margin ComfyUI's centre crop takes off each side of a width by height picture for a canvas of
// width by height: off the left and right where the picture is wider than the canvas's aspect, off the top and bottom
// where it is taller, rounded half to even.
func centreCrop(inputWidth, inputHeight, width, height int) (left, top int) {
	old := float64(inputWidth) / float64(inputHeight)
	aspect := float64(width) / float64(height)
	switch {
	case old > aspect:
		left = int(math.RoundToEven((float64(inputWidth) - float64(inputWidth)*(aspect/old)) / 2))
	case old < aspect:
		top = int(math.RoundToEven((float64(inputHeight) - float64(inputHeight)*(old/aspect)) / 2))
	}
	return left, top
}

// pillowWindow is the input positions one output position of a Pillow pass reads, from first, and their
// fixed-point coefficients.
type pillowWindow struct {
	first        int
	coefficients []int32
}

// pillowLanczosWindows are Pillow's LANCZOS windows for a side of input positions resampled to output (precompute_coeffs
// with normalize_coeffs_8bpc): each output position's centre, the inputs within the filter's support, widened by the
// scale where it shrinks, their weights normalized and rounded half away from zero to fixed point.
func pillowLanczosWindows(input, output int) []pillowWindow {
	scale := float64(input) / float64(output)
	filterScale := max(scale, 1)
	support := lanczosSupport * filterScale
	windows := make([]pillowWindow, output)
	for index := range output {
		centre := (float64(index) + 0.5) * scale
		first := max(int(centre-support+0.5), 0)
		last := min(int(centre+support+0.5), input)
		weights := make([]float64, last-first)
		total := 0.0
		for offset := range weights {
			weights[offset] = pillowLanczos((float64(offset+first) - centre + 0.5) / filterScale)
			total += weights[offset]
		}
		coefficients := make([]int32, len(weights))
		for offset, weight := range weights {
			if total != 0 {
				weight /= total
			}
			if weight < 0 {
				coefficients[offset] = int32(-0.5 + weight*(1<<pillowPrecisionBits))
			} else {
				coefficients[offset] = int32(0.5 + weight*(1<<pillowPrecisionBits))
			}
		}
		windows[index] = pillowWindow{first: first, coefficients: coefficients}
	}
	return windows
}

// pillowLanczos is Pillow's LANCZOS filter: sinc times sinc of a third, within three lobes.
func pillowLanczos(x float64) float64 {
	if x < -lanczosSupport || x >= lanczosSupport {
		return 0
	}
	return pillowSinc(x) * pillowSinc(x/lanczosSupport)
}

func pillowSinc(x float64) float64 {
	if x == 0 {
		return 1
	}
	x *= math.Pi
	return math.Sin(x) / x
}

// pillowLevel is a fixed-point sum's 8-bit level: shifted down, clipped to 0..255 (Pillow's clip8).
func pillowLevel(sum int32) uint8 {
	return uint8(min(max(sum>>pillowPrecisionBits, 0), UnitLevels))
}

// pillowLanczosRows resamples each row of a width by height RGB picture to output pixels (ImagingResampleHorizontal_8bpc).
func pillowLanczosRows(levels []uint8, width, height, output int) []uint8 {
	windows := pillowLanczosWindows(width, output)
	resampled := make([]uint8, output*height*RGBChannels)
	for y := range height {
		row := levels[y*width*RGBChannels:]
		for x, window := range windows {
			for channel := range RGBChannels {
				sum := int32(1 << (pillowPrecisionBits - 1))
				for offset, coefficient := range window.coefficients {
					sum += int32(row[(window.first+offset)*RGBChannels+channel]) * coefficient
				}
				resampled[(y*output+x)*RGBChannels+channel] = pillowLevel(sum)
			}
		}
	}
	return resampled
}

// pillowLanczosColumns resamples each column of a width by height RGB picture to output pixels
// (ImagingResampleVertical_8bpc).
func pillowLanczosColumns(levels []uint8, width, height, output int) []uint8 {
	windows := pillowLanczosWindows(height, output)
	resampled := make([]uint8, width*output*RGBChannels)
	for y, window := range windows {
		for x := range width {
			for channel := range RGBChannels {
				sum := int32(1 << (pillowPrecisionBits - 1))
				for offset, coefficient := range window.coefficients {
					sum += int32(levels[((window.first+offset)*width+x)*RGBChannels+channel]) * coefficient
				}
				resampled[(y*width+x)*RGBChannels+channel] = pillowLevel(sum)
			}
		}
	}
	return resampled
}
