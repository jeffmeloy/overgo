package media

import (
	"fmt"
	"math"
	"strconv"
	"strings"

	"overgo/internal/checked"
)

// sharpenGain is the factor the alpha of a sharpening scales the subtracted blur by.
const sharpenGain = 10

// megapixelSide is the side of a square megapixel, as ComfyUI counts one in sizing a canvas: 1024 pixels.
const megapixelSide = 1024

// Sharpening is how a picture is sharpened after it is drawn, as ComfyUI's ImageSharpen takes it (the MOSS Ultra
// workflow's is radius 1, sigma 0.6 and alpha 0.7).
type Sharpening struct {
	Radius int     `json:"radius"`
	Sigma  float64 `json:"sigma"`
	Alpha  float64 `json:"alpha"`
}

// Validate holds a sharpening to a radius of 1 or more, a positive sigma and an alpha of 0 or more.
func (s Sharpening) Validate() error {
	if s.Radius < 1 || !checked.PositiveFinite64(s.Sigma) || !checked.NonNegativeFinite64(s.Alpha) {
		return fmt.Errorf("media: a sharpening has a radius of 1 or more, a positive sigma and an alpha of 0 or more, not %d, %g and %g", s.Radius, s.Sigma, s.Alpha)
	}
	return nil
}

// Sharpen sharpens a picture of unit RGB, height × width pixels of RGBChannels interleaved, as ComfyUI's ImageSharpen
// does (moss-ultra-sections/finishing): an unsharp mask whose kernel is (1 + 10α) at its centre less 10α times a
// Gaussian blur, the blur's 2·radius + 1 taps a side lying at even steps over [−1, 1] with deviation sigma and summing
// to 1, the picture's edges reflected (an edge pixel not repeated), and each value clamped to [0, 1].
func Sharpen(pixels []float32, height, width int, s Sharpening) ([]float32, error) {
	if err := s.Validate(); err != nil {
		return nil, err
	}
	if height <= 0 || width <= 0 || len(pixels) != height*width*RGBChannels || s.Radius >= min(height, width) {
		return nil, fmt.Errorf("media: sharpening at radius %d needs %d×%d RGB pixels and a picture wider and taller than the radius, not %d values", s.Radius, height, width, len(pixels))
	}
	size := 2*s.Radius + 1
	kernel := make([]float64, size*size)
	var total float64
	for i := range size {
		for j := range size {
			y := -1 + 2*float64(i)/float64(size-1)
			x := -1 + 2*float64(j)/float64(size-1)
			kernel[i*size+j] = math.Exp(-(x*x + y*y) / (2 * s.Sigma * s.Sigma))
			total += kernel[i*size+j]
		}
	}
	for i := range kernel {
		kernel[i] *= -sharpenGain * s.Alpha / total
	}
	kernel[s.Radius*size+s.Radius] += 1 + sharpenGain*s.Alpha
	reflect := func(at, extent int) int {
		if at < 0 {
			return -at
		}
		if at >= extent {
			return 2*(extent-1) - at
		}
		return at
	}
	sharpened := make([]float32, len(pixels))
	for y := range height {
		for x := range width {
			for channel := range RGBChannels {
				var value float64
				for i := range size {
					row := reflect(y+i-s.Radius, height) * width
					for j := range size {
						value += kernel[i*size+j] * float64(pixels[(row+reflect(x+j-s.Radius, width))*RGBChannels+channel])
					}
				}
				sharpened[(y*width+x)*RGBChannels+channel] = float32(min(max(value, 0), 1))
			}
		}
	}
	return sharpened, nil
}

// Canvas is a picture's size asked as ComfyUI's ResolutionSelector asks it: an aspect ratio ("16:9"), megapixels and
// the multiple each side is rounded to.
type Canvas struct {
	Aspect     string  `json:"aspect"`
	Megapixels float64 `json:"megapixels"`
	Multiple   int     `json:"multiple"`
}

// Size is the canvas's width and height (AspectSize).
func (c Canvas) Size() (width, height int, err error) {
	across, down, found := strings.Cut(c.Aspect, ":")
	ratioWidth, widthErr := strconv.Atoi(across)
	ratioHeight, heightErr := strconv.Atoi(down)
	if !found || widthErr != nil || heightErr != nil {
		return 0, 0, fmt.Errorf("media: a canvas's aspect is a ratio of two whole numbers such as 16:9, not %q", c.Aspect)
	}
	return AspectSize(ratioWidth, ratioHeight, c.Megapixels, c.Multiple)
}

// AspectSize is the canvas ComfyUI's ResolutionSelector sizes for megapixels at the aspect ratio ratioWidth:ratioHeight:
// each side the one a canvas of that ratio and area takes, the area megapixels of 1024 × 1024 pixels, rounded to the
// nearest multiple of multiple, a half rounding to even.
func AspectSize(ratioWidth, ratioHeight int, megapixels float64, multiple int) (width, height int, err error) {
	if !checked.PositiveInts(ratioWidth, ratioHeight, multiple) || !checked.PositiveFinite64(megapixels) {
		return 0, 0, fmt.Errorf("media: a canvas is sized from a positive ratio, megapixels and multiple, not %d:%d, %g and %d", ratioWidth, ratioHeight, megapixels, multiple)
	}
	area := megapixels * megapixelSide * megapixelSide
	ratio := float64(ratioWidth) / float64(ratioHeight)
	side := func(length float64) int { return int(math.RoundToEven(length/float64(multiple))) * multiple }
	width, height = side(math.Sqrt(area*ratio)), side(math.Sqrt(area/ratio))
	if width <= 0 || height <= 0 {
		return 0, 0, fmt.Errorf("media: %g megapixels at %d:%d round to a %d×%d canvas at multiples of %d", megapixels, ratioWidth, ratioHeight, width, height, multiple)
	}
	return width, height, nil
}
