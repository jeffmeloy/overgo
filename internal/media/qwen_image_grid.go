package media

import (
	"errors"
	"math"
	"sync"
)

// The Qwen-Image VAE's decoder (AutoencoderKLQwenImage, Krea-2-Turbo's) leaves a 2-pixel grid: its rows and columns
// alternate by about 0.3 of a level over a picture and 0.15 in flat areas, in diffusers' own pipeline as in overgo's
// (krea-degrid). RemoveQwenImageGrid takes it away with the local finish the owner measured: what alternates pixel to
// pixel along a row or a column is isolated, its signed strength averaged over a neighbourhood, where the grid keeps
// one phase and texture does not, and that average subtracted, limited to a few levels. Removing the whole 2-pixel
// frequency lost 6 to 10% of the finest texture, and one pattern for the whole picture inverted the grid in flat areas.
const (
	// qwenImageGridRadius is the half side of the square the grid's strength is averaged over: 17 pixels.
	qwenImageGridRadius = 8
	// qwenImageGridLimit is the most the finish moves a value: 2 levels of 255.
	qwenImageGridLimit = 2.0 / 255
	// qwenImageGridTaps is the half length of alternatingKernel.
	qwenImageGridTaps = 4
	// qwenImageGridScale is the sum of alternatingKernel's magnitudes, 2^8, so its response at the 2-pixel period is 1.
	qwenImageGridScale = 256
)

// alternatingKernel is the binomial (1, 8, 28, 56, 70, 56, 28, 8, 1) with alternating signs: its response is 1 at the
// 2-pixel period and falls as the eighth power of a sine toward lower frequencies.
var alternatingKernel = [2*qwenImageGridTaps + 1]float64{1, -8, 28, -56, 70, -56, 28, -8, 1}

// RemoveQwenImageGrid is HWC unit RGB pixels of height by width with the Qwen-Image decoder's 2-pixel grid taken
// away, each channel apart: per pixel, the alternation along the row and along the column, each read by
// alternatingKernel with the border mirrored by an even distance so it keeps its phase, signed by the pixel's column or
// row parity and averaged over the 17-pixel square around it, is subtracted, limited to 2 levels. pixels is not changed.
func RemoveQwenImageGrid(pixels []float32, height, width int) ([]float32, error) {
	if height <= 0 || width <= 0 || len(pixels) != height*width*RGBChannels {
		return nil, errors.New("media: the Qwen-Image grid finish takes HWC RGB pixels of its height and width")
	}
	out := make([]float32, len(pixels))
	var channels sync.WaitGroup
	for channel := range RGBChannels {
		channels.Go(func() {
			plane := make([]float64, height*width)
			for index := range plane {
				plane[index] = float64(pixels[index*RGBChannels+channel])
			}
			columns := gridStrength(plane, height, width, false)
			rows := gridStrength(plane, height, width, true)
			for row := range height {
				for column := range width {
					index := row*width + column
					correction := columns[index]*parity(column) + rows[index]*parity(row)
					correction = math.Max(-qwenImageGridLimit, math.Min(qwenImageGridLimit, correction))
					out[index*RGBChannels+channel] = float32(plane[index] - correction)
				}
			}
		})
	}
	channels.Wait()
	return out, nil
}

// parity is +1 at an even index and -1 at an odd one: the grid's sign.
func parity(index int) float64 { return float64(1 - 2*(index&1)) }

// gridStrength is, at each pixel of plane, the alternation along its row (vertical false) or its column, signed by the
// pixel's parity along it, averaged over the square of side 2 qwenImageGridRadius + 1 around the pixel, cut at the
// picture's edge.
func gridStrength(plane []float64, height, width int, vertical bool) []float64 {
	mirror := func(at, length int) int {
		if at < 0 {
			at = -at
		}
		if at >= length {
			at = 2*(length-1) - at
		}
		return min(max(at, 0), length-1)
	}
	signed := make([]float64, len(plane))
	for row := range height {
		for column := range width {
			sum := 0.0
			for tap, weight := range alternatingKernel {
				r, c := row, column
				if vertical {
					r = mirror(row+tap-qwenImageGridTaps, height)
				} else {
					c = mirror(column+tap-qwenImageGridTaps, width)
				}
				sum += weight * plane[r*width+c]
			}
			along := column
			if vertical {
				along = row
			}
			signed[row*width+column] = sum / qwenImageGridScale * parity(along)
		}
	}
	return boxAverage(signed, height, width, qwenImageGridRadius)
}

// boxAverage is values averaged over the square of side 2 radius + 1 around each cell, cut at the edge, by running sums
// along the rows and then the columns.
func boxAverage(values []float64, height, width, radius int) []float64 {
	across := make([]float64, len(values))
	for row := range height {
		line := values[row*width : (row+1)*width]
		sum, count := 0.0, 0
		for column := range min(radius, width) {
			sum += line[column]
			count++
		}
		for column := range width {
			if entering := column + radius; entering < width {
				sum += line[entering]
				count++
			}
			if leaving := column - radius - 1; leaving >= 0 {
				sum -= line[leaving]
				count--
			}
			across[row*width+column] = sum / float64(count)
		}
	}
	out := make([]float64, len(values))
	for column := range width {
		sum, count := 0.0, 0
		for row := range min(radius, height) {
			sum += across[row*width+column]
			count++
		}
		for row := range height {
			if entering := row + radius; entering < height {
				sum += across[entering*width+column]
				count++
			}
			if leaving := row - radius - 1; leaving >= 0 {
				sum -= across[leaving*width+column]
				count--
			}
			out[row*width+column] = sum / float64(count)
		}
	}
	return out
}
