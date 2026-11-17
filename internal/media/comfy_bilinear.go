package media

import (
	"fmt"

	"overgo/internal/checked"
	"overgo/internal/hostmath"
)

// BilinearLikeComfyUI is a height by width picture of channels interleaved float32 values scaled to outHeight by
// outWidth as ComfyUI's ImageScale scales it at bilinear without a crop (torch.nn.functional.interpolate on the host,
// corners not aligned, no antialias): each output at the source position (x + 0.5) * in / out - 0.5, at least 0, its
// scale in float32, its two neighbours across weighted, then the two rows down, in float32.
func BilinearLikeComfyUI(values []float32, height, width, channels, outHeight, outWidth int) ([]float32, error) {
	if !checked.PositiveInts(height, width, channels, outHeight, outWidth) || len(values) != height*width*channels {
		return nil, fmt.Errorf("media: %d values are not a %dx%d picture of %d channels", len(values), width, height, channels)
	}
	type tap struct {
		first, next int
		low, high   float32
	}
	taps := func(input, output int) []tap {
		scale := float32(input) / float32(output)
		out := make([]tap, output)
		for index := range output {
			real := max(scale*(float32(index)+0.5)-0.5, 0)
			first := int(real)
			next := 0
			if first < input-1 {
				next = 1
			}
			high := real - float32(first)
			out[index] = tap{first: first, next: next, low: 1 - high, high: high}
		}
		return out
	}
	rows, columns := taps(height, outHeight), taps(width, outWidth)
	out := make([]float32, outHeight*outWidth*channels)
	hostmath.ParallelRangeF64(outHeight, outWidth*channels*4, func(start, end int) {
		for y := start; y < end; y++ {
			down := rows[y]
			upper := values[down.first*width*channels:]
			lower := values[(down.first+down.next)*width*channels:]
			for x, across := range columns {
				for c := range channels {
					left, right := across.first*channels+c, (across.first+across.next)*channels+c
					top := upper[left]*across.low + upper[right]*across.high
					bottom := lower[left]*across.low + lower[right]*across.high
					out[(y*outWidth+x)*channels+c] = top*down.low + bottom*down.high
				}
			}
		}
	})
	return out, nil
}
