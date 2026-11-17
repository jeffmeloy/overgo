package media

import (
	"fmt"
	"math"

	"overgo/internal/checked"
	"overgo/internal/hostmath"
)

// OpenCV's resizes as the Depth Anything V2 preprocessor of comfyui_controlnet_aux runs them (cv2.resize): bicubic
// on a picture of float64 values (its model's input), and bicubic and area on 8-bit levels (its map's resize to the
// detect resolution), reproduced from OpenCV's resize.cpp: the source position of each output (x + 0.5) * scale - 0.5 in
// float32, the cubic's four weights from Keys' kernel at A = -0.75 in float32 with the last the remainder of 1, edges
// replicated, the rows resampled across before down; the area's weights the share of each source cell an output cell
// covers, in float32, summed in float32 and rounded to the nearest level.

// CubicResizeFloat64 is a height by width picture of channels interleaved float64 values resized to outHeight by
// outWidth as cv2.resize resizes it at INTER_CUBIC.
func CubicResizeFloat64(values []float64, height, width, channels, outHeight, outWidth int) ([]float64, error) {
	if !checked.PositiveInts(height, width, channels, outHeight, outWidth) || len(values) != height*width*channels {
		return nil, fmt.Errorf("media: %d values are not a %dx%d picture of %d channels", len(values), width, height, channels)
	}
	columns, columnWeights := openCVCubicTaps(width, outWidth)
	rows, rowWeights := openCVCubicTaps(height, outHeight)
	// Each source row the output reads, resampled across.
	across := make([]float64, height*outWidth*channels)
	hostmath.ParallelRangeF64(height, outWidth*channels*4, func(start, end int) {
		for y := start; y < end; y++ {
			row := values[y*width*channels:]
			for x := range outWidth {
				for c := range channels {
					sum := 0.0
					for k, weight := range columnWeights[x] {
						sum += row[columns[x][k]*channels+c] * float64(weight)
					}
					across[(y*outWidth+x)*channels+c] = sum
				}
			}
		}
	})
	out := make([]float64, outHeight*outWidth*channels)
	hostmath.ParallelRangeF64(outHeight, outWidth*channels*4, func(start, end int) {
		for y := start; y < end; y++ {
			for x := range outWidth * channels {
				sum := 0.0
				for k, weight := range rowWeights[y] {
					sum += across[rows[y][k]*outWidth*channels+x] * float64(weight)
				}
				out[y*outWidth*channels+x] = sum
			}
		}
	})
	return out, nil
}

// CubicResizeLevels is a height by width picture of channels interleaved 8-bit levels resized to outHeight by outWidth
// as cv2.resize (OpenCV 5) resizes it at INTER_CUBIC: the float64 resize of the levels, each rounded to the nearest
// level and clipped. OpenCV 5 sums its 8-bit cubic in floating point; the fixed point of its earlier versions (weights
// times 2048, sums shifted off 22 bits) misses its levels by one in about 4% of values.
func CubicResizeLevels(levels []uint8, height, width, channels, outHeight, outWidth int) ([]uint8, error) {
	values := make([]float64, len(levels))
	for index, level := range levels {
		values[index] = float64(level)
	}
	resized, err := CubicResizeFloat64(values, height, width, channels, outHeight, outWidth)
	if err != nil {
		return nil, err
	}
	out := make([]uint8, len(resized))
	for index, value := range resized {
		out[index] = uint8(min(max(math.RoundToEven(value), 0), 255))
	}
	return out, nil
}

// openCVCubicTaps are, for each of output positions over input, the four input positions OpenCV's cubic reads, replicated
// at the edges, and their float32 weights.
func openCVCubicTaps(input, output int) ([][4]int, [][4]float32) {
	scale := float64(input) / float64(output)
	positions, weights := make([][4]int, output), make([][4]float32, output)
	for index := range output {
		real := float32((float64(index)+0.5)*scale - 0.5)
		first := int(math.Floor(float64(real)))
		fraction := real - float32(first)
		weights[index] = openCVCubic(fraction)
		for k := range 4 {
			positions[index][k] = min(max(first-1+k, 0), input-1)
		}
	}
	return positions, weights
}

// openCVCubic is OpenCV's interpolateCubic: Keys' kernel at A = -0.75 for the three nearer taps, the fourth the
// remainder of 1, in float32.
func openCVCubic(x float32) [4]float32 {
	const a = float32(-0.75)
	var coefficients [4]float32
	coefficients[0] = ((a*(x+1)-5*a)*(x+1)+8*a)*(x+1) - 4*a
	coefficients[1] = ((a+2)*x-(a+3))*x*x + 1
	coefficients[2] = ((a+2)*(1-x)-(a+3))*(1-x)*(1-x) + 1
	coefficients[3] = 1 - coefficients[0] - coefficients[1] - coefficients[2]
	return coefficients
}

// AreaResizeLevels is a height by width picture of channels interleaved 8-bit levels resized down to outHeight by
// outWidth as cv2.resize resizes it at INTER_AREA where a side's scale is no whole number (resizeArea_): each output
// the mean of the source cells its cell covers, each weighted by the share it covers, in float32, rounded to the
// nearest level.
func AreaResizeLevels(levels []uint8, height, width, channels, outHeight, outWidth int) ([]uint8, error) {
	if !checked.PositiveInts(height, width, channels, outHeight, outWidth) || len(levels) != height*width*channels || outHeight > height || outWidth > width {
		return nil, fmt.Errorf("media: %d levels are not a %dx%d picture of %d channels to shrink to %dx%d", len(levels), width, height, channels, outWidth, outHeight)
	}
	columns := areaTable(width, outWidth)
	rows := areaTable(height, outHeight)
	out := make([]uint8, outHeight*outWidth*channels)
	hostmath.ParallelRangeF64(outHeight, outWidth*channels*8, func(start, end int) {
		sums := make([]float32, outWidth*channels)
		across := make([]float32, outWidth*channels)
		for y := start; y < end; y++ {
			clear(sums)
			for _, row := range rows[y] {
				clear(across)
				source := levels[row.source*width*channels:]
				for x := range outWidth {
					for _, column := range columns[x] {
						for c := range channels {
							across[x*channels+c] += float32(source[column.source*channels+c]) * column.weight
						}
					}
				}
				for index, value := range across {
					sums[index] += value * row.weight
				}
			}
			for index, value := range sums {
				out[y*outWidth*channels+index] = uint8(min(max(math.RoundToEven(float64(value)), 0), 255))
			}
		}
	})
	return out, nil
}

// areaCell is a source cell an output cell covers and the share of the output cell it is.
type areaCell struct {
	source int
	weight float32
}

// areaTable is OpenCV's computeResizeAreaTab: for each output cell of a side, the source cells it covers, whole and
// in part, each weighted by the share of the output cell it is.
func areaTable(input, output int) [][]areaCell {
	scale := float64(input) / float64(output)
	table := make([][]areaCell, output)
	for index := range output {
		first := float64(index) * scale
		last := first + scale
		cell := min(scale, float64(input)-first)
		whole := int(math.Ceil(first))
		end := min(int(math.Floor(last)), input-1)
		whole = min(whole, end)
		var cells []areaCell
		if float64(whole)-first > 1e-3 {
			cells = append(cells, areaCell{whole - 1, float32((float64(whole) - first) / cell)})
		}
		for source := whole; source < end; source++ {
			cells = append(cells, areaCell{source, float32(1 / cell)})
		}
		if last-float64(end) > 1e-3 {
			cells = append(cells, areaCell{end, float32(min(min(last-float64(end), 1), cell) / cell)})
		}
		table[index] = cells
	}
	return table
}
