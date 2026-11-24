package testutil

import (
	"bytes"
	"cmp"
	"image"
	"image/png"
	"math"
	"slices"
	"testing"

	"overgo/internal/testcheck"
)

// gridBlock is the side of the square blocks a picture's grid is read over.
const gridBlock = 32

// Luminance is a picture's Rec. 709 luminance in levels of 255, row by row.
func Luminance(picture image.Image) [][]float64 {
	bounds := picture.Bounds()
	plane := make([][]float64, bounds.Dy())
	for row := range plane {
		plane[row] = make([]float64, bounds.Dx())
		for column := range plane[row] {
			r, g, b, _ := picture.At(bounds.Min.X+column, bounds.Min.Y+row).RGBA()
			plane[row][column] = (0.2126*float64(r) + 0.7152*float64(g) + 0.0722*float64(b)) / 257
		}
	}
	return plane
}

// PNGLuminance is an encoded PNG's luminance (Luminance).
func PNGLuminance(t testing.TB, encoded []byte) [][]float64 {
	t.Helper()
	return Luminance(testcheck.Must(png.Decode(bytes.NewReader(encoded)))(t))
}

// PNGRGB is an encoded PNG's 8-bit RGB values, row by row.
func PNGRGB(t testing.TB, encoded []byte) []uint8 {
	t.Helper()
	decoded := testcheck.Must(png.Decode(bytes.NewReader(encoded)))(t)
	bounds := decoded.Bounds()
	values := make([]uint8, 0, bounds.Dx()*bounds.Dy()*3)
	for row := bounds.Min.Y; row < bounds.Max.Y; row++ {
		for column := bounds.Min.X; column < bounds.Max.X; column++ {
			r, g, b, _ := decoded.At(column, row).RGBA()
			values = append(values, uint8(r>>8), uint8(g>>8), uint8(b>>8))
		}
	}
	return values
}

// PixelGrid is a plane's 2-pixel grid in levels, as the owner's degrid program reads it (krea-degrid): the mean signed
// alternation along columns and along rows over its 32 by 32 blocks, or, smoothest, over the quarter of them whose
// values spread least, the flat areas a grid shows in.
func PixelGrid(plane [][]float64, smoothest bool) (columns, rows float64) {
	type block struct{ spread, columns, rows float64 }
	sign := func(n int) float64 { return float64(1 - 2*(n&1)) }
	var blocks []block
	cells := float64(gridBlock * gridBlock)
	for top := 0; top+gridBlock <= len(plane); top += gridBlock {
		for left := 0; left+gridBlock <= len(plane[0]); left += gridBlock {
			var b block
			mean, square := 0.0, 0.0
			for i := range gridBlock {
				for j := range gridBlock {
					v := plane[top+i][left+j]
					mean += v / cells
					square += v * v / cells
					b.columns += v * sign(left+j) / cells
					b.rows += v * sign(top+i) / cells
				}
			}
			b.spread = square - mean*mean
			blocks = append(blocks, b)
		}
	}
	if smoothest {
		slices.SortFunc(blocks, func(a, b block) int { return cmp.Compare(a.spread, b.spread) })
		blocks = blocks[:len(blocks)/4]
	}
	for _, b := range blocks {
		columns += b.columns / float64(len(blocks))
		rows += b.rows / float64(len(blocks))
	}
	return columns, rows
}

// PixelSteps is the root mean square step between neighbouring values along a plane's rows: its finest texture.
func PixelSteps(plane [][]float64) float64 {
	sum, count := 0.0, 0.0
	for _, row := range plane {
		for column := 0; column+1 < len(row); column++ {
			d := row[column+1] - row[column]
			sum += d * d
			count++
		}
	}
	return math.Sqrt(sum / count)
}
