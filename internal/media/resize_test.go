package media

import (
	"bytes"
	"image"
	"image/color"
	"testing"
)

// TestResizeBicubicReadsEachPixelAsAtDoes holds the resize to the reading it replaced, which took every tap of the
// horizontal pass through At: the same bytes for an RGBA source, a sub-image of one, an NRGBA source with partial
// alpha and a YCbCr source, shrunk and enlarged.
func TestResizeBicubicReadsEachPixelAsAtDoes(t *testing.T) {
	t.Parallel()
	const sourceWidth, sourceHeight = 37, 23
	rgba := image.NewRGBA(image.Rect(0, 0, sourceWidth, sourceHeight))
	nrgba := image.NewNRGBA(rgba.Bounds())
	ycbcr := image.NewYCbCr(rgba.Bounds(), image.YCbCrSubsampleRatio420)
	for y := range sourceHeight {
		for x := range sourceWidth {
			rgba.SetRGBA(x, y, color.RGBA{R: uint8(x * 7), G: uint8(y * 11), B: uint8(x*y + 3), A: 255})
			nrgba.SetNRGBA(x, y, color.NRGBA{R: uint8(x * 5), G: uint8(255 - y*3), B: uint8(x + y), A: uint8(x*y%255 + 1)})
			ycbcr.Y[ycbcr.YOffset(x, y)] = uint8(x*9 + y)
			ycbcr.Cb[ycbcr.COffset(x, y)], ycbcr.Cr[ycbcr.COffset(x, y)] = uint8(x*13), uint8(y*17)
		}
	}
	sources := map[string]image.Image{
		"rgba": rgba, "sub-image": rgba.SubImage(image.Rect(3, 2, 30, 21)), "nrgba": nrgba, "ycbcr": ycbcr,
	}
	for name, source := range sources {
		for _, size := range []image.Point{{X: 16, Y: 9}, {X: 61, Y: 40}} {
			got, want := ResizeBicubic(source, size.X, size.Y), resizeReadingAt(source, size.X, size.Y)
			if !bytes.Equal(got.Pix, want.Pix) {
				t.Errorf("%s to %v differs from the reading through At", name, size)
			}
		}
	}
}

// resizeReadingAt is the resize as it read before: every tap of the horizontal pass through At.
func resizeReadingAt(source image.Image, width, height int) *image.RGBA {
	bounds := source.Bounds()
	xMin, xCount, xWeights := antialiasWeights(bounds.Dx(), width)
	intermediate := make([][RGBChannels]uint8, bounds.Dy()*width)
	for y := range bounds.Dy() {
		for outX := range width {
			values := [RGBChannels]float64{}
			for offset := 0; offset < xCount[outX]; offset++ {
				r, g, b, _ := source.At(bounds.Min.X+xMin[outX]+offset, bounds.Min.Y+y).RGBA()
				weight := xWeights[outX][offset]
				values[0] += weight * float64(RGBAChannel8(r))
				values[1] += weight * float64(RGBAChannel8(g))
				values[2] += weight * float64(RGBAChannel8(b))
			}
			for channel := range values {
				intermediate[y*width+outX][channel] = RoundedUint8(values[channel])
			}
		}
	}
	yMin, yCount, yWeights := antialiasWeights(bounds.Dy(), height)
	output := image.NewRGBA(image.Rect(0, 0, width, height))
	for outY := range height {
		for x := range width {
			values := [RGBChannels]float64{}
			for offset := 0; offset < yCount[outY]; offset++ {
				pixel := intermediate[(yMin[outY]+offset)*width+x]
				for channel := range values {
					values[channel] += yWeights[outY][offset] * float64(pixel[channel])
				}
			}
			index := output.PixOffset(x, outY)
			for channel := range values {
				output.Pix[index+channel] = RoundedUint8(values[channel])
			}
			output.Pix[index+RGBChannels] = ^uint8(0)
		}
	}
	return output
}
