package media

import (
	"image"
	"image/color"
	"image/color/palette"
	"image/draw"
	"math"
	"slices"
	"testing"
)

// TestAdaptivePaletteIsCutFromThePicture holds the median cut to what it is
// for. A picture of a few colours keeps each of them exactly. A smooth
// picture of far more colours than the palette is drawn, dithered, several
// times closer than under the fixed palette a clip used to be drawn in. The
// cut is the same cut twice, and asks for no more colours than it is given.
func TestAdaptivePaletteIsCutFromThePicture(t *testing.T) {
	t.Parallel()
	const side = 64
	few := image.NewRGBA(image.Rect(0, 0, side, side))
	colours := []color.RGBA{{R: 250, G: 10, B: 10, A: 255}, {R: 10, G: 250, B: 12, A: 255}, {R: 11, G: 12, B: 249, A: 255}, {R: 128, G: 127, B: 126, A: 255}}
	for y := range side {
		for x := range side {
			few.SetRGBA(x, y, colours[(x/16+y/16)%len(colours)])
		}
	}
	cut := AdaptivePalette(few, 256)
	if len(cut) != len(colours) {
		t.Fatalf("a picture of %d colours was cut into %d", len(colours), len(cut))
	}
	for _, colour := range colours {
		if !slices.Contains(cut, color.Color(colour)) {
			t.Fatalf("the palette %v lost %v", cut, colour)
		}
	}

	smooth := image.NewRGBA(image.Rect(0, 0, side, side))
	for y := range side {
		for x := range side {
			smooth.SetRGBA(x, y, color.RGBA{R: uint8(60 + 2*x), G: uint8(40 + 3*y), B: uint8(200 - x - y), A: 255})
		}
	}
	const colors = 256
	adaptive := AdaptivePalette(smooth, colors)
	if len(adaptive) > colors || len(adaptive) < colors/2 || !slices.Equal(adaptive, AdaptivePalette(smooth, colors)) {
		t.Fatalf("a smooth picture was cut into %d colours, or into other colours the second time", len(adaptive))
	}
	if small := AdaptivePalette(smooth, 16); len(small) != 16 {
		t.Fatalf("sixteen colours asked, %d given", len(small))
	}
	distance := func(colours color.Palette) float64 {
		drawn := image.NewPaletted(smooth.Bounds(), colours)
		draw.FloydSteinberg.Draw(drawn, smooth.Bounds(), smooth, image.Point{})
		var squared float64
		for y := range side {
			for x := range side {
				want := smooth.RGBAAt(x, y)
				red, green, blue, _ := drawn.At(x, y).RGBA()
				for _, pair := range [][2]float64{{float64(red >> 8), float64(want.R)}, {float64(green >> 8), float64(want.G)}, {float64(blue >> 8), float64(want.B)}} {
					squared += (pair[0] - pair[1]) * (pair[0] - pair[1])
				}
			}
		}
		return 20 * math.Log10(255/math.Sqrt(squared/(3*side*side)))
	}
	cutRatio, fixedRatio := distance(adaptive), distance(palette.Plan9)
	t.Logf("a smooth picture dithered: %.1f dB under its own palette, %.1f dB under the fixed one", cutRatio, fixedRatio)
	if cutRatio < fixedRatio+8 {
		t.Fatalf("its own palette draws the picture at %.1f dB and the fixed one at %.1f dB, want its own far closer", cutRatio, fixedRatio)
	}
	if AdaptivePalette(image.NewRGBA(image.Rectangle{}), colors) != nil || AdaptivePalette(smooth, 0) != nil {
		t.Fatal("a palette was cut from no pixels or for no colours")
	}
}
