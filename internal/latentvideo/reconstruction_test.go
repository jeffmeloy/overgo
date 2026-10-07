package latentvideo

import (
	"bytes"
	"image"
	"image/color"
	"image/png"
	"math"
	"testing"

	"overgo/internal/media"
)

// TestAPictureIsExtendedToTheStrideAndCroppedBack holds the two host steps
// around the autoencoder: a picture whose sides are not multiples of the
// stride is extended by repeating its last row and column, so the encoder
// sees no invented edge, and cropping what comes back restores the picture's
// own size with every pixel where it was. A picture that already fits is
// passed through untouched.
func TestAPictureIsExtendedToTheStrideAndCroppedBack(t *testing.T) {
	t.Parallel()
	const height, width = 5, 11
	picture := image.NewRGBA(image.Rect(0, 0, width, height))
	for y := range height {
		for x := range width {
			picture.SetRGBA(x, y, color.RGBA{R: uint8(20 * x), G: uint8(50 * y), B: uint8(x + y), A: math.MaxUint8})
		}
	}
	var encoded bytes.Buffer
	if err := png.Encode(&encoded, picture); err != nil {
		t.Fatal(err)
	}
	source, gotHeight, gotWidth, err := sourcePicture(encoded.Bytes(), 8, 8)
	if err != nil {
		t.Fatal(err)
	}
	if gotHeight != height || gotWidth != width || source.Height != 8 || source.Width != 16 ||
		source.Channels != media.RGBChannels || source.Frames != 1 || len(source.Pixels) != media.RGBChannels*8*16 {
		t.Fatalf("a %dx%d picture became %dx%d of %dx%dx%d with %d values", height, width, gotHeight, gotWidth,
			source.Channels, source.Height, source.Width, len(source.Pixels))
	}
	at := func(channel, y, x int) float32 { return source.Pixels[(channel*source.Height+y)*source.Width+x] }
	for channel := range media.RGBChannels {
		for y := range source.Height {
			for x := range source.Width {
				inside := picture.RGBAAt(min(x, width-1), min(y, height-1))
				want := media.U8ToSignedUnitF32([]uint8{inside.R, inside.G, inside.B}[channel])
				if got := at(channel, y, x); got != want {
					t.Fatalf("channel %d at row %d column %d = %v, want %v", channel, y, x, got, want)
				}
			}
		}
	}
	cropped := cropPicture(source, height, width)
	if len(cropped) != media.RGBChannels*height*width {
		t.Fatalf("the cropped picture has %d values", len(cropped))
	}
	for channel := range media.RGBChannels {
		for y := range height {
			for x := range width {
				if got, want := cropped[(channel*height+y)*width+x], at(channel, y, x); got != want {
					t.Fatalf("cropped channel %d at row %d column %d = %v, want %v", channel, y, x, got, want)
				}
			}
		}
	}
	if whole := cropPicture(source, source.Height, source.Width); &whole[0] != &source.Pixels[0] {
		t.Fatal("a picture that needs no crop was copied")
	}
	if _, _, _, err := sourcePicture([]byte("not a picture"), 8, 8); err == nil {
		t.Fatal("bytes that are no picture were decoded")
	}
	if err := ValidateReconstructionRequest(ReconstructionRequest{}); err == nil {
		t.Fatal("a request with no picture was accepted")
	}
	if stats, err := declaredLatentStats(16); err != nil || len(stats.Mean) != 16 || len(stats.Std) != 16 {
		t.Fatalf("the declared latent statistics for 16 channels = %d means, %d spreads, %v", len(stats.Mean), len(stats.Std), err)
	}
	if _, err := declaredLatentStats(4); err == nil {
		t.Fatal("latent statistics for 16 channels were given to an autoencoder of 4")
	}
}
