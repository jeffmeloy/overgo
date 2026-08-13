package latentimage

import (
	"bytes"
	"image/png"
	"testing"
)

func TestEncodePNGStreamsHWC(t *testing.T) {
	pixels := []float32{-1, 0, 1, 1, 0, -1}
	got, err := encodePNG(pixels, 1, 2)
	if err != nil {
		t.Fatal(err)
	}
	if got.MediaType != "image/png" || got.Channels != 3 || got.Width != 2 || got.Height != 1 || len(got.Data) == 0 {
		t.Fatalf("encoded image=%+v", got)
	}
	decoded, err := png.Decode(bytes.NewReader(got.Data))
	if err != nil {
		t.Fatal(err)
	}
	r, g, b, _ := decoded.At(0, 0).RGBA()
	if r != 0 || g != 127*257 || b != 255*257 {
		t.Fatalf("pixel=(%d,%d,%d)", r, g, b)
	}
}
