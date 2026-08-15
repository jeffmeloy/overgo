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
	content, err := PNGContent(got)
	if err != nil || content.Descriptor.MediaType != got.MediaType || !bytes.Equal(content.Data, got.Data) {
		t.Fatalf("PNG content=(%+v, %v)", content.Descriptor, err)
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

func TestEncodePNGAcceptsFiniteLowContrast(t *testing.T) {
	pixels := []float32{0.25, 0.25, 0.25}
	got, err := encodePNG(pixels, 1, 1)
	if err != nil {
		t.Fatal(err)
	}
	if got.Minimum != 0.25 || got.Maximum != 0.25 {
		t.Fatalf("range=[%g,%g]", got.Minimum, got.Maximum)
	}
	if _, err := png.Decode(bytes.NewReader(got.Data)); err != nil {
		t.Fatal(err)
	}
}

func TestEncodePlanarPNGMatchesHWC(t *testing.T) {
	hwc := []float32{-1, 0, 1, 1, 0, -1}
	planar := []float32{-1, 1, 0, 0, 1, -1}
	want, err := encodePNG(hwc, 1, 2)
	if err != nil {
		t.Fatal(err)
	}
	got, err := EncodePlanarPNG(planar, 1, 2)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(got.Data, want.Data) || got.Width != want.Width || got.Height != want.Height || got.Channels != 3 {
		t.Fatalf("planar publication differs: got=%+v want=%+v", got, want)
	}
}
