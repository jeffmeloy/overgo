package latentimage

import (
	"bytes"
	"math"
	"testing"

	"overgo/internal/media"
	"overgo/internal/testcheck"
)

// TestASharpenedPairIsEncodedTogether holds pair-encode/do: a picture finished with a sharpening gives its sharpened
// PNG and, beside it, the PNG of the picture as drawn, the two encoded together (finishedPNGs) and each byte for byte
// the PNG its picture encodes alone; a picture finished with none gives its PNG alone. The drawn picture had been
// encoded before the sharpening, each PNG deflate's work in turn, about 0.2 s of each of the MOSS Ultra workflow's six
// sections (moss-pass-repeats/clock, a273ae6eb); the workflow's clock reads the time the two save together.
func TestASharpenedPairIsEncodedTogether(t *testing.T) {
	t.Parallel()
	const height, width = 48, 64
	pixels := make([]float32, height*width*media.RGBChannels)
	for index := range pixels {
		pixels[index] = float32(0.5 + 0.45*math.Sin(float64(index)/7.3)*math.Cos(float64(index)/131))
	}
	sharpening := media.Sharpening{Radius: 1, Sigma: 0.6, Alpha: 0.7}
	finished, drawn, err := finishedPNGs(pixels, height, width, &sharpening)
	testcheck.NoError(t, err)
	alone := testcheck.Must(encodeUnitPNG(pixels, height, width))(t)
	sharpened := testcheck.Must(media.Sharpen(pixels, height, width, sharpening))(t)
	sharpenedAlone := testcheck.Must(encodeUnitPNG(sharpened, height, width))(t)
	if !bytes.Equal(drawn.Data, alone.Data) {
		t.Errorf("the drawn picture's PNG encoded beside the sharpened one is %d bytes unlike the %d it encodes alone", len(drawn.Data), len(alone.Data))
	}
	if !bytes.Equal(finished.Data, sharpenedAlone.Data) || bytes.Equal(finished.Data, alone.Data) {
		t.Errorf("the sharpened picture's PNG encoded beside the drawn one is not the PNG it encodes alone")
	}
	plain, none, err := finishedPNGs(pixels, height, width, nil)
	testcheck.NoError(t, err)
	if !bytes.Equal(plain.Data, alone.Data) || none.Data != nil {
		t.Errorf("a picture finished with no sharpening gave %d bytes and a drawn PNG of %d; want its PNG alone", len(plain.Data), len(none.Data))
	}
}
