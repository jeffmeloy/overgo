//go:build windows

package latentimage

import (
	"bytes"
	"encoding/base64"
	"image"
	"image/png"
	"math"
	"strings"
	"testing"

	"overgo/internal/testcheck"
)

// TestAFullDenoiseEncodesNoSource holds a picture request whose schedule starts at σ₀ of 1 to starting from its noise
// without encoding its picture (SQA-429), as the image to image section's recording does at denoise 1: the start is
// the noise bit for bit, though no model is there to build an encoder from; a picture that is no PNG is still refused;
// and a schedule starting below 1, as the Turbo section's at about 0.90, still encodes its picture, which fails here
// for want of the model's VAE.
func TestAFullDenoiseEncodesNoSource(t *testing.T) {
	var picture bytes.Buffer
	testcheck.NoError(t, png.Encode(&picture, image.NewRGBA(image.Rect(0, 0, 16, 16))))
	noise := []float32{0.5, -1.25, 3, float32(math.Copysign(0, -1))}
	start := func(encoded string, first float64) ([]float32, error) {
		generator := &Generator{request: Request{Image: encoded, Width: 16, Height: 16}, modelDir: t.TempDir(),
			schedule: FlowSchedule{Sigmas: []float64{first, 0}}}
		return generator.startLatent(t.Context(), noise)
	}
	encoded := base64.StdEncoding.EncodeToString(picture.Bytes())
	started := testcheck.Must(start(encoded, 1))(t)
	for index, value := range started {
		if math.Float32bits(value) != math.Float32bits(noise[index]) {
			t.Fatalf("a full-denoise start is %v; want its noise %v bit for bit", started, noise)
		}
	}
	if _, err := start(base64.StdEncoding.EncodeToString([]byte("no picture")), 1); err == nil || !strings.Contains(err.Error(), "picture") {
		t.Errorf("a full-denoise request whose picture is no PNG started (%v); want it refused for its picture", err)
	}
	if _, err := start(encoded, 0.90027); err == nil {
		t.Error("a request starting at σ₀ 0.90 started with no model to encode its picture; want its picture encoded")
	}
}
