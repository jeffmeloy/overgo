//go:build windows

package latentimage

import (
	"bytes"
	"context"
	"testing"

	cudatest "overgo/internal/cuda/testutil"
	"overgo/internal/media"
	"overgo/internal/testcheck"
)

// TestASharpenedPictureReturnsThePictureItDrew holds moss-sharpened-pair/do: the MOSS Ultra workflow saves each
// section's picture as drawn and as its ImageSharpen sharpens it, and compares the two (text to image 449 and 482 in
// its comparer 709), and a sharpened request's picture comes with the picture it drew (EncodedImage.Drawn), bit for bit
// the same request's picture unsharpened, the sharpened one apart from it; an unsharpened request's comes with none. A
// sharpened picture had replaced its drawn pixels (generator_rotary_windows.go), so the drawn one was lost.
func TestASharpenedPictureReturnsThePictureItDrew(t *testing.T) {
	cudatest.Require(t)
	root, _ := compactGeneratorFixture(t)
	profile := testcheck.Must(ResolveProfile(root))(t)
	draw := func(request Request) EncodedImage {
		t.Helper()
		generator := testcheck.Must(LoadGenerator(t.Context(), root, profile, request))(t)
		defer func() { testcheck.NoError(t, generator.Close(context.Background())) }()
		session := testcheck.Must(generator.prepare(t.Context(), request))(t)
		session = testcheck.Must(generator.integrate(t.Context(), session))(t)
		return testcheck.Must(generator.decode(t.Context(), session))(t)
	}
	plain := Request{Prompt: "blue cube", NegativePrompt: "red", Width: 48, Height: 32, Steps: 3, Seed: 42}
	sharpened := plain
	sharpened.Sharpen = &media.Sharpening{Radius: 1, Sigma: 0.6, Alpha: 0.7}
	unsharpened, finished := draw(plain), draw(sharpened)
	if len(unsharpened.Drawn) != 0 {
		t.Error("an unsharpened picture came with a picture drawn before it")
	}
	if !bytes.Equal(finished.Drawn, unsharpened.Data) {
		t.Errorf("the sharpened picture came with %d bytes as the picture it drew; want the same request's unsharpened picture, %d bytes", len(finished.Drawn), len(unsharpened.Data))
	}
	if bytes.Equal(finished.Data, unsharpened.Data) {
		t.Error("the sharpened picture is the picture it drew")
	}
	before, kept := finished.BeforeFinishing()
	if drawn, isPicture := before.(EncodedImage); !kept || !isPicture || !bytes.Equal(drawn.Data, finished.Drawn) || drawn.Width != finished.Width || drawn.Height != finished.Height {
		t.Errorf("the sharpened picture gives %v (%t) as what it was before its finishing; want the picture it drew", before, kept)
	}
}
