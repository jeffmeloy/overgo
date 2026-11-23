//go:build windows

package mediacapability

import (
	"bytes"
	"context"
	"encoding/base64"
	"fmt"
	"image"
	"image/png"
	"math/rand/v2"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/latentimage"
	"overgo/internal/testcheck"
)

// standInEnhancer opens a chat standing in for MOSS's enhancer: it writes a long positive and a short negative.
func standInEnhancer(artifact.Repository) latentimage.EnhancerOpener {
	return func(context.Context, string) (func(context.Context, string, string, int) (string, error), func() error, error) {
		chat := func(_ context.Context, system, _ string, _ int) (string, error) {
			// The negative's systems ask "a concise negative prompt"; the positive's mentions negatives too.
			if strings.HasPrefix(system, "Write a concise negative prompt") {
				return "blurry edges, washed out colours", nil
			}
			return strings.Repeat("a red fox resting in fresh powder snow beside a quiet birch forest at first light ", 4), nil
		}
		return chat, func() error { return nil }, nil
	}
}

// TestAServedPictureOfAnySizeReturnsWhatItWasDrawnFrom holds draw-result-carries (SQA-464): a sharpened MOSS request
// carrying its picture inline, above a mebibyte, so its run records the request's digest in place of the request,
// served through the picture capability's serving (pictureExecutor, standInPictures) returns its picture, the prompt
// and negative it was drawn from and the picture it drew, both when drawn and when its stored run answers it; and an
// enhanced request returns what its enhancer wrote both times. The prompts had been read back from the run's stored
// request, so such a request's draw completed and its caller was refused.
func TestAServedPictureOfAnySizeReturnsWhatItWasDrawnFrom(t *testing.T) {
	pictures := newStandInPictures(t, standInEnhancer)
	noise, random := image.NewNRGBA(image.Rect(0, 0, 640, 640)), rand.New(rand.NewPCG(5, 464))
	for index := range noise.Pix {
		noise.Pix[index] = byte(random.Uint32())
	}
	var encoded bytes.Buffer
	testcheck.NoError(t, png.Encode(&encoded, noise))
	inline := base64.StdEncoding.EncodeToString(encoded.Bytes())
	raw := fmt.Sprintf(`{"prompt":"a lighthouse","negative_prompt":"grey sky","seed":5,"width":256,"height":144,"image":%q,"denoise":0.5,"sharpen":{"radius":1,"sigma":0.6,"alpha":0.7}}`, inline)
	if len(raw) <= 1<<20 {
		t.Fatalf("the request is %d bytes; want one above a mebibyte, which its run records as a digest", len(raw))
	}
	for _, served := range []string{"drawn", "answered by its stored run"} {
		measured, picture := pictures.served(t.Context(), raw)
		if !bytes.Contains(measured.Input.Data, []byte(`"request_sha256"`)) {
			t.Fatalf("the run recorded the request whole; want its digest")
		}
		if picture.Prompt != "a lighthouse" || picture.NegativePrompt != "grey sky" || !picture.Sharpened || len(picture.Drawn) == 0 {
			t.Errorf("the picture %s came with %q and %q, sharpened %t, %d bytes drawn; want its prompts and the picture it drew", served, picture.Prompt, picture.NegativePrompt, picture.Sharpened, len(picture.Drawn))
		}
	}
	if pictures.work.draws != 1 {
		t.Errorf("the request was drawn %d times; want once, its repeat answered by its stored run", pictures.work.draws)
	}
	idea := `{"prompt":"a fox","prompt_mode":"enhanced","seed":7,"width":256,"height":144}`
	for _, served := range []string{"drawn", "answered by its stored run"} {
		if _, picture := pictures.served(t.Context(), idea); !strings.Contains(picture.Prompt, "birch forest") || !strings.Contains(picture.NegativePrompt, "blurry edges") {
			t.Errorf("the enhanced picture %s came with %q and %q; want what its enhancer wrote", served, picture.Prompt, picture.NegativePrompt)
		}
	}
}
