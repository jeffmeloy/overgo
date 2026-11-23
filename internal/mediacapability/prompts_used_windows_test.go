//go:build windows

package mediacapability

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"image"
	"image/png"
	"math/rand/v2"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/capabilityruntime"
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

// TestARepeatedEnhancedPictureOpensNoChat holds enhanced-repeat (SQA-465): an enhanced MOSS request, its idea and
// exclusions, served twice through the picture capability's serving (pictureExecutor, standInPictures) opens its
// enhancer chat once: the repeat is answered by the first's stored run, its picture and the prompts the enhancer wrote,
// which travel in the picture's bytes; a fresh run opens the chat again. The run records the idea, and its answer is
// kept under the enhancer folder's files as well, so another enhancer writes again. The chat had been opened and run
// before the stored answer was sought, which the rewritten request then named.
func TestARepeatedEnhancedPictureOpensNoChat(t *testing.T) {
	chats := 0
	counted := func(store artifact.Repository) latentimage.EnhancerOpener {
		open := standInEnhancer(store)
		return func(ctx context.Context, directory string) (func(context.Context, string, string, int) (string, error), func() error, error) {
			chats++
			return open(ctx, directory)
		}
	}
	pictures := newStandInPictures(t, counted)
	enhancer := filepath.Join(pictures.root, "enhancer", "enhancer.gguf")
	testcheck.NoError(t, os.MkdirAll(filepath.Dir(enhancer), 0o755))
	testcheck.NoError(t, os.WriteFile(enhancer, []byte("the enhancer's weights"), 0o600))
	const idea = `{"prompt":"a fox","negative_prompt":"grey sky","prompt_mode":"enhanced","seed":7,"width":256,"height":144}`
	first, drawn := pictures.served(t.Context(), idea)
	repeat, again := pictures.served(t.Context(), idea)
	if chats != 1 || pictures.work.draws != 1 || repeat.Run != first.Run || again.Prompt != drawn.Prompt || again.NegativePrompt != drawn.NegativePrompt ||
		!strings.Contains(drawn.Prompt, "birch forest") {
		t.Errorf("the repeat opened %d chats in all, drew %d times, gave run %s (the first %s) and %q; want one chat, one draw, the first's run and its enhancer's prompt",
			chats, pictures.work.draws, repeat.Run, first.Run, again.Prompt)
	}
	var recorded latentimage.Request
	testcheck.NoError(t, json.Unmarshal(first.Input.Data, &recorded))
	if !recorded.Enhanced() || recorded.Prompt != "a fox" {
		t.Errorf("the run recorded %+v; want the idea as given", recorded)
	}
	if named := testcheck.Must(pictureFiles(t.Context(), pictures.store, pictures.root, pictures.execution, idea))(t); !slices.Contains(named, enhancer) {
		t.Errorf("the enhanced request's answer is kept under %v; want its enhancer's file among them", named)
	}
	pictures.served(capabilityruntime.FreshRun(t.Context()), idea)
	if chats != 2 {
		t.Errorf("a fresh run opened %d chats in all; want its own", chats)
	}
}
