//go:build windows

package mediacapability

import (
	"context"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/latentimage"
	"overgo/internal/strictjson"
	"overgo/internal/testcheck"
)

// workflowBackupNegative is the MOSS Ultra workflow's backup negative, which a section given no negative is drawn from
// (its "Negative prompt used" in every recorded section).
const workflowBackupNegative = "unwanted rendering artifacts, unintended distortion, accidental duplicate subjects, inconsistent perspective, unrequested text overlays, unwanted watermarks"

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

// TestAServedPictureReportsThePromptsItDrewFrom holds moss-prompts-used/report: each of the MOSS Ultra workflow's
// sections shows the prompt and the negative prompt its picture was drawn from ("Prompt used", "Negative prompt used",
// its PreviewAny nodes), and a picture served through the picture capability (pictureExecutor, standInPictures) reports
// them: in prompt mode 1 the prompt and negative given, or the workflow's backup negative where none is given; in mode 2
// what its enhancer wrote, which the request it runs records, not the idea it was given; in mode 3 the layout prompt
// its builder's fields make on its canvas, where its request gives no prompt. A picture its stored run answers reports
// the same as the one drawn. A picture had reported its data, size and preset alone, and mode 3's prompt was recorded
// nowhere.
func TestAServedPictureReportsThePromptsItDrewFrom(t *testing.T) {
	pictures := newStandInPictures(t, standInEnhancer)
	reported := func(raw string) latentimage.EncodedImage {
		t.Helper()
		drawn, picture := pictures.served(t.Context(), raw)
		replayed, again := pictures.served(t.Context(), raw)
		if replayed.Run != drawn.Run || again.Prompt != picture.Prompt || again.NegativePrompt != picture.NegativePrompt {
			t.Errorf("%s's stored run reported %q and %q where its draw reported %q and %q", raw, again.Prompt, again.NegativePrompt, picture.Prompt, picture.NegativePrompt)
		}
		return picture
	}
	const sized = `"preset":"ultra-text-to-image","seed":7,"width":256,"height":144`

	if given := reported(`{"prompt":"a red fox","negative_prompt":"grey sky",` + sized + `}`); given.Prompt != "a red fox" || given.NegativePrompt != "grey sky" {
		t.Errorf("prompt mode 1 reported %q and %q; want the prompt and negative given", given.Prompt, given.NegativePrompt)
	}
	if backed := reported(`{"prompt":"a red fox",` + sized + `}`); backed.Prompt != "a red fox" || backed.NegativePrompt != workflowBackupNegative {
		t.Errorf("prompt mode 1 with no negative reported %q and %q; want the prompt and the workflow's backup negative", backed.Prompt, backed.NegativePrompt)
	}

	idea := `{"prompt":"a fox","prompt_mode":"enhanced",` + sized + `}`
	enhanced, picture := pictures.served(t.Context(), idea)
	var request latentimage.Request
	testcheck.NoError(t, strictjson.DecodeBytes(enhanced.Input.Data, &request))
	if picture.Prompt != request.Prompt || picture.NegativePrompt != request.NegativePrompt || picture.Prompt == "a fox" || !strings.Contains(picture.Prompt, "birch forest") ||
		!strings.Contains(picture.NegativePrompt, "blurry edges") {
		t.Errorf("prompt mode 2 reported %q and %q, its run recording %q and %q; want what its enhancer wrote", picture.Prompt, picture.NegativePrompt, request.Prompt, request.NegativePrompt)
	}

	laid := reported(`{"layout":{"high_level_description":"a fox at dawn","regions":[{"x":0.1,"y":0.2,"w":0.4,"h":0.5,"desc":"a fox"}]},` + sized + `}`)
	if !strings.HasPrefix(laid.Prompt, `{"summary":{"caption":`) || !strings.Contains(laid.Prompt, `"caption":"a fox"`) || !strings.Contains(laid.Prompt, "a fox at dawn") ||
		laid.NegativePrompt != workflowBackupNegative {
		t.Errorf("prompt mode 3 reported %q and %q; want the layout prompt its builder's fields make and the backup negative", laid.Prompt, laid.NegativePrompt)
	}
}
