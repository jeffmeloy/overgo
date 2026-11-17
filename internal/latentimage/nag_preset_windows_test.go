//go:build modeltest && windows

package latentimage

import (
	"bytes"
	"testing"

	cudatest "overgo/internal/cuda/testutil"
	"overgo/internal/testcheck"
)

// TestATypedNegativePromptGuidesKreaByNAG holds Krea's negative prompt on the card (guidance-ops/nag-krea,
// common-controls/negative): Krea-2-Turbo runs without guidance, so it honoured no negative prompt; a negative prompt a
// person gives now turns on its NAG, with no preset to pick, and draws another picture; the same request given no
// negative prompt again draws the authors' picture byte for byte; and a session loaded with the negative prompt draws
// the picture one given it by Reset draws.
func TestATypedNegativePromptGuidesKreaByNAG(t *testing.T) {
	cudatest.Require(t)
	ctx := t.Context()
	authors := Request{Prompt: "a red fox licking a vanilla ice cream cone in snow", Width: 256, Height: 256, Steps: 2, Seed: 42}
	negative := authors
	negative.NegativePrompt = "snow"
	directory, profile := kreaDirOrSkip(t), kreaProfileOrSkip(t)
	generator := testcheck.Must(LoadGenerator(ctx, directory, profile, authors))(t)
	reference, _ := runResidentGeneration(t, ctx, generator, authors)
	draw := func(request Request) EncodedImage {
		t.Helper()
		testcheck.NoError(t, generator.Reset(ctx, request))
		image, took := runResidentGeneration(t, ctx, generator, request)
		t.Logf("%q against %q drew in %s", request.Prompt, request.NegativePrompt, took)
		return image
	}
	guided, unguided := draw(negative), draw(authors)
	closeReporting(t, ctx, generator)
	if bytes.Equal(guided.Data, reference.Data) {
		t.Error("a typed negative prompt left the picture as the authors' recipe drew it")
	}
	if !bytes.Equal(unguided.Data, reference.Data) {
		t.Error("given no negative prompt again, the session drew another picture than the authors' recipe")
	}
	loaded := testcheck.Must(LoadGenerator(ctx, directory, profile, negative))(t)
	defer closeReporting(t, ctx, loaded)
	if fresh, _ := runResidentGeneration(t, ctx, loaded, negative); !bytes.Equal(fresh.Data, guided.Data) {
		t.Error("a session loaded with the negative prompt drew another picture than one given it by Reset")
	}
}
