//go:build modeltest && windows

package latentimage

import (
	"bytes"
	"testing"

	cudatest "overgo/internal/cuda/testutil"
	"overgo/internal/testcheck"
)

// TestNAGAppliesANegativePromptAtGuidanceOne holds Krea's nag preset on the card (guidance-ops/nag-krea): Krea-2-Turbo
// runs without guidance, so it honoured no negative prompt; with the nag preset a negative prompt draws another picture,
// the nag preset without one draws the authors' picture byte for byte, and a session loaded with the negative prompt
// draws the picture one given it by Reset draws.
func TestNAGAppliesANegativePromptAtGuidanceOne(t *testing.T) {
	cudatest.Require(t)
	ctx := t.Context()
	authors := Request{Prompt: "a red fox licking a vanilla ice cream cone in snow", Width: 256, Height: 256, Steps: 2, Seed: 42}
	nag := authors
	nag.Preset = "nag"
	negative := nag
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
	unguided, guided := draw(nag), draw(negative)
	closeReporting(t, ctx, generator)
	if !bytes.Equal(unguided.Data, reference.Data) {
		t.Error("the nag preset with no negative prompt drew another picture than the authors' recipe")
	}
	if bytes.Equal(guided.Data, reference.Data) {
		t.Error("the nag preset's negative prompt left the picture as the authors' recipe drew it")
	}
	loaded := testcheck.Must(LoadGenerator(ctx, directory, profile, negative))(t)
	defer closeReporting(t, ctx, loaded)
	if fresh, _ := runResidentGeneration(t, ctx, loaded, negative); !bytes.Equal(fresh.Data, guided.Data) {
		t.Error("a session loaded with the negative prompt drew another picture than one given it by Reset")
	}
}
