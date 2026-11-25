//go:build windows

package mediacapability

import (
	"context"
	"slices"
	"strings"
	"testing"
	"time"

	"overgo/internal/artifact"
	"overgo/internal/capabilityruntime"
	"overgo/internal/modelrecipe"
	"overgo/internal/overgodbtest"
	"overgo/internal/processmeasure"
	"overgo/internal/recipe"
	"overgo/internal/speechsynth"
	"overgo/internal/testcheck"
	"overgo/internal/testutil"
	"overgo/internal/workflowruntime"
)

// standInVoice is a speech model that loads from nothing and speaks through stand-in stages.
type standInVoice struct{}

// heldParts serves one request under a ledger and holds its parts: each named part carries its wall and its processor
// time and bytes read, the named parts' walls sum within the request's, and every name want lists is among them.
func heldParts(t *testing.T, label string, serve func(context.Context), want ...string) {
	t.Helper()
	ctx, stages := processmeasure.WithStages(t.Context())
	clock := processmeasure.NewStopwatch()
	serve(ctx)
	wall := time.Duration(testcheck.Must(clock.Elapsed())(t))
	parts := testcheck.Must(stages.Parts())(t)
	var named []string
	sum := time.Duration(0)
	for _, part := range parts {
		named, sum = append(named, part.Name), sum+part.Wall
		if part.Use == nil || part.Times == 0 {
			t.Errorf("%s: the part %q carries %+v; want its wall, processor time and bytes read", label, part.Name, part)
		}
	}
	if sum > wall {
		t.Errorf("%s: the parts' walls sum to %v, past the request's %v", label, sum, wall)
	}
	for _, name := range want {
		if !slices.Contains(named, name) {
			t.Errorf("%s: the request's parts are %q; want %q among them", label, named, name)
		}
	}
	t.Logf("%s: %d parts within %v: %s", label, len(parts), wall, strings.Join(named, ", "))
}

// TestEachServedPathSplitsItsRequestIntoParts holds a served request to a ledger of its parts (served-parts/do), each
// with its wall, processor time and bytes read, as the MOSS workflow's clock reads a section's. A served request is
// composed one of two ways: through a session director, as a picture and a Wan clip are, or through JSONScalar, which
// loads a model for the request, as speech, transcription and an oscillator clip are.
//   - A MOSS picture is served through the catalog's own picture executor (pictureExecutor) over a stand-in session.
//     Its parts name its lease, its stages by their modules, and its run recorded.
//   - A speech request is served through the catalog's own speech routing (speechExecutor) over JSONScalar with a
//     stand-in model. Its parts name the request read, the stored answer sought, the model loaded and closed, and its
//     stages by their modules.
//
// Only the director had named its parts, and none carried processor time or bytes read: the MOSS clock read those by
// hand. A chat turn's parts are held by cmd/recipe's measurement (TestInferenceMeasurementFillsTheServingCells), since
// a chat turn is served by the server, not the media catalog.
func TestEachServedPathSplitsItsRequestIntoParts(t *testing.T) {
	// Serial: the stand-in pictures hold the image device's session capacity, as the other served-picture tests do.
	t.Run("a picture", func(t *testing.T) {
		pictures := newStandInPictures(t, enhancerOpener)
		raw := `{"prompt":"a lighthouse at dusk","preset":"ultra-image-to-image-turbo","seed":3,"image":"` + pictures.sourceA.String() + `"}`
		heldParts(t, "a picture", func(ctx context.Context) { pictures.served(ctx, raw) },
			"model session leased", string(modelrecipe.ModuleLatentImagePrepare), string(modelrecipe.ModuleLatentImageIntegrate),
			string(modelrecipe.ModuleLatentImageDecode), "run recorded")
	})
	t.Run("speech", func(t *testing.T) {
		store := overgodbtest.Open(t, t.TempDir())
		model := testutil.ArtifactID(t, artifact.KindModel, "speech-parts-fixture")
		testutil.PublishArtifact(t, store, model)
		program := testcheck.Must(modelrecipe.CompileCapability(testcheck.Must(modelrecipe.CapabilityDefinition(recipe.TaskSpeech, model))(t)))(t)
		recording := speechsynth.Audio{PCM: []float32{0, 0.25, -0.25, 0.5}, SampleRate: 24000, Channels: 1, Complete: true}
		load := func(context.Context, artifact.Repository, string, recipe.Program, speechsynth.SynthesisRequest) (*standInVoice, error) {
			return &standInVoice{}, nil
		}
		bind := func(runtime *workflowruntime.Runtime, modelID artifact.ID, _ *standInVoice) error {
			return workflowruntime.RegisterContextPipeline(runtime, modelID,
				modelrecipe.ModuleSpeechTokenize, func(context.Context, speechsynth.SynthesisRequest) (string, error) { return "tokens", nil },
				modelrecipe.ModuleSpeechGenerate, func(context.Context, string) (int, error) { return 1, nil },
				modelrecipe.ModuleSpeechDecode, func(context.Context, int) (speechsynth.Audio, error) { return recording, nil },
				func(audio speechsynth.Audio) (artifact.Content, error) { return OutputContent(audio) })
		}
		scalar := capabilityruntime.JSONScalar[speechsynth.SynthesisRequest, *standInVoice, speechsynth.Audio]("speech", speechsynth.ValidateSynthesisRequest, load, bind)
		serve := speechExecutor(scalar, scalar)
		heldParts(t, "speech", func(ctx context.Context) {
			testcheck.Must(serve(ctx, store, t.TempDir(), modelrecipe.CapabilityEvidenceSelection{Program: program}, `{"text":"The tide went out before dawn.","voice":"alba"}`))(t)
		}, "request read", "stored answer sought", "model loaded", string(modelrecipe.ModuleSpeechTokenize), string(modelrecipe.ModuleSpeechGenerate),
			string(modelrecipe.ModuleSpeechDecode), "model closed")
	})
}
