//go:build modeltest

package mediacapability

import (
	"bytes"
	"encoding/json"
	"image/gif"
	"slices"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/capabilityruntime"
	cudatest "overgo/internal/cuda/testutil"
	"overgo/internal/latentvideo"
	"overgo/internal/media"
	"overgo/internal/modelrecipe"
	"overgo/internal/modelrecipetest"
	"overgo/internal/modeltest"
	"overgo/internal/overgodbtest"
	"overgo/internal/recipe"
	"overgo/internal/testcheck"
)

// TestTheWanStudentIsServedAsItsOwnModel holds the Self-Forcing student to being served as its own text-to-video
// model: wan-few-step/causal-t2v ran it from a test alone. Its directory resolves to the generation definition Wan has,
// prompt-conditioned with no video to edit, under its own profile: 81 frames at 832 x 480 in its four steps at shift 5,
// unguided. A prompt-form request through the video capability gives a clip of the frames and size it asks with that
// preset, and a second request with another prompt is served by the session the first loaded.
func TestTheWanStudentIsServedAsItsOwnModel(t *testing.T) {
	t.Parallel()
	cudatest.Require(t)
	directory := modeltest.Directory(t, "Self-Forcing")
	capability := videoCapability()
	source := testcheck.Must(capability.Resolve(t.Context(), directory))(t)
	modelID := source.Inventory.Manifest.ID
	define := func() (recipe.Definition, error) {
		definition, _, err := source.Define(modelID)
		return definition, err
	}
	definition := testcheck.Must(define())(t)
	inputs := make([]recipe.DataKind, len(definition.Inputs))
	for index, input := range definition.Inputs {
		inputs[index] = input.Data
	}
	if !slices.Contains(inputs, recipe.DataPromptConditioning) || slices.Contains(inputs, recipe.DataVideo) {
		t.Fatalf("the student's definition reads %v, want prompt conditioning and no video to edit", inputs)
	}
	profile := testcheck.Must(latentvideo.ResolveProfile(directory))(t)
	want := media.Preset{Width: 832, Height: 480, Frames: 81, Steps: 4, Guidance: 1, Sampler: "chunk-causal flow", Shift: 5}
	bounds := testcheck.Must(latentvideo.ControlBounds(directory))(t)
	if sized := (media.Preset{Width: bounds["width"].Default, Height: bounds["height"].Default, Frames: bounds["frames"].Default,
		Steps: bounds["steps"].Default, Guidance: want.Guidance, Sampler: want.Sampler, Shift: want.Shift}); sized != want || bounds["frames"].Step != 12 {
		t.Fatalf("the student's profile asks %+v with frames moving by %d, want %+v by 12", sized, bounds["frames"].Step, want)
	}
	profileContent := testcheck.Must(profile.Content())(t)
	program := testcheck.Must(modelrecipe.CompileCapability(definition))(t)
	store := overgodbtest.Open(t, t.TempDir())
	publishArtifactExtent(t, store, modelID, directory)
	testcheck.Must(store.Commit(t.Context(), artifact.Batch{Key: "test/wan-student", Contents: []artifact.Content{profileContent}}))(t)
	execution := modelrecipetest.CandidateExecution(t, store, program)
	for run, prompt := range []string{"a red fox runs through fresh snow in a pine forest", "a paper boat drifts down a rainy street gutter"} {
		raw := testcheck.Must(json.Marshal(latentvideo.WanRequest{Prompt: prompt, Frames: 21, Width: 64, Height: 32, Seed: 7}))(t)
		output := testcheck.Must(capability.Execute(capabilityruntime.FreshRun(t.Context()), store, directory, execution, string(raw)))(t)
		video, ok := capabilityruntime.Unwrap(output).(latentvideo.EncodedVideo)
		if !ok {
			t.Fatalf("video output type=%T", output)
		}
		animation := testcheck.Must(gif.DecodeAll(bytes.NewReader(video.Data)))(t)
		asked := media.Preset{Width: 64, Height: 32, Frames: 21, Steps: want.Steps, Guidance: want.Guidance, Sampler: want.Sampler, Shift: want.Shift}
		if len(animation.Image) != 21 || video.Width != 64 || video.Height != 32 || video.Preset != asked || video.ResidentRun != run+1 {
			t.Fatalf("request %d gave %d frames at %d x %d, preset %+v, resident run %d; want 21 at 64 x 32, %+v, run %d",
				run+1, len(animation.Image), video.Width, video.Height, video.Preset, video.ResidentRun, asked, run+1)
		}
	}
}
