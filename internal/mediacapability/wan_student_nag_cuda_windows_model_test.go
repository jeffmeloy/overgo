//go:build modeltest

package mediacapability

import (
	"bytes"
	"encoding/json"
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
	"overgo/internal/sampling"
	"overgo/internal/testcheck"
)

// TestATypedNegativePromptGuidesTheStudentByNAG serves the Wan student requests of one prompt and seed through the
// video capability (guidance-ops/nag-wan, common-controls/negative). A negative prompt a person gives turns on NAG in
// each block's cross-attention at the paper's Table 5 values, with no preset to pick: it draws another clip than the
// plain request, at the student's own steps and guidance, and another negative prompt draws another clip again, so the
// negative prompt is what steers it. The negative prompt the profile fills where none is given guides nothing.
func TestATypedNegativePromptGuidesTheStudentByNAG(t *testing.T) {
	cudatest.Require(t)
	directory := modeltest.Directory(t, "Self-Forcing")
	capability := videoCapability()
	source := testcheck.Must(capability.Resolve(t.Context(), directory))(t)
	modelID := source.Inventory.Manifest.ID
	definition, _, err := source.Define(modelID)
	testcheck.NoError(t, err)
	profileContent := testcheck.Must(testcheck.Must(latentvideo.ResolveProfile(directory))(t).Content())(t)
	program := testcheck.Must(modelrecipe.CompileCapability(definition))(t)
	store := overgodbtest.Open(t, t.TempDir())
	publishArtifactExtent(t, store, modelID, directory)
	testcheck.Must(store.Commit(t.Context(), artifact.Batch{Key: "test/wan-student-nag", Contents: []artifact.Content{profileContent}}))(t)
	execution := modelrecipetest.CandidateExecution(t, store, program)
	draw := func(name string, request latentvideo.WanRequest) latentvideo.EncodedVideo {
		t.Helper()
		request.Prompt, request.Frames, request.Width, request.Height, request.Seed = "a red fox runs through fresh snow in a pine forest", 21, 64, 32, 7
		raw := testcheck.Must(json.Marshal(request))(t)
		output, err := capability.Execute(capabilityruntime.FreshRun(t.Context()), store, directory, execution, string(raw))
		if err != nil {
			t.Fatalf("the %s request was refused: %v", name, err)
		}
		video, ok := capabilityruntime.Unwrap(output).(latentvideo.EncodedVideo)
		if !ok {
			t.Fatalf("the %s request gave a %T", name, output)
		}
		return video
	}
	plain := draw("plain", latentvideo.WanRequest{})
	guided := draw("negative prompt", latentvideo.WanRequest{NegativePrompt: "blurry, static, low resolution"})
	if bytes.Equal(plain.Data, guided.Data) {
		t.Fatal("a typed negative prompt drew the plain clip")
	}
	want := media.Preset{Width: 64, Height: 32, Frames: 21, Steps: 4, Guidance: 1, Sampler: "chunk-causal flow", Shift: 5}
	if guided.Preset != want {
		t.Fatalf("the negative prompt's request ran %+v, want the student's own %+v", guided.Preset, want)
	}
	other := draw("another negative prompt", latentvideo.WanRequest{NegativePrompt: "a dog, a city street, summer"})
	if bytes.Equal(guided.Data, other.Data) {
		t.Fatal("two negative prompts drew one clip")
	}
	// Of guidance the student reads NAG alone, and no preset, at the steps its distillation fixed.
	for name, request := range map[string]latentvideo.WanRequest{
		"CFG-Zero*":        {Guidance: sampling.Guidance{ZeroSteps: 1, Projected: true}},
		"Wan's model-card": {Preset: "model-card"},
		"fifty steps":      {Steps: 50},
	} {
		request.Prompt, request.Frames, request.Width, request.Height, request.Seed = "a red fox runs through fresh snow in a pine forest", 21, 64, 32, 7
		raw := testcheck.Must(json.Marshal(request))(t)
		if _, err := capability.Execute(capabilityruntime.FreshRun(t.Context()), store, directory, execution, string(raw)); err == nil {
			t.Errorf("the student drew a request asking %s", name)
		}
	}
}
