package latentvideo

import (
	"encoding/json"
	"path/filepath"
	"strings"
	"testing"

	"overgo/internal/sampling"
)

// TestWanRequestFormAdmitsPromptOrTensors pins the two request forms: the
// prompt form leaves generation parameters to the profile and carries no
// plan of its own, the tensor form keeps its strict validation, and the
// profile fills exactly the zero-valued parameters.
func TestWanRequestFormAdmitsPromptOrTensors(t *testing.T) {
	policy := generationPolicy{Frames: 81, Width: 832, Height: 480, Steps: 50, Shift: 5, GuideScale: 6, Seed: 31}
	prompt := WanRequest{Prompt: "a fox"}
	if err := ValidateWanRequestForm(prompt); err != nil {
		t.Fatal(err)
	}
	filled := prompt.withGeneration(policy)
	if filled.Frames != 81 || filled.Width != 832 || filled.Height != 480 || filled.Steps != 50 ||
		filled.Shift != 5 || filled.GuideScale != 6 || filled.Seed != 31 ||
		filled.NegativePrompt != ReferenceNegativePrompt || !filled.PromptForm() {
		t.Fatalf("profile defaults = %+v", filled)
	}
	partial := WanRequest{Prompt: "a fox", NegativePrompt: "blurry", Frames: 5, Steps: 2, Seed: 7}.withGeneration(policy)
	if partial.Frames != 5 || partial.Steps != 2 || partial.Width != 832 || partial.Height != 480 ||
		partial.NegativePrompt != "blurry" || partial.Seed != 7 {
		t.Fatalf("named parameters survive the defaults: %+v", partial)
	}
	for name, request := range map[string]WanRequest{
		"empty":                  {},
		"negative frames":        {Prompt: "a fox", Frames: -1},
		"negative guidance":      {Prompt: "a fox", GuideScale: -1},
		"noise plan with prompt": {Prompt: "a fox", Noise: sampling.CounterNoisePlan{Seed: 1, Grid: 1, Block: 256, Unroll: 4}},
		"sample with prompt":     {Prompt: "a fox", InitialSample: []float32{0}},
		"prompt beside contexts": {Prompt: "a fox", CondContext: []float32{1}},
	} {
		if ValidateWanRequestForm(request) == nil {
			t.Errorf("%s: admitted %+v", name, request)
		}
	}
	tensor := WanRequest{
		Frames: 1, Width: 32, Height: 16, Steps: 1, Shift: 5, GuideScale: 6,
		CondContext: []float32{1}, UncondContext: []float32{1},
	}
	if err := ValidateWanRequestForm(tensor); err != nil {
		t.Fatal(err)
	}
	if kept := tensor.withGeneration(policy); kept.Frames != 1 || kept.Steps != 1 || kept.NegativePrompt != "" || kept.PromptForm() {
		t.Fatalf("tensor form altered by the defaults: %+v", kept)
	}
}

// TestWanPromptFormMarshalsWithoutTensors pins the page's wire form: a
// prompt request carries no context, sample or noise keys, so the run
// record stays the typed request the page composed.
func TestWanPromptFormMarshalsWithoutTensors(t *testing.T) {
	encoded, err := json.Marshal(WanRequest{Prompt: "a fox", Seed: 31, Steps: 2})
	if err != nil {
		t.Fatal(err)
	}
	for _, absent := range []string{"cond_context", "uncond_context", "initial_sample", "noise"} {
		if strings.Contains(string(encoded), absent) {
			t.Errorf("prompt form carries %s: %s", absent, encoded)
		}
	}
	var decoded WanRequest
	if err := json.Unmarshal(encoded, &decoded); err != nil {
		t.Fatal(err)
	}
	if decoded.Prompt != "a fox" || decoded.Seed != 31 || decoded.Steps != 2 || !decoded.PromptForm() {
		t.Fatalf("round trip = %+v", decoded)
	}
}

// TestWanTextConditioningSpecLocatesThePipeline pins the reference layout
// of a Wan model directory the text pipeline reads.
func TestWanTextConditioningSpecLocatesThePipeline(t *testing.T) {
	directory := filepath.Join("models", "Wan2.1-T2V-1.3B")
	spec := WanTextConditioningSpec(directory, 512)
	want := TextConditioningSpec{
		TokenizerDir:        filepath.Join(directory, "google", "umt5-xxl"),
		EncoderCheckpoint:   filepath.Join(directory, "models_t5_umt5-xxl-enc-bf16.pth"),
		ProjectionDir:       directory,
		SequenceLength:      512,
		RelativeMaxDistance: ReferenceEncoderConfig.RelativeMaxDistance,
		NormEps:             ReferenceEncoderConfig.NormEps,
	}
	if spec != want {
		t.Fatalf("spec = %+v want %+v", spec, want)
	}
}
