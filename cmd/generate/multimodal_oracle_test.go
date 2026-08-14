package main

import (
	"context"
	"image"
	_ "image/png"
	"os"
	"slices"
	"testing"

	cudatest "overgo/internal/cuda/testutil"
	"overgo/internal/inference"
	"overgo/internal/jsonfile"
	"overgo/internal/media"
	"overgo/internal/modelrecipe"
	"overgo/internal/projector"
	"overgo/internal/recipe"
	"overgo/internal/sampling"
	"overgo/internal/servingtest"
	"overgo/internal/tokenizer"
)

func openFixtureRunner(
	path string, options inference.OpenOptions, residency recipe.ResidencyPolicy,
) (*inference.Runner, error) {
	loaded, err := servingtest.ResolveActiveGGUFWithPolicy(
		path, recipe.PlacementHybrid, modelrecipe.DecodeSessionCapacity, residency,
	)
	if err != nil {
		return nil, err
	}
	return inference.OpenWithProgram(&loaded, options)
}

func TestGemma4ImageEndToEndOracle(t *testing.T) {
	requireIntegration(t)
	modelPath := os.Getenv("OVERGO_GEMMA4_MODEL")
	projectorPath := os.Getenv("OVERGO_GEMMA4_MMPROJ")
	imagePath := os.Getenv("OVERGO_GEMMA4_IMAGE")
	goldenPath := os.Getenv("OVERGO_GEMMA4_GOLDEN")
	if modelPath == "" || projectorPath == "" || imagePath == "" || goldenPath == "" {
		t.Skip("set OVERGO_GEMMA4_MODEL, OVERGO_GEMMA4_MMPROJ, OVERGO_GEMMA4_IMAGE, and OVERGO_GEMMA4_GOLDEN")
	}
	var golden struct {
		InputIDs          []tokenizer.TokenID `json:"input_ids"`
		GeneratedTokenIDs []tokenizer.TokenID `json:"generated_token_ids"`
	}
	if err := jsonfile.Decode(goldenPath, &golden); err != nil {
		t.Fatal(err)
	}
	runner, err := openFixtureRunner(modelPath, inference.OpenOptions{}, recipe.ResidencyDeviceNative)
	if err != nil {
		t.Fatal(err)
	}
	defer runner.Close()
	vision, err := projector.OpenImageProjector(projectorPath)
	if err != nil {
		t.Fatal(err)
	}
	defer vision.Close()
	file, err := os.Open(imagePath)
	if err != nil {
		t.Fatal(err)
	}
	input, _, err := image.Decode(file)
	_ = file.Close()
	if err != nil {
		t.Fatal(err)
	}
	prompt, err := vision.BuildImagePrompt(
		context.Background(), runner, input, "", "What color dominates this image? One word.", false,
	)
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(prompt.TokenIDs, golden.InputIDs) {
		t.Fatalf("image prompt IDs differ: got %d, want %d", len(prompt.TokenIDs), len(golden.InputIDs))
	}
	ids, projected, err := inference.ProjectedInputsForPrompt(runner, prompt)
	if err != nil {
		t.Fatal(err)
	}
	sampler, err := sampling.New(sampling.Config{Temperature: 0, TopK: 1, TopP: 1})
	if err != nil {
		t.Fatal(err)
	}
	generated, _, err := runner.Generate(context.Background(), "", inference.GenerateOptions{
		MaxNewTokens: 1, Sampler: sampler, PromptTokenIDs: ids, ProjectedInputs: &projected,
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(generated) != len(ids)+1 || len(golden.GeneratedTokenIDs) == 0 || generated[len(ids)] != golden.GeneratedTokenIDs[0] {
		t.Fatalf("first generated token = %v, want %v", generated[len(ids):], golden.GeneratedTokenIDs[:1])
	}
}

// TestGemma4ImageDeviceProjectorEndToEndOracle: same as the image oracle but the
// projector forward runs on the CUDA device, so the entire image->soft-token->
// language path executes on-device. Closes the single-process device-projector +
// device-language combination.
func TestGemma4ImageDeviceProjectorEndToEndOracle(t *testing.T) {
	requireIntegration(t)
	cudatest.Require(t)
	modelPath := os.Getenv("OVERGO_GEMMA4_MODEL")
	projectorPath := os.Getenv("OVERGO_GEMMA4_MMPROJ")
	imagePath := os.Getenv("OVERGO_GEMMA4_IMAGE")
	goldenPath := os.Getenv("OVERGO_GEMMA4_GOLDEN")
	if modelPath == "" || projectorPath == "" || imagePath == "" || goldenPath == "" {
		t.Skip("set OVERGO_GEMMA4_MODEL, OVERGO_GEMMA4_MMPROJ, OVERGO_GEMMA4_IMAGE, and OVERGO_GEMMA4_GOLDEN")
	}
	var golden struct {
		InputIDs          []tokenizer.TokenID `json:"input_ids"`
		GeneratedTokenIDs []tokenizer.TokenID `json:"generated_token_ids"`
	}
	if err := jsonfile.Decode(goldenPath, &golden); err != nil {
		t.Fatal(err)
	}
	runner, err := openFixtureRunner(modelPath, inference.OpenOptions{}, recipe.ResidencyDeviceNative)
	if err != nil {
		t.Fatal(err)
	}
	defer runner.Close()
	vision, err := projector.OpenImageProjectorWithOptions(projectorPath, projector.OpenOptions{CUDA: true})
	if err != nil {
		t.Fatal(err)
	}
	defer vision.Close()
	file, err := os.Open(imagePath)
	if err != nil {
		t.Fatal(err)
	}
	input, _, err := image.Decode(file)
	_ = file.Close()
	if err != nil {
		t.Fatal(err)
	}
	prompt, err := vision.BuildImagePrompt(
		context.Background(), runner, input, "", "What color dominates this image? One word.", false,
	)
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(prompt.TokenIDs, golden.InputIDs) {
		t.Fatalf("image prompt IDs differ: got %d, want %d", len(prompt.TokenIDs), len(golden.InputIDs))
	}
	ids, projected, err := inference.ProjectedInputsForPrompt(runner, prompt)
	if err != nil {
		t.Fatal(err)
	}
	sampler, err := sampling.New(sampling.Config{Temperature: 0, TopK: 1, TopP: 1})
	if err != nil {
		t.Fatal(err)
	}
	generated, _, err := runner.Generate(context.Background(), "", inference.GenerateOptions{
		MaxNewTokens: 1, Sampler: sampler, PromptTokenIDs: ids, ProjectedInputs: &projected,
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(generated) != len(ids)+1 || len(golden.GeneratedTokenIDs) == 0 || generated[len(ids)] != golden.GeneratedTokenIDs[0] {
		t.Fatalf("device-projector first generated token = %v, want %v", generated[len(ids):], golden.GeneratedTokenIDs[:1])
	}
}

func TestGemma4AudioEndToEndOracle(t *testing.T) {
	requireIntegration(t)
	modelPath := os.Getenv("OVERGO_GEMMA4_MODEL")
	projectorPath := os.Getenv("OVERGO_GEMMA4_MMPROJ")
	wavePath := os.Getenv("OVERGO_GEMMA4_AUDIO_WAVE")
	goldenPath := os.Getenv("OVERGO_GEMMA4_AUDIO_GOLDEN")
	if modelPath == "" || projectorPath == "" || wavePath == "" || goldenPath == "" {
		t.Skip("set OVERGO_GEMMA4_MODEL, OVERGO_GEMMA4_MMPROJ, OVERGO_GEMMA4_AUDIO_WAVE, and OVERGO_GEMMA4_AUDIO_GOLDEN")
	}
	var golden struct {
		InputIDs          []tokenizer.TokenID `json:"input_ids"`
		GeneratedTokenIDs []tokenizer.TokenID `json:"generated_token_ids"`
		Steps             []struct {
			TopIDs []tokenizer.TokenID `json:"top_ids"`
		} `json:"steps"`
	}
	if err := jsonfile.Decode(goldenPath, &golden); err != nil {
		t.Fatal(err)
	}
	wave, err := os.ReadFile(wavePath)
	if err != nil {
		t.Fatal(err)
	}
	samples, err := media.DecodeFloat32LE(wave)
	if err != nil {
		t.Fatal(err)
	}
	runner, err := openFixtureRunner(modelPath, inference.OpenOptions{}, recipe.ResidencyDeviceNative)
	if err != nil {
		t.Fatal(err)
	}
	defer runner.Close()
	audio, err := projector.OpenAudioProjector(projectorPath)
	if err != nil {
		t.Fatal(err)
	}
	defer audio.Close()
	prompt, err := audio.BuildAudioPrompt(
		context.Background(), runner, samples, "", "What note do you hear? One word.",
	)
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(prompt.TokenIDs, golden.InputIDs) {
		t.Fatalf("audio prompt IDs differ: got %d, want %d", len(prompt.TokenIDs), len(golden.InputIDs))
	}
	ids, projected, err := inference.ProjectedInputsForPrompt(runner, prompt)
	if err != nil {
		t.Fatal(err)
	}
	sampler, err := sampling.New(sampling.Config{Temperature: 0, TopK: 1, TopP: 1})
	if err != nil {
		t.Fatal(err)
	}
	generated, _, err := runner.Generate(context.Background(), "", inference.GenerateOptions{
		MaxNewTokens: 1, Sampler: sampler, PromptTokenIDs: ids, ProjectedInputs: &projected,
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(generated) != len(ids)+1 || len(golden.GeneratedTokenIDs) == 0 || len(golden.Steps) == 0 {
		t.Fatalf("generated IDs = %v; golden is incomplete", generated)
	}
	got := generated[len(ids)]
	if os.Getenv("OVERGO_GEMMA4_AUDIO_EXACT") != "" {
		if got != golden.GeneratedTokenIDs[0] {
			t.Fatalf("first generated token = %d, want %d", got, golden.GeneratedTokenIDs[0])
		}
		return
	}
	if !slices.Contains(golden.Steps[0].TopIDs, got) {
		t.Fatalf("first generated token = %d, outside reference top IDs %v", got, golden.Steps[0].TopIDs)
	}
}
