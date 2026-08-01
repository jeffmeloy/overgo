package main

import (
	"context"
	"encoding/json"
	"image"
	"image/color"
	_ "image/png"
	"os"
	"slices"
	"testing"

	"llamacpp2go/internal/inference"
	"llamacpp2go/internal/projector"
	"llamacpp2go/internal/sampling"
	"llamacpp2go/internal/tokenizer"
)

func TestQwen35VideoEndToEndOracle(t *testing.T) {
	modelPath := os.Getenv("LLAMACPP2GO_QWEN35_MODEL")
	projectorPath := os.Getenv("LLAMACPP2GO_QWEN35_MMPROJ")
	goldenPath := os.Getenv("LLAMACPP2GO_QWEN35_VIDEO_GOLDEN")
	if modelPath == "" || projectorPath == "" || goldenPath == "" {
		t.Skip("set LLAMACPP2GO_QWEN35_MODEL, LLAMACPP2GO_QWEN35_MMPROJ, and LLAMACPP2GO_QWEN35_VIDEO_GOLDEN")
	}
	var golden struct {
		InputIDs          []tokenizer.TokenID `json:"input_ids"`
		Question          string              `json:"question"`
		GeneratedTokenIDs []tokenizer.TokenID `json:"generated_token_ids"`
	}
	data, err := os.ReadFile(goldenPath)
	if err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(data, &golden); err != nil {
		t.Fatal(err)
	}
	runner, err := inference.OpenWithOptions(modelPath, inference.OpenOptions{PreloadDeviceWeights: true})
	if err != nil {
		t.Fatal(err)
	}
	defer runner.Close()
	vision, err := projector.OpenQwen3VL(projectorPath)
	if err != nil {
		t.Fatal(err)
	}
	defer vision.Close()
	frames := make([]image.Image, 16)
	for temporal := range frames {
		frame := image.NewRGBA(image.Rect(0, 0, 224, 224))
		for y := 0; y < 224; y++ {
			for x := 0; x < 224; x++ {
				frame.SetRGBA(x, y, color.RGBA{
					R: uint8((x*4 + temporal*8) % 256),
					G: uint8((y*5 + temporal*4) % 256),
					B: uint8(((x+y)*3 + temporal*16) % 256), A: 255,
				})
			}
		}
		frames[temporal] = frame
	}
	prompt, err := vision.BuildQwen35VideoPrompt(
		context.Background(), runner, frames, "", golden.Question, 24, true,
	)
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(prompt.TokenIDs, golden.InputIDs) {
		t.Fatalf("video prompt IDs differ: got %d, want %d", len(prompt.TokenIDs), len(golden.InputIDs))
	}
	ids, projected, err := projectedInputsForPrompt(runner, prompt)
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

func TestGemma4ImageEndToEndOracle(t *testing.T) {
	modelPath := os.Getenv("LLAMACPP2GO_GEMMA4_MODEL")
	projectorPath := os.Getenv("LLAMACPP2GO_GEMMA4_MMPROJ")
	imagePath := os.Getenv("LLAMACPP2GO_GEMMA4_IMAGE")
	goldenPath := os.Getenv("LLAMACPP2GO_GEMMA4_GOLDEN")
	if modelPath == "" || projectorPath == "" || imagePath == "" || goldenPath == "" {
		t.Skip("set LLAMACPP2GO_GEMMA4_MODEL, LLAMACPP2GO_GEMMA4_MMPROJ, LLAMACPP2GO_GEMMA4_IMAGE, and LLAMACPP2GO_GEMMA4_GOLDEN")
	}
	var golden struct {
		InputIDs          []tokenizer.TokenID `json:"input_ids"`
		GeneratedTokenIDs []tokenizer.TokenID `json:"generated_token_ids"`
	}
	data, err := os.ReadFile(goldenPath)
	if err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(data, &golden); err != nil {
		t.Fatal(err)
	}
	runner, err := inference.OpenWithOptions(modelPath, inference.OpenOptions{PreloadQuantizedWeights: true})
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
	ids, projected, err := projectedInputsForPrompt(runner, prompt)
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

func TestGemma4AudioEndToEndOracle(t *testing.T) {
	modelPath := os.Getenv("LLAMACPP2GO_GEMMA4_MODEL")
	projectorPath := os.Getenv("LLAMACPP2GO_GEMMA4_MMPROJ")
	wavePath := os.Getenv("LLAMACPP2GO_GEMMA4_AUDIO_WAVE")
	goldenPath := os.Getenv("LLAMACPP2GO_GEMMA4_AUDIO_GOLDEN")
	if modelPath == "" || projectorPath == "" || wavePath == "" || goldenPath == "" {
		t.Skip("set LLAMACPP2GO_GEMMA4_MODEL, LLAMACPP2GO_GEMMA4_MMPROJ, LLAMACPP2GO_GEMMA4_AUDIO_WAVE, and LLAMACPP2GO_GEMMA4_AUDIO_GOLDEN")
	}
	var golden struct {
		InputIDs          []tokenizer.TokenID `json:"input_ids"`
		GeneratedTokenIDs []tokenizer.TokenID `json:"generated_token_ids"`
		Steps             []struct {
			TopIDs []tokenizer.TokenID `json:"top_ids"`
		} `json:"steps"`
	}
	data, err := os.ReadFile(goldenPath)
	if err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(data, &golden); err != nil {
		t.Fatal(err)
	}
	wave, err := os.ReadFile(wavePath)
	if err != nil {
		t.Fatal(err)
	}
	samples, err := projector.DecodeFloat32LE(wave)
	if err != nil {
		t.Fatal(err)
	}
	runner, err := inference.OpenWithOptions(modelPath, inference.OpenOptions{PreloadQuantizedWeights: true})
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
	ids, projected, err := projectedInputsForPrompt(runner, prompt)
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
	if os.Getenv("LLAMACPP2GO_GEMMA4_AUDIO_EXACT") != "" {
		if got != golden.GeneratedTokenIDs[0] {
			t.Fatalf("first generated token = %d, want %d", got, golden.GeneratedTokenIDs[0])
		}
		return
	}
	if !slices.Contains(golden.Steps[0].TopIDs, got) {
		t.Fatalf("first generated token = %d, outside reference top IDs %v", got, golden.Steps[0].TopIDs)
	}
}
