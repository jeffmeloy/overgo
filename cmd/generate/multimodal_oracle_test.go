package main

import (
	"context"
	"encoding/json"
	"image"
	"image/color"
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
