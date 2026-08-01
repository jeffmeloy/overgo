package main

import (
	"context"
	"fmt"
	"image"
	_ "image/gif"
	_ "image/jpeg"
	_ "image/png"
	"os"

	"llamacpp2go/internal/inference"
	"llamacpp2go/internal/projector"
	"llamacpp2go/internal/tokenizer"
)

func qwen3VLProjectedPrompt(
	ctx context.Context,
	runner *inference.Runner,
	projectorPath, imagePath, question string,
	thinking bool,
) ([]tokenizer.TokenID, inference.ProjectedInputs, error) {
	vision, err := projector.OpenQwen3VL(projectorPath)
	if err != nil {
		return nil, inference.ProjectedInputs{}, fmt.Errorf("generate: open multimodal projector: %w", err)
	}
	defer vision.Close()
	file, err := os.Open(imagePath)
	if err != nil {
		return nil, inference.ProjectedInputs{}, fmt.Errorf("generate: open image: %w", err)
	}
	input, _, err := image.Decode(file)
	_ = file.Close()
	if err != nil {
		return nil, inference.ProjectedInputs{}, fmt.Errorf("generate: decode image: %w", err)
	}
	prompt, err := vision.BuildQwen35ImagePrompt(ctx, runner, input, "", question, thinking)
	if err != nil {
		return nil, inference.ProjectedInputs{}, fmt.Errorf("generate: encode image: %w", err)
	}
	if prompt.EmbeddingWidth != int(runner.Spec().EmbeddingLength) {
		return nil, inference.ProjectedInputs{}, fmt.Errorf(
			"generate: projector width %d differs from model width %d",
			prompt.EmbeddingWidth, runner.Spec().EmbeddingLength,
		)
	}
	imageTokens := len(prompt.Embeddings) / prompt.EmbeddingWidth
	overrides := make([]inference.EmbeddingOverride, imageTokens)
	for index := range overrides {
		start := index * prompt.EmbeddingWidth
		overrides[index] = inference.EmbeddingOverride{
			TokenIndex: uint32(prompt.ImageStart + index),
			Embedding:  prompt.Embeddings[start : start+prompt.EmbeddingWidth],
		}
	}
	positions := inference.MultiAxisPositions(prompt.MultiAxisPositions)
	return prompt.TokenIDs, inference.ProjectedInputs{
		EmbeddingOverrides: overrides,
		MultiAxisPositions: &positions,
	}, nil
}
