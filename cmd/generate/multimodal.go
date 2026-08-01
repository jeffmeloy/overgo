package main

import (
	"context"
	"errors"
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

func imageProjectedPrompt(
	ctx context.Context,
	runner *inference.Runner,
	projectorPath, imagePath, question string,
	thinking bool,
) ([]tokenizer.TokenID, inference.ProjectedInputs, error) {
	vision, err := projector.OpenImageProjector(projectorPath)
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
	prompt, err := vision.BuildImagePrompt(ctx, runner, input, "", question, thinking)
	if err != nil {
		return nil, inference.ProjectedInputs{}, fmt.Errorf("generate: encode image: %w", err)
	}
	return projectedInputsForPrompt(runner, prompt)
}

func qwen3VLProjectedVideoPrompt(
	ctx context.Context,
	runner *inference.Runner,
	projectorPath string,
	framePaths []string,
	question string,
	fps float64,
	thinking bool,
) ([]tokenizer.TokenID, inference.ProjectedInputs, error) {
	vision, err := projector.OpenQwen3VL(projectorPath)
	if err != nil {
		return nil, inference.ProjectedInputs{}, fmt.Errorf("generate: open multimodal projector: %w", err)
	}
	defer vision.Close()
	frames := make([]image.Image, len(framePaths))
	for index, path := range framePaths {
		file, openErr := os.Open(path)
		if openErr != nil {
			return nil, inference.ProjectedInputs{}, fmt.Errorf("generate: open video frame %d: %w", index, openErr)
		}
		frames[index], _, openErr = image.Decode(file)
		_ = file.Close()
		if openErr != nil {
			return nil, inference.ProjectedInputs{}, fmt.Errorf("generate: decode video frame %d: %w", index, openErr)
		}
	}
	prompt, err := vision.BuildQwen35VideoPrompt(ctx, runner, frames, "", question, fps, thinking)
	if err != nil {
		return nil, inference.ProjectedInputs{}, fmt.Errorf("generate: encode video: %w", err)
	}
	return projectedInputsForPrompt(runner, prompt)
}

func projectedInputsForPrompt(
	runner *inference.Runner,
	prompt projector.MultimodalPrompt,
) ([]tokenizer.TokenID, inference.ProjectedInputs, error) {
	if prompt.EmbeddingWidth != int(runner.Spec().EmbeddingLength) {
		return nil, inference.ProjectedInputs{}, fmt.Errorf(
			"generate: projector width %d differs from model width %d",
			prompt.EmbeddingWidth, runner.Spec().EmbeddingLength,
		)
	}
	imageTokens := len(prompt.EmbeddingTokenIndices)
	if len(prompt.Embeddings) != imageTokens*prompt.EmbeddingWidth {
		return nil, inference.ProjectedInputs{}, errors.New("generate: projector returned invalid embedding indices")
	}
	overrides := make([]inference.EmbeddingOverride, imageTokens)
	for index := range overrides {
		start := index * prompt.EmbeddingWidth
		overrides[index] = inference.EmbeddingOverride{
			TokenIndex: prompt.EmbeddingTokenIndices[index],
			Embedding:  prompt.Embeddings[start : start+prompt.EmbeddingWidth],
		}
	}
	projected := inference.ProjectedInputs{EmbeddingOverrides: overrides}
	hasMultiAxis := false
	for _, axis := range prompt.MultiAxisPositions {
		hasMultiAxis = hasMultiAxis || len(axis) > 0
	}
	if hasMultiAxis {
		positions := inference.MultiAxisPositions(prompt.MultiAxisPositions)
		projected.MultiAxisPositions = &positions
	}
	return prompt.TokenIDs, projected, nil
}
