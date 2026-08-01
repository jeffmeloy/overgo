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
	"strings"

	"llamacpp2go/internal/inference"
	"llamacpp2go/internal/projector"
	"llamacpp2go/internal/tokenizer"
)

const qwen3VLImagePad = "<|image_pad|>"

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
	output, err := vision.EncodeImage(ctx, input, projector.DefaultQwen3VLPreprocessOptions())
	if err != nil {
		return nil, inference.ProjectedInputs{}, fmt.Errorf("generate: encode image: %w", err)
	}
	if output.Embeddings.Shape.Dims[0] != uint64(runner.Spec().EmbeddingLength) {
		return nil, inference.ProjectedInputs{}, fmt.Errorf(
			"generate: projector width %d differs from model width %d",
			output.Embeddings.Shape.Dims[0], runner.Spec().EmbeddingLength,
		)
	}
	imageTokens := int(output.Embeddings.Shape.Dims[1])
	prompt := qwen3VLPromptText(question, imageTokens, thinking)
	ids, err := runner.TokenizeText(prompt, false, true)
	if err != nil {
		return nil, inference.ProjectedInputs{}, fmt.Errorf("generate: tokenize image prompt: %w", err)
	}
	padIDs, err := runner.TokenizeText(qwen3VLImagePad, false, true)
	if err != nil {
		return nil, inference.ProjectedInputs{}, fmt.Errorf("generate: tokenize image placeholder: %w", err)
	}
	if len(padIDs) != 1 {
		return nil, inference.ProjectedInputs{}, fmt.Errorf("generate: image placeholder maps to %d tokens", len(padIDs))
	}
	imageStart, err := contiguousTokenRun(ids, padIDs[0], imageTokens)
	if err != nil {
		return nil, inference.ProjectedInputs{}, fmt.Errorf("generate: image prompt: %w", err)
	}
	positions, err := qwen3VLMultiAxisPositions(
		len(ids), imageStart, imageTokens,
		output.GridH, output.GridW, output.MergeSize,
	)
	if err != nil {
		return nil, inference.ProjectedInputs{}, err
	}
	overrides := make([]inference.EmbeddingOverride, imageTokens)
	width := int(output.Embeddings.Shape.Dims[0])
	for index := range overrides {
		start := index * width
		overrides[index] = inference.EmbeddingOverride{
			TokenIndex: uint32(imageStart + index),
			Embedding:  output.Embeddings.Data[start : start+width],
		}
	}
	return ids, inference.ProjectedInputs{
		EmbeddingOverrides: overrides,
		MultiAxisPositions: &positions,
	}, nil
}

func qwen3VLPromptText(question string, imageTokens int, thinking bool) string {
	var prompt strings.Builder
	prompt.WriteString("<|im_start|>user\n<|vision_start|>")
	prompt.WriteString(strings.Repeat(qwen3VLImagePad, imageTokens))
	prompt.WriteString("<|vision_end|>")
	prompt.WriteString(question)
	prompt.WriteString("<|im_end|>\n<|im_start|>assistant\n<think>\n")
	if !thinking {
		prompt.WriteString("\n</think>\n\n")
	}
	return prompt.String()
}

func contiguousTokenRun(ids []tokenizer.TokenID, token tokenizer.TokenID, count int) (int, error) {
	if count <= 0 {
		return 0, errors.New("image token count is not positive")
	}
	found := -1
	for start := 0; start+count <= len(ids); start++ {
		if ids[start] != token {
			continue
		}
		end := start
		for end < len(ids) && ids[end] == token {
			end++
		}
		if end-start != count {
			return 0, fmt.Errorf("placeholder run has %d tokens, want %d", end-start, count)
		}
		if found >= 0 {
			return 0, errors.New("multiple image placeholder runs")
		}
		found = start
		start = end - 1
	}
	if found < 0 {
		return 0, errors.New("image placeholder run is absent")
	}
	return found, nil
}

func qwen3VLMultiAxisPositions(
	tokens, imageStart, imageTokens, gridH, gridW, merge int,
) (inference.MultiAxisPositions, error) {
	if tokens <= 0 || imageStart < 0 || imageTokens <= 0 || imageStart+imageTokens > tokens ||
		gridH <= 0 || gridW <= 0 || merge <= 0 || gridH%merge != 0 || gridW%merge != 0 {
		return inference.MultiAxisPositions{}, errors.New("generate: invalid image position geometry")
	}
	rows, columns := gridH/merge, gridW/merge
	if rows*columns != imageTokens {
		return inference.MultiAxisPositions{}, fmt.Errorf(
			"generate: image grid produces %d tokens, projector emitted %d", rows*columns, imageTokens,
		)
	}
	var positions inference.MultiAxisPositions
	for axis := range positions {
		positions[axis] = make([]uint32, tokens)
	}
	for index := 0; index < imageStart; index++ {
		for axis := range positions {
			positions[axis][index] = uint32(index)
		}
	}
	base := uint32(imageStart)
	for index := 0; index < imageTokens; index++ {
		positions[0][imageStart+index] = base
		positions[1][imageStart+index] = base + uint32(index/columns)
		positions[2][imageStart+index] = base + uint32(index%columns)
		positions[3][imageStart+index] = 0
	}
	next := base + uint32(max(rows, columns))
	for index := imageStart + imageTokens; index < tokens; index++ {
		for axis := range positions {
			positions[axis][index] = next
		}
		next++
	}
	return positions, nil
}
