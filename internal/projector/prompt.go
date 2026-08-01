package projector

import (
	"context"
	"errors"
	"fmt"
	"image"
	"strings"

	"llamacpp2go/internal/tokenizer"
)

const Qwen3VLImagePad = "<|image_pad|>"

type Qwen3VLTokenizer interface {
	TokenizeText(string, bool, bool) ([]tokenizer.TokenID, error)
}

type Qwen3VLPrompt struct {
	TokenIDs           []tokenizer.TokenID
	Embeddings         []float32
	EmbeddingWidth     int
	ImageStart         int
	MultiAxisPositions [4][]uint32
}

func (r *Qwen3VLRunner) BuildQwen35ImagePrompt(
	ctx context.Context,
	tokenizer Qwen3VLTokenizer,
	source image.Image,
	beforeImage, afterImage string,
	thinking bool,
) (Qwen3VLPrompt, error) {
	if tokenizer == nil {
		return Qwen3VLPrompt{}, errors.New("projector: tokenizer is nil")
	}
	output, err := r.EncodeImage(ctx, source, DefaultQwen3VLPreprocessOptions())
	if err != nil {
		return Qwen3VLPrompt{}, err
	}
	imageTokens := int(output.Embeddings.Shape.Dims[1])
	text := Qwen35ImagePromptText(beforeImage, afterImage, imageTokens, thinking)
	ids, err := tokenizer.TokenizeText(text, false, true)
	if err != nil {
		return Qwen3VLPrompt{}, fmt.Errorf("projector: tokenize image prompt: %w", err)
	}
	padIDs, err := tokenizer.TokenizeText(Qwen3VLImagePad, false, true)
	if err != nil {
		return Qwen3VLPrompt{}, fmt.Errorf("projector: tokenize image placeholder: %w", err)
	}
	if len(padIDs) != 1 {
		return Qwen3VLPrompt{}, fmt.Errorf("projector: image placeholder maps to %d tokens", len(padIDs))
	}
	imageStart, err := contiguousTokenRun(ids, padIDs[0], imageTokens)
	if err != nil {
		return Qwen3VLPrompt{}, fmt.Errorf("projector: image prompt: %w", err)
	}
	positions, err := Qwen3VLMultiAxisPositions(
		len(ids), imageStart, imageTokens,
		output.GridH, output.GridW, output.MergeSize,
	)
	if err != nil {
		return Qwen3VLPrompt{}, err
	}
	return Qwen3VLPrompt{
		TokenIDs: ids, Embeddings: output.Embeddings.Data,
		EmbeddingWidth: int(output.Embeddings.Shape.Dims[0]), ImageStart: imageStart,
		MultiAxisPositions: positions,
	}, nil
}

func Qwen35ImagePromptText(beforeImage, afterImage string, imageTokens int, thinking bool) string {
	var prompt strings.Builder
	prompt.WriteString("<|im_start|>user\n")
	prompt.WriteString(beforeImage)
	prompt.WriteString("<|vision_start|>")
	prompt.WriteString(strings.Repeat(Qwen3VLImagePad, imageTokens))
	prompt.WriteString("<|vision_end|>")
	prompt.WriteString(afterImage)
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

func Qwen3VLMultiAxisPositions(
	tokens, imageStart, imageTokens, gridH, gridW, merge int,
) ([4][]uint32, error) {
	if tokens <= 0 || imageStart < 0 || imageTokens <= 0 || imageStart+imageTokens > tokens ||
		gridH <= 0 || gridW <= 0 || merge <= 0 || gridH%merge != 0 || gridW%merge != 0 {
		return [4][]uint32{}, errors.New("projector: invalid image position geometry")
	}
	rows, columns := gridH/merge, gridW/merge
	if rows*columns != imageTokens {
		return [4][]uint32{}, fmt.Errorf(
			"projector: image grid produces %d tokens, projector emitted %d", rows*columns, imageTokens,
		)
	}
	var positions [4][]uint32
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
