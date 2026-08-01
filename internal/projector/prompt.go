package projector

import (
	"context"
	"errors"
	"fmt"
	"image"
	"math"
	"strings"

	"llamacpp2go/internal/gguf"
	"llamacpp2go/internal/tokenizer"
)

const Qwen3VLImagePad = "<|image_pad|>"
const Qwen3VLVideoPad = "<|video_pad|>"

type ImageTokenizer interface {
	TokenizeText(string, bool, bool) ([]tokenizer.TokenID, error)
}

type Qwen3VLTokenizer = ImageTokenizer

type MultimodalPrompt struct {
	TokenIDs              []tokenizer.TokenID
	Embeddings            []float32
	EmbeddingWidth        int
	ImageStart            int
	EmbeddingTokenIndices []uint32
	MultiAxisPositions    [4][]uint32
}

type Qwen3VLPrompt = MultimodalPrompt
type Gemma4Prompt = MultimodalPrompt

type ImageProjector interface {
	BuildImagePrompt(context.Context, ImageTokenizer, image.Image, string, string, bool) (MultimodalPrompt, error)
	Close() error
}

func OpenImageProjector(path string) (ImageProjector, error) {
	file, err := gguf.Open(path)
	if err != nil {
		return nil, err
	}
	projectorType := ""
	for _, key := range []string{"clip.projector_type", "clip.vision.projector_type"} {
		if value, ok := file.MetadataValue(key); ok && value.Type == gguf.ValueTypeString {
			projectorType, _ = value.Data.(string)
			if projectorType != "" {
				break
			}
		}
	}
	_ = file.Close()
	switch projectorType {
	case qwen3VLProjectorType:
		return OpenQwen3VL(path)
	case gemma4UVProjectorType:
		return OpenGemma4(path)
	default:
		return nil, fmt.Errorf("projector: image projector type %q is unsupported", projectorType)
	}
}

func (r *Qwen3VLRunner) BuildImagePrompt(
	ctx context.Context,
	tokenizer ImageTokenizer,
	source image.Image,
	beforeImage, afterImage string,
	thinking bool,
) (MultimodalPrompt, error) {
	return r.BuildQwen35ImagePrompt(ctx, tokenizer, source, beforeImage, afterImage, thinking)
}

func (r *Gemma4Runner) BuildImagePrompt(
	ctx context.Context,
	tokenizer ImageTokenizer,
	source image.Image,
	beforeImage, afterImage string,
	_ bool,
) (MultimodalPrompt, error) {
	if tokenizer == nil {
		return MultimodalPrompt{}, errors.New("projector: tokenizer is nil")
	}
	if strings.TrimSpace(beforeImage) != "" {
		return MultimodalPrompt{}, errors.New("projector: Gemma 4 requires the image before user text")
	}
	output, err := r.EncodeImage(ctx, source)
	if err != nil {
		return MultimodalPrompt{}, err
	}
	imageTokens := int(output.Embeddings.Shape.Dims[1])
	text := Gemma4ImagePromptText(afterImage, imageTokens)
	ids, err := tokenizer.TokenizeText(text, false, true)
	if err != nil {
		return MultimodalPrompt{}, fmt.Errorf("projector: tokenize Gemma 4 image prompt: %w", err)
	}
	padIDs, err := tokenizer.TokenizeText("<|image|>", false, true)
	if err != nil {
		return MultimodalPrompt{}, fmt.Errorf("projector: tokenize Gemma 4 image placeholder: %w", err)
	}
	if len(padIDs) != 1 {
		return MultimodalPrompt{}, fmt.Errorf("projector: Gemma 4 image placeholder maps to %d tokens", len(padIDs))
	}
	imageStart, err := contiguousTokenRun(ids, padIDs[0], imageTokens)
	if err != nil {
		return MultimodalPrompt{}, fmt.Errorf("projector: Gemma 4 image prompt: %w", err)
	}
	return MultimodalPrompt{
		TokenIDs: ids, Embeddings: output.Embeddings.Data,
		EmbeddingWidth: int(output.Embeddings.Shape.Dims[0]), ImageStart: imageStart,
		EmbeddingTokenIndices: sequentialTokenIndices(imageStart, imageTokens),
	}, nil
}

func Gemma4ImagePromptText(question string, imageTokens int) string {
	return "<bos><|turn>user\n<|image>" + strings.Repeat("<|image|>", imageTokens) +
		"<image|>" + strings.TrimSpace(question) +
		"<turn|>\n<|turn>model\n<|channel>thought\n<channel|>"
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
		EmbeddingTokenIndices: sequentialTokenIndices(imageStart, imageTokens),
		MultiAxisPositions:    positions,
	}, nil
}

func (r *Qwen3VLRunner) BuildQwen35VideoPrompt(
	ctx context.Context,
	tokenizer Qwen3VLTokenizer,
	frames []image.Image,
	beforeVideo, afterVideo string,
	fps float64,
	thinking bool,
) (Qwen3VLPrompt, error) {
	if tokenizer == nil {
		return Qwen3VLPrompt{}, errors.New("projector: tokenizer is nil")
	}
	if fps <= 0 || math.IsNaN(fps) || math.IsInf(fps, 0) {
		return Qwen3VLPrompt{}, errors.New("projector: video FPS must be positive and finite")
	}
	output, err := r.EncodeFrames(ctx, frames, DefaultQwen3VLVideoPreprocessOptions())
	if err != nil {
		return Qwen3VLPrompt{}, err
	}
	rows, columns := output.GridH/output.MergeSize, output.GridW/output.MergeSize
	perGroup := rows * columns
	text := Qwen35VideoPromptText(beforeVideo, afterVideo, output.GridT, perGroup, fps, thinking)
	ids, err := tokenizer.TokenizeText(text, false, true)
	if err != nil {
		return Qwen3VLPrompt{}, fmt.Errorf("projector: tokenize video prompt: %w", err)
	}
	padIDs, err := tokenizer.TokenizeText(Qwen3VLVideoPad, false, true)
	if err != nil {
		return Qwen3VLPrompt{}, fmt.Errorf("projector: tokenize video placeholder: %w", err)
	}
	if len(padIDs) != 1 {
		return Qwen3VLPrompt{}, fmt.Errorf("projector: video placeholder maps to %d tokens", len(padIDs))
	}
	starts, err := contiguousTokenRuns(ids, padIDs[0], perGroup, output.GridT)
	if err != nil {
		return Qwen3VLPrompt{}, fmt.Errorf("projector: video prompt: %w", err)
	}
	positions, err := Qwen3VLMultiChunkPositions(len(ids), starts, perGroup, rows, columns)
	if err != nil {
		return Qwen3VLPrompt{}, err
	}
	indices := make([]uint32, 0, output.GridT*perGroup)
	for _, start := range starts {
		indices = append(indices, sequentialTokenIndices(start, perGroup)...)
	}
	return Qwen3VLPrompt{
		TokenIDs: ids, Embeddings: output.Embeddings.Data,
		EmbeddingWidth:        int(output.Embeddings.Shape.Dims[0]),
		EmbeddingTokenIndices: indices, MultiAxisPositions: positions,
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

func Qwen35VideoPromptText(beforeVideo, afterVideo string, groups, tokensPerGroup int, fps float64, thinking bool) string {
	var prompt strings.Builder
	prompt.WriteString("<|im_start|>user\n")
	prompt.WriteString(beforeVideo)
	prompt.WriteString("<|vision_start|>")
	for group := 0; group < groups; group++ {
		timestamp := (float64(group*2) + 0.5) / fps
		fmt.Fprintf(&prompt, "<%.1f seconds><|vision_start|>", timestamp)
		prompt.WriteString(strings.Repeat(Qwen3VLVideoPad, tokensPerGroup))
		prompt.WriteString("<|vision_end|>")
	}
	prompt.WriteString("<|vision_end|>")
	prompt.WriteString(afterVideo)
	prompt.WriteString("<|im_end|>\n<|im_start|>assistant\n<think>\n")
	if !thinking {
		prompt.WriteString("\n</think>\n\n")
	}
	return prompt.String()
}

func contiguousTokenRun(ids []tokenizer.TokenID, token tokenizer.TokenID, count int) (int, error) {
	starts, err := contiguousTokenRuns(ids, token, count, 1)
	if err != nil {
		return 0, err
	}
	return starts[0], nil
}

func contiguousTokenRuns(ids []tokenizer.TokenID, token tokenizer.TokenID, count, runs int) ([]int, error) {
	if count <= 0 {
		return nil, errors.New("media token count is not positive")
	}
	found := make([]int, 0, runs)
	for start := 0; start+count <= len(ids); start++ {
		if ids[start] != token {
			continue
		}
		end := start
		for end < len(ids) && ids[end] == token {
			end++
		}
		if end-start != count {
			return nil, fmt.Errorf("placeholder run has %d tokens, want %d", end-start, count)
		}
		found = append(found, start)
		start = end - 1
	}
	if len(found) != runs {
		return nil, fmt.Errorf("placeholder runs = %d, want %d", len(found), runs)
	}
	return found, nil
}

func sequentialTokenIndices(start, count int) []uint32 {
	indices := make([]uint32, count)
	for index := range indices {
		indices[index] = uint32(start + index)
	}
	return indices
}

func Qwen3VLMultiAxisPositions(
	tokens, imageStart, imageTokens, gridH, gridW, merge int,
) ([4][]uint32, error) {
	if gridH <= 0 || gridW <= 0 || merge <= 0 || gridH%merge != 0 || gridW%merge != 0 {
		return [4][]uint32{}, errors.New("projector: invalid image position geometry")
	}
	rows, columns := gridH/merge, gridW/merge
	if rows*columns != imageTokens {
		return [4][]uint32{}, fmt.Errorf(
			"projector: image grid produces %d tokens, projector emitted %d", rows*columns, imageTokens,
		)
	}
	return Qwen3VLMultiChunkPositions(tokens, []int{imageStart}, imageTokens, rows, columns)
}

func Qwen3VLMultiChunkPositions(tokens int, starts []int, tokensPerChunk, rows, columns int) ([4][]uint32, error) {
	if tokens <= 0 || len(starts) == 0 || tokensPerChunk <= 0 || rows <= 0 || columns <= 0 ||
		rows*columns != tokensPerChunk {
		return [4][]uint32{}, errors.New("projector: invalid media position geometry")
	}
	var positions [4][]uint32
	for axis := range positions {
		positions[axis] = make([]uint32, tokens)
	}
	next, physical := uint32(0), 0
	for _, start := range starts {
		if start < physical || start+tokensPerChunk > tokens {
			return [4][]uint32{}, errors.New("projector: media chunks overlap or exceed prompt")
		}
		for physical < start {
			for axis := range positions {
				positions[axis][physical] = next
			}
			next++
			physical++
		}
		base := next
		for index := 0; index < tokensPerChunk; index++ {
			positions[0][physical+index] = base
			positions[1][physical+index] = base + uint32(index/columns)
			positions[2][physical+index] = base + uint32(index%columns)
			positions[3][physical+index] = 0
		}
		physical += tokensPerChunk
		next = base + uint32(max(rows, columns))
	}
	for physical < tokens {
		for axis := range positions {
			positions[axis][physical] = next
		}
		next++
		physical++
	}
	return positions, nil
}
