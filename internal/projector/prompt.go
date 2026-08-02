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
	EmbeddingStart        int
	EmbeddingTokenIndices []uint32
	MultiAxisPositions    [4][]uint32
	AttentionBlocks       []AttentionBlock
}

type AttentionBlock struct {
	Start uint32
	End   uint32
}

type Qwen3VLPrompt = MultimodalPrompt
type Gemma4Prompt = MultimodalPrompt

type ImageProjector interface {
	BuildImagePrompt(context.Context, ImageTokenizer, image.Image, string, string, bool) (MultimodalPrompt, error)
	Close() error
}

type MultiImageProjector interface {
	BuildImagesPrompt(context.Context, ImageTokenizer, []image.Image, []string, bool) (MultimodalPrompt, error)
}

type ImageHistoryProjector interface {
	BuildImagesHistoryPrompt(context.Context, ImageTokenizer, []image.Image, []string) (MultimodalPrompt, error)
}

type AudioProjector interface {
	BuildAudioPrompt(context.Context, ImageTokenizer, []float32, string, string) (MultimodalPrompt, error)
	Close() error
}

type VideoProjector interface {
	BuildVideoPrompt(context.Context, ImageTokenizer, []image.Image, string, string, float64, bool) (MultimodalPrompt, error)
	Close() error
}

type OpenOptions struct {
	CUDA          bool
	DeviceOrdinal int
}

func OpenImageProjector(path string) (ImageProjector, error) {
	return OpenImageProjectorWithOptions(path, OpenOptions{})
}

func OpenImageProjectorWithOptions(path string, options OpenOptions) (ImageProjector, error) {
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
		return OpenQwen3VLWithOptions(path, Qwen3VLOpenOptions(options))
	case gemma4UVProjectorType:
		return OpenGemma4WithOptions(path, Gemma4OpenOptions(options))
	default:
		return nil, fmt.Errorf("projector: image projector type %q is unsupported", projectorType)
	}
}

func OpenAudioProjector(path string) (AudioProjector, error) {
	return OpenAudioProjectorWithOptions(path, OpenOptions{})
}

func OpenAudioProjectorWithOptions(path string, options OpenOptions) (AudioProjector, error) {
	file, err := gguf.Open(path)
	if err != nil {
		return nil, err
	}
	projectorType := ""
	if value, ok := file.MetadataValue("clip.audio.projector_type"); ok && value.Type == gguf.ValueTypeString {
		projectorType, _ = value.Data.(string)
	}
	_ = file.Close()
	switch projectorType {
	case gemma4UAProjectorType:
		return OpenGemma4WithOptions(path, Gemma4OpenOptions(options))
	default:
		return nil, fmt.Errorf("projector: audio projector type %q is unsupported", projectorType)
	}
}

func OpenVideoProjector(path string) (VideoProjector, error) {
	return OpenVideoProjectorWithOptions(path, OpenOptions{})
}

func OpenVideoProjectorWithOptions(path string, options OpenOptions) (VideoProjector, error) {
	projector, err := OpenImageProjectorWithOptions(path, options)
	if err != nil {
		return nil, err
	}
	video, ok := projector.(VideoProjector)
	if !ok {
		_ = projector.Close()
		return nil, errors.New("projector: selected image projector has no video path")
	}
	return video, nil
}

func (r *Qwen3VLRunner) BuildImagePrompt(
	ctx context.Context,
	tokenizer ImageTokenizer,
	source image.Image,
	beforeImage, afterImage string,
	thinking bool,
) (MultimodalPrompt, error) {
	return r.BuildImagesPrompt(ctx, tokenizer, []image.Image{source}, []string{beforeImage, afterImage}, thinking)
}

func (r *Gemma4Runner) BuildImagePrompt(
	ctx context.Context,
	tokenizer ImageTokenizer,
	source image.Image,
	beforeImage, afterImage string,
	_ bool,
) (MultimodalPrompt, error) {
	return r.BuildImagesPrompt(ctx, tokenizer, []image.Image{source}, []string{beforeImage, afterImage}, false)
}

func (r *Gemma4Runner) BuildImagesPrompt(
	ctx context.Context,
	tokenizer ImageTokenizer,
	sources []image.Image,
	text []string,
	_ bool,
) (MultimodalPrompt, error) {
	if tokenizer == nil {
		return MultimodalPrompt{}, errors.New("projector: tokenizer is nil")
	}
	if len(sources) == 0 || len(text) != len(sources)+1 {
		return MultimodalPrompt{}, errors.New("projector: Gemma 4 image/text sequence is inconsistent")
	}
	for _, segment := range text[:len(text)-1] {
		if strings.TrimSpace(segment) != "" {
			return MultimodalPrompt{}, errors.New("projector: Gemma 4 requires images before user text")
		}
	}
	outputs := make([]Gemma4Output, len(sources))
	counts := make([]int, len(sources))
	var prompt strings.Builder
	prompt.WriteString("<bos><|turn>user\n")
	var embeddings []float32
	for index, source := range sources {
		output, err := r.EncodeImage(ctx, source)
		if err != nil {
			return MultimodalPrompt{}, fmt.Errorf("projector: encode Gemma 4 image %d: %w", index, err)
		}
		outputs[index] = output
		counts[index] = int(output.Embeddings.Shape.Dims[1])
		prompt.WriteString("<|image>")
		prompt.WriteString(strings.Repeat("<|image|>", counts[index]))
		prompt.WriteString("<image|>")
		embeddings = append(embeddings, output.Embeddings.Data...)
	}
	prompt.WriteString(strings.TrimSpace(text[len(text)-1]))
	prompt.WriteString("<turn|>\n<|turn>model\n<|channel>thought\n<channel|>")
	ids, err := tokenizer.TokenizeText(prompt.String(), false, true)
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
	starts, err := variableTokenRuns(ids, padIDs[0], counts)
	if err != nil {
		return MultimodalPrompt{}, fmt.Errorf("projector: Gemma 4 image prompt: %w", err)
	}
	indices := make([]uint32, 0)
	blocks := make([]AttentionBlock, len(starts))
	for index, start := range starts {
		indices = append(indices, sequentialTokenIndices(start, counts[index])...)
		blocks[index] = AttentionBlock{Start: uint32(start), End: uint32(start + counts[index])}
	}
	return MultimodalPrompt{
		TokenIDs: ids, Embeddings: embeddings,
		EmbeddingWidth: int(outputs[0].Embeddings.Shape.Dims[0]), EmbeddingStart: starts[0],
		EmbeddingTokenIndices: indices, AttentionBlocks: blocks,
	}, nil
}

func (r *Gemma4Runner) BuildImagesHistoryPrompt(
	ctx context.Context,
	tokenizer ImageTokenizer,
	sources []image.Image,
	text []string,
) (MultimodalPrompt, error) {
	if tokenizer == nil {
		return MultimodalPrompt{}, errors.New("projector: tokenizer is nil")
	}
	if len(sources) == 0 || len(text) != len(sources)+1 {
		return MultimodalPrompt{}, errors.New("projector: Gemma 4 history sequence is inconsistent")
	}
	outputs := make([]Gemma4Output, len(sources))
	counts := make([]int, len(sources))
	var embeddings []float32
	var prompt strings.Builder
	for index, source := range sources {
		prompt.WriteString(text[index])
		output, err := r.EncodeImage(ctx, source)
		if err != nil {
			return MultimodalPrompt{}, fmt.Errorf("projector: encode Gemma 4 history image %d: %w", index, err)
		}
		outputs[index] = output
		counts[index] = int(output.Embeddings.Shape.Dims[1])
		prompt.WriteString("<|image>")
		prompt.WriteString(strings.Repeat("<|image|>", counts[index]))
		prompt.WriteString("<image|>")
		embeddings = append(embeddings, output.Embeddings.Data...)
	}
	prompt.WriteString(text[len(text)-1])
	ids, err := tokenizer.TokenizeText(prompt.String(), true, true)
	if err != nil {
		return MultimodalPrompt{}, fmt.Errorf("projector: tokenize Gemma 4 history: %w", err)
	}
	padIDs, err := tokenizer.TokenizeText("<|image|>", false, true)
	if err != nil {
		return MultimodalPrompt{}, fmt.Errorf("projector: tokenize Gemma 4 image placeholder: %w", err)
	}
	if len(padIDs) != 1 {
		return MultimodalPrompt{}, fmt.Errorf("projector: Gemma 4 image placeholder maps to %d tokens", len(padIDs))
	}
	starts, err := variableTokenRuns(ids, padIDs[0], counts)
	if err != nil {
		return MultimodalPrompt{}, fmt.Errorf("projector: Gemma 4 history: %w", err)
	}
	indices := make([]uint32, 0)
	blocks := make([]AttentionBlock, len(starts))
	for index, start := range starts {
		indices = append(indices, sequentialTokenIndices(start, counts[index])...)
		blocks[index] = AttentionBlock{Start: uint32(start), End: uint32(start + counts[index])}
	}
	return MultimodalPrompt{
		TokenIDs: ids, Embeddings: embeddings,
		EmbeddingWidth: int(outputs[0].Embeddings.Shape.Dims[0]), EmbeddingStart: starts[0],
		EmbeddingTokenIndices: indices, AttentionBlocks: blocks,
	}, nil
}

func Gemma4ImagePromptText(question string, imageTokens int) string {
	return "<bos><|turn>user\n<|image>" + strings.Repeat("<|image|>", imageTokens) +
		"<image|>" + strings.TrimSpace(question) +
		"<turn|>\n<|turn>model\n<|channel>thought\n<channel|>"
}

func (r *Gemma4Runner) BuildAudioPrompt(
	ctx context.Context,
	tokenizer ImageTokenizer,
	samples []float32,
	beforeAudio, afterAudio string,
) (MultimodalPrompt, error) {
	if tokenizer == nil {
		return MultimodalPrompt{}, errors.New("projector: tokenizer is nil")
	}
	if strings.TrimSpace(beforeAudio) != "" {
		return MultimodalPrompt{}, errors.New("projector: Gemma 4 requires audio before user text")
	}
	output, err := r.EncodeAudio(ctx, samples)
	if err != nil {
		return MultimodalPrompt{}, err
	}
	audioTokens := int(output.Embeddings.Shape.Dims[1])
	text := Gemma4AudioPromptText(afterAudio, audioTokens)
	ids, err := tokenizer.TokenizeText(text, false, true)
	if err != nil {
		return MultimodalPrompt{}, fmt.Errorf("projector: tokenize Gemma 4 audio prompt: %w", err)
	}
	padIDs, err := tokenizer.TokenizeText("<|audio|>", false, true)
	if err != nil {
		return MultimodalPrompt{}, fmt.Errorf("projector: tokenize Gemma 4 audio placeholder: %w", err)
	}
	if len(padIDs) != 1 {
		return MultimodalPrompt{}, fmt.Errorf("projector: Gemma 4 audio placeholder maps to %d tokens", len(padIDs))
	}
	audioStart, err := contiguousTokenRun(ids, padIDs[0], audioTokens)
	if err != nil {
		return MultimodalPrompt{}, fmt.Errorf("projector: Gemma 4 audio prompt: %w", err)
	}
	return MultimodalPrompt{
		TokenIDs: ids, Embeddings: output.Embeddings.Data,
		EmbeddingWidth: int(output.Embeddings.Shape.Dims[0]), EmbeddingStart: audioStart,
		EmbeddingTokenIndices: sequentialTokenIndices(audioStart, audioTokens),
	}, nil
}

func Gemma4AudioPromptText(question string, audioTokens int) string {
	return "<bos><|turn>user\n<|audio>" + strings.Repeat("<|audio|>", audioTokens) +
		"<audio|>" + strings.TrimSpace(question) +
		"<turn|>\n<|turn>model\n<|channel>thought\n<channel|>"
}

func (r *Gemma4Runner) BuildVideoPrompt(
	ctx context.Context,
	tokenizer ImageTokenizer,
	frames []image.Image,
	beforeVideo, afterVideo string,
	fps float64,
	_ bool,
) (MultimodalPrompt, error) {
	if tokenizer == nil {
		return MultimodalPrompt{}, errors.New("projector: tokenizer is nil")
	}
	if strings.TrimSpace(beforeVideo) != "" {
		return MultimodalPrompt{}, errors.New("projector: Gemma 4 requires video before user text")
	}
	if fps <= 0 || math.IsNaN(fps) || math.IsInf(fps, 0) {
		return MultimodalPrompt{}, errors.New("projector: video FPS must be positive and finite")
	}
	output, err := r.EncodeVideoFrames(ctx, frames)
	if err != nil {
		return MultimodalPrompt{}, err
	}
	text := Gemma4VideoPromptText(afterVideo, output.Frames, output.TokensPerFrame, fps)
	ids, err := tokenizer.TokenizeText(text, false, true)
	if err != nil {
		return MultimodalPrompt{}, fmt.Errorf("projector: tokenize Gemma 4 video prompt: %w", err)
	}
	padIDs, err := tokenizer.TokenizeText("<|video|>", false, true)
	if err != nil {
		return MultimodalPrompt{}, fmt.Errorf("projector: tokenize Gemma 4 video placeholder: %w", err)
	}
	if len(padIDs) != 1 {
		return MultimodalPrompt{}, fmt.Errorf("projector: Gemma 4 video placeholder maps to %d tokens", len(padIDs))
	}
	starts, err := contiguousTokenRuns(ids, padIDs[0], output.TokensPerFrame, output.Frames)
	if err != nil {
		return MultimodalPrompt{}, fmt.Errorf("projector: Gemma 4 video prompt: %w", err)
	}
	indices := make([]uint32, 0, output.Frames*output.TokensPerFrame)
	blocks := make([]AttentionBlock, len(starts))
	for index, start := range starts {
		indices = append(indices, sequentialTokenIndices(start, output.TokensPerFrame)...)
		blocks[index] = AttentionBlock{
			Start: uint32(start), End: uint32(start + output.TokensPerFrame),
		}
	}
	return MultimodalPrompt{
		TokenIDs: ids, Embeddings: output.Embeddings.Data,
		EmbeddingWidth: int(output.Embeddings.Shape.Dims[0]), EmbeddingStart: starts[0],
		EmbeddingTokenIndices: indices, AttentionBlocks: blocks,
	}, nil
}

func Gemma4VideoPromptText(question string, frames, tokensPerFrame int, fps float64) string {
	var prompt strings.Builder
	prompt.WriteString("<bos><|turn>user\n")
	for frame := 0; frame < frames; frame++ {
		if frame > 0 {
			prompt.WriteByte(' ')
		}
		seconds := int(float64(frame) / fps)
		fmt.Fprintf(&prompt, "%02d:%02d <|image>", seconds/60, seconds%60)
		prompt.WriteString(strings.Repeat("<|video|>", tokensPerFrame))
		prompt.WriteString("<image|>")
	}
	prompt.WriteString(strings.TrimSpace(question))
	prompt.WriteString("<turn|>\n<|turn>model\n<|channel>thought\n<channel|>")
	return prompt.String()
}

func (r *Qwen3VLRunner) BuildQwen35ImagePrompt(
	ctx context.Context,
	tokenizer Qwen3VLTokenizer,
	source image.Image,
	beforeImage, afterImage string,
	thinking bool,
) (Qwen3VLPrompt, error) {
	return r.BuildImagesPrompt(ctx, tokenizer, []image.Image{source}, []string{beforeImage, afterImage}, thinking)
}

func (r *Qwen3VLRunner) BuildImagesPrompt(
	ctx context.Context,
	tokenizer ImageTokenizer,
	sources []image.Image,
	text []string,
	thinking bool,
) (MultimodalPrompt, error) {
	if tokenizer == nil {
		return MultimodalPrompt{}, errors.New("projector: tokenizer is nil")
	}
	if len(sources) == 0 || len(text) != len(sources)+1 {
		return MultimodalPrompt{}, errors.New("projector: Qwen3-VL image/text sequence is inconsistent")
	}
	type geometry struct{ rows, columns int }
	counts := make([]int, len(sources))
	geometries := make([]geometry, len(sources))
	var embeddings []float32
	var prompt strings.Builder
	prompt.WriteString("<|im_start|>user\n")
	for index, source := range sources {
		prompt.WriteString(text[index])
		output, err := r.EncodeImage(ctx, source, DefaultQwen3VLPreprocessOptions())
		if err != nil {
			return MultimodalPrompt{}, fmt.Errorf("projector: encode Qwen3-VL image %d: %w", index, err)
		}
		counts[index] = int(output.Embeddings.Shape.Dims[1])
		geometries[index] = geometry{output.GridH / output.MergeSize, output.GridW / output.MergeSize}
		prompt.WriteString("<|vision_start|>")
		prompt.WriteString(strings.Repeat(Qwen3VLImagePad, counts[index]))
		prompt.WriteString("<|vision_end|>")
		embeddings = append(embeddings, output.Embeddings.Data...)
	}
	prompt.WriteString(text[len(text)-1])
	prompt.WriteString("<|im_end|>\n<|im_start|>assistant\n<think>\n")
	if !thinking {
		prompt.WriteString("\n</think>\n\n")
	}
	ids, err := tokenizer.TokenizeText(prompt.String(), false, true)
	if err != nil {
		return MultimodalPrompt{}, fmt.Errorf("projector: tokenize image prompt: %w", err)
	}
	padIDs, err := tokenizer.TokenizeText(Qwen3VLImagePad, false, true)
	if err != nil {
		return Qwen3VLPrompt{}, fmt.Errorf("projector: tokenize image placeholder: %w", err)
	}
	if len(padIDs) != 1 {
		return Qwen3VLPrompt{}, fmt.Errorf("projector: image placeholder maps to %d tokens", len(padIDs))
	}
	starts, err := variableTokenRuns(ids, padIDs[0], counts)
	if err != nil {
		return MultimodalPrompt{}, fmt.Errorf("projector: image prompt: %w", err)
	}
	chunks := make([]Qwen3VLPositionChunk, len(starts))
	var indices []uint32
	for index, start := range starts {
		chunks[index] = Qwen3VLPositionChunk{Start: start, Rows: geometries[index].rows, Columns: geometries[index].columns}
		indices = append(indices, sequentialTokenIndices(start, counts[index])...)
	}
	positions, err := Qwen3VLVariableChunkPositions(len(ids), chunks)
	if err != nil {
		return MultimodalPrompt{}, err
	}
	return MultimodalPrompt{
		TokenIDs: ids, Embeddings: embeddings,
		EmbeddingWidth: r.spec.OutputHidden, EmbeddingStart: starts[0],
		EmbeddingTokenIndices: indices,
		MultiAxisPositions:    positions,
	}, nil
}

func (r *Qwen3VLRunner) BuildImagesHistoryPrompt(
	ctx context.Context,
	tokenizer ImageTokenizer,
	sources []image.Image,
	text []string,
) (MultimodalPrompt, error) {
	if tokenizer == nil {
		return MultimodalPrompt{}, errors.New("projector: tokenizer is nil")
	}
	if len(sources) == 0 || len(text) != len(sources)+1 {
		return MultimodalPrompt{}, errors.New("projector: Qwen3-VL history sequence is inconsistent")
	}
	type geometry struct{ rows, columns int }
	counts := make([]int, len(sources))
	geometries := make([]geometry, len(sources))
	var embeddings []float32
	var prompt strings.Builder
	for index, source := range sources {
		prompt.WriteString(text[index])
		output, err := r.EncodeImage(ctx, source, DefaultQwen3VLPreprocessOptions())
		if err != nil {
			return MultimodalPrompt{}, fmt.Errorf("projector: encode Qwen3-VL history image %d: %w", index, err)
		}
		counts[index] = int(output.Embeddings.Shape.Dims[1])
		geometries[index] = geometry{output.GridH / output.MergeSize, output.GridW / output.MergeSize}
		prompt.WriteString("<|vision_start|>")
		prompt.WriteString(strings.Repeat(Qwen3VLImagePad, counts[index]))
		prompt.WriteString("<|vision_end|>")
		embeddings = append(embeddings, output.Embeddings.Data...)
	}
	prompt.WriteString(text[len(text)-1])
	ids, err := tokenizer.TokenizeText(prompt.String(), true, true)
	if err != nil {
		return MultimodalPrompt{}, fmt.Errorf("projector: tokenize Qwen3-VL history: %w", err)
	}
	padIDs, err := tokenizer.TokenizeText(Qwen3VLImagePad, false, true)
	if err != nil {
		return MultimodalPrompt{}, fmt.Errorf("projector: tokenize image placeholder: %w", err)
	}
	if len(padIDs) != 1 {
		return MultimodalPrompt{}, fmt.Errorf("projector: image placeholder maps to %d tokens", len(padIDs))
	}
	starts, err := variableTokenRuns(ids, padIDs[0], counts)
	if err != nil {
		return MultimodalPrompt{}, fmt.Errorf("projector: Qwen3-VL history: %w", err)
	}
	chunks := make([]Qwen3VLPositionChunk, len(starts))
	var indices []uint32
	for index, start := range starts {
		chunks[index] = Qwen3VLPositionChunk{Start: start, Rows: geometries[index].rows, Columns: geometries[index].columns}
		indices = append(indices, sequentialTokenIndices(start, counts[index])...)
	}
	positions, err := Qwen3VLVariableChunkPositions(len(ids), chunks)
	if err != nil {
		return MultimodalPrompt{}, err
	}
	return MultimodalPrompt{
		TokenIDs: ids, Embeddings: embeddings,
		EmbeddingWidth: r.spec.OutputHidden, EmbeddingStart: starts[0],
		EmbeddingTokenIndices: indices, MultiAxisPositions: positions,
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

func (r *Qwen3VLRunner) BuildVideoPrompt(
	ctx context.Context,
	tokenizer ImageTokenizer,
	frames []image.Image,
	beforeVideo, afterVideo string,
	fps float64,
	thinking bool,
) (MultimodalPrompt, error) {
	return r.BuildQwen35VideoPrompt(ctx, tokenizer, frames, beforeVideo, afterVideo, fps, thinking)
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

func variableTokenRuns(ids []tokenizer.TokenID, token tokenizer.TokenID, counts []int) ([]int, error) {
	if len(counts) == 0 {
		return nil, errors.New("media token counts are empty")
	}
	starts := make([]int, 0, len(counts))
	for index := 0; index < len(ids); {
		if ids[index] != token {
			index++
			continue
		}
		end := index + 1
		for end < len(ids) && ids[end] == token {
			end++
		}
		starts = append(starts, index)
		if len(starts) > len(counts) || end-index != counts[len(starts)-1] {
			return nil, fmt.Errorf("placeholder run %d has %d tokens", len(starts)-1, end-index)
		}
		index = end
	}
	if len(starts) != len(counts) {
		return nil, fmt.Errorf("placeholder runs = %d, want %d", len(starts), len(counts))
	}
	return starts, nil
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

type Qwen3VLPositionChunk struct {
	Start   int
	Rows    int
	Columns int
}

func Qwen3VLVariableChunkPositions(tokens int, chunks []Qwen3VLPositionChunk) ([4][]uint32, error) {
	if tokens <= 0 || len(chunks) == 0 {
		return [4][]uint32{}, errors.New("projector: invalid variable media positions")
	}
	var positions [4][]uint32
	for axis := range positions {
		positions[axis] = make([]uint32, tokens)
	}
	next, physical := uint32(0), 0
	for _, chunk := range chunks {
		count := chunk.Rows * chunk.Columns
		if chunk.Rows <= 0 || chunk.Columns <= 0 || chunk.Start < physical || chunk.Start+count > tokens {
			return [4][]uint32{}, errors.New("projector: variable media chunk is invalid")
		}
		for physical < chunk.Start {
			for axis := range positions {
				positions[axis][physical] = next
			}
			next++
			physical++
		}
		base := next
		for index := 0; index < count; index++ {
			positions[0][physical+index] = base
			positions[1][physical+index] = base + uint32(index/chunk.Columns)
			positions[2][physical+index] = base + uint32(index%chunk.Columns)
			positions[3][physical+index] = 0
		}
		physical += count
		next = base + uint32(max(chunk.Rows, chunk.Columns))
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
