package projector

import (
	"context"
	"errors"
	"fmt"
	"image"
	"math"
	"slices"
	"strings"

	"llamacpp2go/internal/gguf"
	"llamacpp2go/internal/tokenizer"
)

const Qwen3VLImagePad = "<|image_pad|>"
const Qwen3VLVideoPad = "<|video_pad|>"
const MiMoVLSystemPrompt = "You are MiMo, an AI assistant developed by Xiaomi."
const PaddleOCRImagePad = "<|IMAGE_PLACEHOLDER|>"
const Llama4ImageStart = "<|image_start|>"
const Llama4ImageEnd = "<|image_end|>"
const Llama4ImagePad = "<|image|>"
const Granite4VisionImageToken = "<image>"
const HunyuanVLImageStart = "<｜hy_place▁holder▁no▁100｜>"
const HunyuanVLImageEnd = "<｜hy_place▁holder▁no▁101｜>"
const HunyuanVLImagePad = "<｜hy_place▁holder▁no▁102｜>"

type ImageTokenizer interface {
	TokenizeText(string, bool, bool) ([]tokenizer.TokenID, error)
}

type Qwen3VLTokenizer = ImageTokenizer

type MultimodalPrompt struct {
	TokenIDs              []tokenizer.TokenID
	Embeddings            []float32
	DeepstackEmbeddings   [][]float32
	EmbeddingWidth        int
	EmbeddingStart        int
	EmbeddingTokenIndices []uint32
	MultiAxisPositions    [4][]uint32
	AttentionBlocks       []AttentionBlock
	VisualBlocks          []AttentionBlock
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

type MediaKind uint8

const (
	MediaImage MediaKind = iota + 1
	MediaAudio
)

type MediaInput struct {
	Kind  MediaKind
	Image image.Image
	Audio []float32
}

type MediaHistoryProjector interface {
	BuildMediaHistoryPrompt(context.Context, ImageTokenizer, []MediaInput, []string) (MultimodalPrompt, error)
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

func validateImagePromptInputs(
	tokenizer ImageTokenizer,
	sources []image.Image,
	text []string,
	family string,
) error {
	return validatePromptInputs(tokenizer, sources, text, family+" image/text")
}

func validatePromptInputs(
	tokenizer ImageTokenizer,
	sources []image.Image,
	text []string,
	sequenceLabel string,
) error {
	if tokenizer == nil {
		return errors.New("projector: tokenizer is nil")
	}
	if len(sources) == 0 || len(text) != len(sources)+1 {
		return fmt.Errorf("projector: %s sequence is inconsistent", sequenceLabel)
	}
	return nil
}

func tokenizeImagePromptRuns(
	tokenizer ImageTokenizer,
	prompt string,
	history bool,
	placeholder string,
	counts []int,
	family string,
	placeholderLabel string,
) ([]tokenizer.TokenID, []int, error) {
	return tokenizePromptRuns(
		tokenizer, prompt, history, placeholder, counts,
		family+" image prompt", placeholderLabel, family+" image prompt",
	)
}

func tokenizePromptRuns(
	tokenizer ImageTokenizer,
	prompt string,
	history bool,
	placeholder string,
	counts []int,
	promptLabel string,
	placeholderLabel string,
	runsLabel string,
) ([]tokenizer.TokenID, []int, error) {
	ids, err := tokenizer.TokenizeText(prompt, history, true)
	if err != nil {
		return nil, nil, fmt.Errorf("projector: tokenize %s: %w", promptLabel, err)
	}
	placeholderIDs, err := tokenizer.TokenizeText(placeholder, false, true)
	if err != nil {
		return nil, nil, fmt.Errorf("projector: tokenize %s: %w", placeholderLabel, err)
	}
	if len(placeholderIDs) != 1 {
		return nil, nil, fmt.Errorf("projector: %s maps to %d tokens", placeholderLabel, len(placeholderIDs))
	}
	starts, err := variableTokenRuns(ids, placeholderIDs[0], counts)
	if err != nil {
		return nil, nil, fmt.Errorf("projector: %s: %w", runsLabel, err)
	}
	return ids, starts, nil
}

func embeddingTokenIndices(starts, counts []int, offset int) []uint32 {
	var indices []uint32
	for index, start := range starts {
		indices = append(indices, sequentialTokenIndices(start+offset, counts[index])...)
	}
	return indices
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
	case deepSeekOCR2ProjectorType:
		return OpenDeepSeekOCR2WithOptions(path, DeepSeekOCR2OpenOptions(options))
	case deepSeekOCRProjectorType:
		return OpenDeepSeekOCRWithOptions(path, DeepSeekOCROpenOptions(options))
	case cogVLMProjectorType:
		return OpenCogVLMVisionWithOptions(path, CogVLMVisionOpenOptions(options))
	case gemma3nVisionProjectorType:
		return OpenGemma3nVisionWithOptions(path, Gemma3nVisionOpenOptions(options))
	case mimoVLProjectorType:
		return OpenMiMoVLWithOptions(path, MiMoVLOpenOptions(options))
	case granite4VisionProjectorType:
		return OpenGranite4VisionWithOptions(path, Granite4VisionOpenOptions(options))
	case llama4ProjectorType:
		return OpenLlama4VisionWithOptions(path, Llama4VisionOpenOptions(options))
	case hunyuanVLProjectorType:
		return OpenHunyuanVLWithOptions(path, HunyuanVLOpenOptions(options))
	case paddleOCRProjectorType:
		return OpenPaddleOCRWithOptions(path, PaddleOCROpenOptions(options))
	case qwen2VLProjectorType:
		return OpenQwen2VLWithOptions(path, Qwen2VLOpenOptions(options))
	case qwen3VLProjectorType:
		return OpenQwen3VLWithOptions(path, Qwen3VLOpenOptions(options))
	case gemma4UVProjectorType:
		return OpenGemma4WithOptions(path, Gemma4OpenOptions(options))
	default:
		return nil, fmt.Errorf("projector: image projector type %q is unsupported", projectorType)
	}
}

func (r *Granite4VisionRunner) BuildImagePrompt(
	ctx context.Context,
	tokenizer ImageTokenizer,
	source image.Image,
	beforeImage, afterImage string,
	_ bool,
) (MultimodalPrompt, error) {
	return r.BuildImagesPrompt(ctx, tokenizer, []image.Image{source}, []string{beforeImage, afterImage}, false)
}

func (r *Granite4VisionRunner) BuildImagesPrompt(
	ctx context.Context,
	tokenizer ImageTokenizer,
	sources []image.Image,
	text []string,
	_ bool,
) (MultimodalPrompt, error) {
	return r.buildImagesPrompt(ctx, tokenizer, sources, text, false)
}

func (r *Granite4VisionRunner) BuildImagesHistoryPrompt(
	ctx context.Context,
	tokenizer ImageTokenizer,
	sources []image.Image,
	text []string,
) (MultimodalPrompt, error) {
	return r.buildImagesPrompt(ctx, tokenizer, sources, text, true)
}

func (r *Granite4VisionRunner) buildImagesPrompt(
	ctx context.Context,
	tokenizer ImageTokenizer,
	sources []image.Image,
	text []string,
	history bool,
) (MultimodalPrompt, error) {
	plan := imagePromptPlan{
		Family: "Granite 4 Vision", Placeholder: Granite4VisionImageToken,
		PlaceholderLabel: "Granite 4 Vision image token", History: history,
		EmbeddingWidth: r.spec.ProjectionDim, EmbeddingOffset: 1,
		Render: func(text []string, items []imagePromptItem) string {
			var prompt strings.Builder
			if !history {
				prompt.WriteString("<|start_of_role|>user<|end_of_role|>\n")
			}
			for index, item := range items {
				prompt.WriteString(text[index])
				prompt.WriteString(strings.Repeat(Granite4VisionImageToken, item.RunCount))
			}
			prompt.WriteString(text[len(text)-1])
			if !history {
				prompt.WriteString("<|end_of_text|>\n<|start_of_role|>assistant<|end_of_role|>\n")
			}
			return prompt.String()
		},
	}
	return executeImagePromptPlan(ctx, tokenizer, sources, text, plan, func(ctx context.Context, source image.Image) (imagePromptItem, error) {
		output, err := r.EncodeImage(ctx, source)
		if err != nil {
			return imagePromptItem{}, err
		}
		deepstack := make([][]float32, len(output.DeepstackEmbeddings))
		for index := range output.DeepstackEmbeddings {
			deepstack[index] = output.DeepstackEmbeddings[index].Data
		}
		count := int(output.Embeddings.Shape.Dims[1])
		return imagePromptItem{
			Embeddings: output.Embeddings.Data, Deepstack: deepstack, Count: count, RunCount: count + 1,
		}, nil
	})
}

func (r *Llama4VisionRunner) BuildImagePrompt(
	ctx context.Context,
	tokenizer ImageTokenizer,
	source image.Image,
	beforeImage, afterImage string,
	_ bool,
) (MultimodalPrompt, error) {
	return r.BuildImagesPrompt(ctx, tokenizer, []image.Image{source}, []string{beforeImage, afterImage}, false)
}

func (r *Llama4VisionRunner) BuildImagesPrompt(
	ctx context.Context,
	tokenizer ImageTokenizer,
	sources []image.Image,
	text []string,
	_ bool,
) (MultimodalPrompt, error) {
	return r.buildImagesPrompt(ctx, tokenizer, sources, text, false)
}

func (r *Llama4VisionRunner) BuildImagesHistoryPrompt(
	ctx context.Context,
	tokenizer ImageTokenizer,
	sources []image.Image,
	text []string,
) (MultimodalPrompt, error) {
	return r.buildImagesPrompt(ctx, tokenizer, sources, text, true)
}

func (r *Llama4VisionRunner) buildImagesPrompt(
	ctx context.Context,
	tokenizer ImageTokenizer,
	sources []image.Image,
	text []string,
	history bool,
) (MultimodalPrompt, error) {
	plan := imagePromptPlan{
		Family: "Llama-4", Placeholder: Llama4ImagePad, PlaceholderLabel: "Llama-4 placeholder",
		History: history, EmbeddingWidth: r.spec.OutputHidden,
		Render: func(text []string, items []imagePromptItem) string {
			var prompt strings.Builder
			if !history {
				prompt.WriteString("<|begin_of_text|><|header_start|>user<|header_end|>\n\n")
			}
			for index, item := range items {
				prompt.WriteString(text[index])
				prompt.WriteString(Llama4ImageStart)
				prompt.WriteString(strings.Repeat(Llama4ImagePad, item.RunCount))
				prompt.WriteString(Llama4ImageEnd)
			}
			prompt.WriteString(text[len(text)-1])
			if !history {
				prompt.WriteString("<|eot|><|header_start|>assistant<|header_end|>\n\n")
			}
			return prompt.String()
		},
	}
	return executeImagePromptPlan(ctx, tokenizer, sources, text, plan, func(ctx context.Context, source image.Image) (imagePromptItem, error) {
		output, err := r.EncodeImage(ctx, source)
		if err != nil {
			return imagePromptItem{}, err
		}
		return imagePromptItem{Embeddings: output.Embeddings.Data, Count: int(output.Embeddings.Shape.Dims[1])}, nil
	})
}

func (r *HunyuanVLRunner) BuildImagePrompt(
	ctx context.Context,
	tokenizer ImageTokenizer,
	source image.Image,
	beforeImage, afterImage string,
	_ bool,
) (MultimodalPrompt, error) {
	return r.BuildImagesPrompt(ctx, tokenizer, []image.Image{source}, []string{beforeImage, afterImage}, false)
}

func (r *HunyuanVLRunner) BuildImagesPrompt(
	ctx context.Context,
	tokenizer ImageTokenizer,
	sources []image.Image,
	text []string,
	_ bool,
) (MultimodalPrompt, error) {
	return r.buildImagesPrompt(ctx, tokenizer, sources, text, false)
}

func (r *HunyuanVLRunner) BuildImagesHistoryPrompt(
	ctx context.Context,
	tokenizer ImageTokenizer,
	sources []image.Image,
	text []string,
) (MultimodalPrompt, error) {
	return r.buildImagesPrompt(ctx, tokenizer, sources, text, true)
}

func (r *HunyuanVLRunner) buildImagesPrompt(
	ctx context.Context,
	tokenizer ImageTokenizer,
	sources []image.Image,
	text []string,
	history bool,
) (MultimodalPrompt, error) {
	plan := imagePromptPlan{
		Family: "Hunyuan-VL", Placeholder: HunyuanVLImagePad, PlaceholderLabel: "Hunyuan-VL placeholder",
		History: history, EmbeddingWidth: r.spec.OutputHidden, Positions: hunyuanImagePromptPositions,
		Render: func(text []string, items []imagePromptItem) string {
			var prompt strings.Builder
			if !history {
				prompt.WriteString("<｜hy_begin▁of▁sentence｜>")
			}
			for index, item := range items {
				prompt.WriteString(text[index])
				prompt.WriteString(HunyuanVLImageStart)
				prompt.WriteString(strings.Repeat(HunyuanVLImagePad, item.RunCount))
				prompt.WriteString(HunyuanVLImageEnd)
			}
			prompt.WriteString(text[len(text)-1])
			if !history {
				prompt.WriteString("<｜hy_User｜>")
			}
			return prompt.String()
		},
	}
	return executeImagePromptPlan(ctx, tokenizer, sources, text, plan, func(ctx context.Context, source image.Image) (imagePromptItem, error) {
		output, err := r.EncodeImage(ctx, source, DefaultHunyuanVLPreprocessOptions(r.spec))
		if err != nil {
			return imagePromptItem{}, err
		}
		return imagePromptItem{
			Embeddings: output.Embeddings.Data, Count: int(output.Embeddings.Shape.Dims[1]),
			Rows: output.GridH / output.MergeSize, Columns: output.GridW / output.MergeSize,
		}, nil
	})
}

func (r *PaddleOCRRunner) BuildImagePrompt(
	ctx context.Context,
	tokenizer ImageTokenizer,
	source image.Image,
	beforeImage, afterImage string,
	_ bool,
) (MultimodalPrompt, error) {
	return r.BuildImagesPrompt(ctx, tokenizer, []image.Image{source}, []string{beforeImage, afterImage}, false)
}

func (r *PaddleOCRRunner) BuildImagesPrompt(
	ctx context.Context,
	tokenizer ImageTokenizer,
	sources []image.Image,
	text []string,
	_ bool,
) (MultimodalPrompt, error) {
	return r.buildImagesPrompt(ctx, tokenizer, sources, text, false)
}

func (r *PaddleOCRRunner) BuildImagesHistoryPrompt(
	ctx context.Context,
	tokenizer ImageTokenizer,
	sources []image.Image,
	text []string,
) (MultimodalPrompt, error) {
	return r.buildImagesPrompt(ctx, tokenizer, sources, text, true)
}

func (r *PaddleOCRRunner) buildImagesPrompt(
	ctx context.Context,
	tokenizer ImageTokenizer,
	sources []image.Image,
	text []string,
	history bool,
) (MultimodalPrompt, error) {
	plan := imagePromptPlan{
		Family: "PaddleOCR", Placeholder: PaddleOCRImagePad, PlaceholderLabel: "PaddleOCR placeholder",
		History: history, EmbeddingWidth: r.spec.OutputHidden, Positions: qwenImagePromptPositions,
		Render: func(text []string, items []imagePromptItem) string {
			var prompt strings.Builder
			if !history {
				prompt.WriteString("<|begin_of_sentence|>User: ")
			}
			for index, item := range items {
				if history {
					prompt.WriteString(text[index])
				}
				prompt.WriteString("<|IMAGE_START|>")
				prompt.WriteString(strings.Repeat(PaddleOCRImagePad, item.RunCount))
				prompt.WriteString("<|IMAGE_END|>")
			}
			if history {
				prompt.WriteString(text[len(text)-1])
			} else {
				for _, fragment := range text {
					prompt.WriteString(fragment)
				}
				prompt.WriteString("\nAssistant: ")
			}
			return prompt.String()
		},
	}
	return executeImagePromptPlan(ctx, tokenizer, sources, text, plan, func(ctx context.Context, source image.Image) (imagePromptItem, error) {
		output, err := r.EncodeImage(ctx, source, DefaultPaddleOCRPreprocessOptions(r.spec))
		if err != nil {
			return imagePromptItem{}, err
		}
		return imagePromptItem{
			Embeddings: output.Embeddings.Data, Count: int(output.Embeddings.Shape.Dims[1]),
			Rows:    (output.GridH + output.MergeSize - 1) / output.MergeSize,
			Columns: (output.GridW + output.MergeSize - 1) / output.MergeSize,
		}, nil
	})
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

func (r *MiMoVLRunner) BuildImagePrompt(
	ctx context.Context,
	tokenizer ImageTokenizer,
	source image.Image,
	beforeImage, afterImage string,
	_ bool,
) (MultimodalPrompt, error) {
	return r.BuildImagesPrompt(ctx, tokenizer, []image.Image{source}, []string{beforeImage, afterImage}, false)
}

func (r *MiMoVLRunner) BuildImagesPrompt(
	ctx context.Context,
	tokenizer ImageTokenizer,
	sources []image.Image,
	text []string,
	_ bool,
) (MultimodalPrompt, error) {
	return r.buildImagesPrompt(ctx, tokenizer, sources, text, false)
}

func (r *MiMoVLRunner) BuildImagesHistoryPrompt(
	ctx context.Context,
	tokenizer ImageTokenizer,
	sources []image.Image,
	text []string,
) (MultimodalPrompt, error) {
	return r.buildImagesPrompt(ctx, tokenizer, sources, text, true)
}

func (r *MiMoVLRunner) buildImagesPrompt(
	ctx context.Context,
	tokenizer ImageTokenizer,
	sources []image.Image,
	text []string,
	history bool,
) (MultimodalPrompt, error) {
	plan := imagePromptPlan{
		Family: "MiMo-VL", Placeholder: Qwen3VLImagePad, PlaceholderLabel: "MiMo-VL placeholder",
		History: history, EmbeddingWidth: r.spec.ProjectionDim,
		Render: func(text []string, items []imagePromptItem) string {
			var prompt strings.Builder
			if !history {
				prompt.WriteString("<|im_start|>system\n")
				prompt.WriteString(MiMoVLSystemPrompt)
				prompt.WriteString("<|im_end|>\n<|im_start|>user\n")
			}
			for index, item := range items {
				prompt.WriteString(text[index])
				prompt.WriteString("<|vision_start|>")
				prompt.WriteString(strings.Repeat(Qwen3VLImagePad, item.RunCount))
				prompt.WriteString("<|vision_end|>")
			}
			prompt.WriteString(text[len(text)-1])
			if !history {
				prompt.WriteString("<|im_end|>\n<|im_start|>assistant\n")
			}
			return prompt.String()
		},
	}
	return executeImagePromptPlan(ctx, tokenizer, sources, text, plan, func(ctx context.Context, source image.Image) (imagePromptItem, error) {
		output, err := r.EncodeImage(ctx, source)
		if err != nil {
			return imagePromptItem{}, err
		}
		return imagePromptItem{Embeddings: output.Embeddings.Data, Count: int(output.Embeddings.Shape.Dims[1])}, nil
	})
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

func (r *Qwen2VLRunner) BuildImagePrompt(
	ctx context.Context,
	tokenizer ImageTokenizer,
	source image.Image,
	beforeImage, afterImage string,
	_ bool,
) (MultimodalPrompt, error) {
	return r.BuildImagesPrompt(ctx, tokenizer, []image.Image{source}, []string{beforeImage, afterImage}, false)
}

func (r *Qwen2VLRunner) BuildImagesPrompt(
	ctx context.Context,
	tokenizer ImageTokenizer,
	sources []image.Image,
	text []string,
	_ bool,
) (MultimodalPrompt, error) {
	return r.buildImagesPrompt(ctx, tokenizer, sources, text, false)
}

func (r *Qwen2VLRunner) BuildImagesHistoryPrompt(
	ctx context.Context,
	tokenizer ImageTokenizer,
	sources []image.Image,
	text []string,
) (MultimodalPrompt, error) {
	return r.buildImagesPrompt(ctx, tokenizer, sources, text, true)
}

func (r *Qwen2VLRunner) buildImagesPrompt(
	ctx context.Context,
	tokenizer ImageTokenizer,
	sources []image.Image,
	text []string,
	history bool,
) (MultimodalPrompt, error) {
	plan := qwenImagePlan("Qwen2-VL", history, r.spec.OutputHidden, func(text []string, items []imagePromptItem) string {
		return renderQwenImagePrompt(text, items, history, "<|im_end|>\n<|im_start|>assistant\n")
	})
	return executeImagePromptPlan(ctx, tokenizer, sources, text, plan, func(ctx context.Context, source image.Image) (imagePromptItem, error) {
		output, err := r.EncodeImage(ctx, source, DefaultQwen3VLPreprocessOptions())
		if err != nil {
			return imagePromptItem{}, err
		}
		return qwenImagePromptItem(output)
	})
}

func (r *Qwen2VLRunner) BuildVideoPrompt(
	ctx context.Context,
	tokenizer ImageTokenizer,
	frames []image.Image,
	beforeVideo, afterVideo string,
	_ float64,
	_ bool,
) (MultimodalPrompt, error) {
	if tokenizer == nil {
		return MultimodalPrompt{}, errors.New("projector: tokenizer is nil")
	}
	output, err := r.EncodeFrames(ctx, frames, DefaultQwen3VLVideoPreprocessOptions())
	if err != nil {
		return MultimodalPrompt{}, err
	}
	count := int(output.Embeddings.Shape.Dims[1])
	var prompt strings.Builder
	prompt.WriteString("<|im_start|>user\n")
	prompt.WriteString(beforeVideo)
	prompt.WriteString("<|vision_start|>")
	prompt.WriteString(strings.Repeat(Qwen3VLVideoPad, count))
	prompt.WriteString("<|vision_end|>")
	prompt.WriteString(afterVideo)
	prompt.WriteString("<|im_end|>\n<|im_start|>assistant\n")
	ids, err := tokenizer.TokenizeText(prompt.String(), false, true)
	if err != nil {
		return MultimodalPrompt{}, fmt.Errorf("projector: tokenize Qwen2-VL video prompt: %w", err)
	}
	padIDs, err := tokenizer.TokenizeText(Qwen3VLVideoPad, false, true)
	if err != nil {
		return MultimodalPrompt{}, fmt.Errorf("projector: tokenize video placeholder: %w", err)
	}
	if len(padIDs) != 1 {
		return MultimodalPrompt{}, fmt.Errorf("projector: video placeholder maps to %d tokens", len(padIDs))
	}
	start, err := contiguousTokenRun(ids, padIDs[0], count)
	if err != nil {
		return MultimodalPrompt{}, fmt.Errorf("projector: Qwen2-VL video prompt: %w", err)
	}
	rows, columns := output.GridH/output.MergeSize, output.GridW/output.MergeSize
	positions, err := Qwen2VLVideoPositions(len(ids), start, output.GridT, rows, columns)
	if err != nil {
		return MultimodalPrompt{}, err
	}
	return MultimodalPrompt{
		TokenIDs: ids, Embeddings: output.Embeddings.Data, EmbeddingWidth: r.spec.OutputHidden,
		EmbeddingStart: start, EmbeddingTokenIndices: sequentialTokenIndices(start, count),
		MultiAxisPositions: positions,
	}, nil
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
	if err := validateImagePromptInputs(tokenizer, sources, text, "Gemma 4"); err != nil {
		return MultimodalPrompt{}, err
	}
	for _, segment := range text[:len(text)-1] {
		if strings.TrimSpace(segment) != "" {
			return MultimodalPrompt{}, errors.New("projector: Gemma 4 requires images before user text")
		}
	}
	plan := imagePromptPlan{
		Family: "Gemma 4", Placeholder: "<|image|>", PlaceholderLabel: "Gemma 4 image placeholder",
		AttentionBlocks: imagePromptBlocks,
		Render: func(text []string, items []imagePromptItem) string {
			var prompt strings.Builder
			prompt.WriteString("<bos><|turn>user\n")
			for _, item := range items {
				prompt.WriteString("<|image>")
				prompt.WriteString(strings.Repeat("<|image|>", item.RunCount))
				prompt.WriteString("<image|>")
			}
			prompt.WriteString(strings.TrimSpace(text[len(text)-1]))
			prompt.WriteString("<turn|>\n<|turn>model\n<|channel>thought\n<channel|>")
			return prompt.String()
		},
	}
	return executeImagePromptPlan(ctx, tokenizer, sources, text, plan, r.gemma4ImagePromptItem)
}

func (r *Gemma4Runner) BuildImagesHistoryPrompt(
	ctx context.Context,
	tokenizer ImageTokenizer,
	sources []image.Image,
	text []string,
) (MultimodalPrompt, error) {
	plan := imagePromptPlan{
		Family: "Gemma 4 history", Placeholder: "<|image|>", PlaceholderLabel: "Gemma 4 image placeholder",
		History: true, AttentionBlocks: imagePromptBlocks,
		Render: func(text []string, items []imagePromptItem) string {
			var prompt strings.Builder
			for index, item := range items {
				prompt.WriteString(text[index])
				prompt.WriteString("<|image>")
				prompt.WriteString(strings.Repeat("<|image|>", item.RunCount))
				prompt.WriteString("<image|>")
			}
			prompt.WriteString(text[len(text)-1])
			return prompt.String()
		},
	}
	return executeImagePromptPlan(ctx, tokenizer, sources, text, plan, r.gemma4ImagePromptItem)
}

func (r *Gemma4Runner) gemma4ImagePromptItem(ctx context.Context, source image.Image) (imagePromptItem, error) {
	output, err := r.EncodeImage(ctx, source)
	if err != nil {
		return imagePromptItem{}, err
	}
	return imagePromptItem{
		Embeddings: output.Embeddings.Data,
		Count:      int(output.Embeddings.Shape.Dims[1]),
		Width:      int(output.Embeddings.Shape.Dims[0]),
	}, nil
}

func (r *Gemma4Runner) BuildMediaHistoryPrompt(
	ctx context.Context,
	tokenizerAPI ImageTokenizer,
	media []MediaInput,
	text []string,
) (MultimodalPrompt, error) {
	if tokenizerAPI == nil {
		return MultimodalPrompt{}, errors.New("projector: tokenizer is nil")
	}
	if len(media) == 0 || len(text) != len(media)+1 {
		return MultimodalPrompt{}, errors.New("projector: Gemma 4 media history sequence is inconsistent")
	}
	type mediaRun struct {
		token tokenizer.TokenID
		count int
		image bool
	}
	runs := make([]mediaRun, len(media))
	var embeddings []float32
	var prompt strings.Builder
	embeddingWidth := 0
	imagePadIDs, err := tokenizerAPI.TokenizeText("<|image|>", false, true)
	if err != nil {
		return MultimodalPrompt{}, fmt.Errorf("projector: tokenize Gemma 4 image placeholder: %w", err)
	}
	if len(imagePadIDs) != 1 {
		return MultimodalPrompt{}, fmt.Errorf("projector: Gemma 4 image placeholder maps to %d tokens", len(imagePadIDs))
	}
	audioPadIDs, err := tokenizerAPI.TokenizeText("<|audio|>", false, true)
	if err != nil {
		return MultimodalPrompt{}, fmt.Errorf("projector: tokenize Gemma 4 audio placeholder: %w", err)
	}
	if len(audioPadIDs) != 1 {
		return MultimodalPrompt{}, fmt.Errorf("projector: Gemma 4 audio placeholder maps to %d tokens", len(audioPadIDs))
	}
	for index, input := range media {
		prompt.WriteString(text[index])
		switch input.Kind {
		case MediaImage:
			if input.Image == nil {
				return MultimodalPrompt{}, fmt.Errorf("projector: Gemma 4 history image %d is nil", index)
			}
			output, encodeErr := r.EncodeImage(ctx, input.Image)
			if encodeErr != nil {
				return MultimodalPrompt{}, fmt.Errorf("projector: encode Gemma 4 history image %d: %w", index, encodeErr)
			}
			width := int(output.Embeddings.Shape.Dims[0])
			if embeddingWidth != 0 && width != embeddingWidth {
				return MultimodalPrompt{}, errors.New("projector: Gemma 4 media embedding widths differ")
			}
			embeddingWidth = width
			runs[index] = mediaRun{token: imagePadIDs[0], count: int(output.Embeddings.Shape.Dims[1]), image: true}
			prompt.WriteString("<|image>")
			prompt.WriteString(strings.Repeat("<|image|>", runs[index].count))
			prompt.WriteString("<image|>")
			embeddings = append(embeddings, output.Embeddings.Data...)
		case MediaAudio:
			if len(input.Audio) == 0 {
				return MultimodalPrompt{}, fmt.Errorf("projector: Gemma 4 history audio %d is empty", index)
			}
			output, encodeErr := r.EncodeAudio(ctx, input.Audio)
			if encodeErr != nil {
				return MultimodalPrompt{}, fmt.Errorf("projector: encode Gemma 4 history audio %d: %w", index, encodeErr)
			}
			width := int(output.Embeddings.Shape.Dims[0])
			if embeddingWidth != 0 && width != embeddingWidth {
				return MultimodalPrompt{}, errors.New("projector: Gemma 4 media embedding widths differ")
			}
			embeddingWidth = width
			runs[index] = mediaRun{token: audioPadIDs[0], count: int(output.Embeddings.Shape.Dims[1])}
			prompt.WriteString("<|audio>")
			prompt.WriteString(strings.Repeat("<|audio|>", runs[index].count))
			prompt.WriteString("<audio|>")
			embeddings = append(embeddings, output.Embeddings.Data...)
		default:
			return MultimodalPrompt{}, fmt.Errorf("projector: unsupported Gemma 4 media kind %d", input.Kind)
		}
	}
	prompt.WriteString(text[len(text)-1])
	ids, err := tokenizerAPI.TokenizeText(prompt.String(), true, true)
	if err != nil {
		return MultimodalPrompt{}, fmt.Errorf("projector: tokenize Gemma 4 media history: %w", err)
	}
	expectedTokens := make([]tokenizer.TokenID, len(runs))
	counts := make([]int, len(runs))
	for index, run := range runs {
		expectedTokens[index], counts[index] = run.token, run.count
	}
	starts, err := orderedVariableTokenRuns(ids, expectedTokens, counts)
	if err != nil {
		return MultimodalPrompt{}, fmt.Errorf("projector: Gemma 4 media history: %w", err)
	}
	indices := make([]uint32, 0)
	blocks := make([]AttentionBlock, 0, len(runs))
	for index, start := range starts {
		indices = append(indices, sequentialTokenIndices(start, runs[index].count)...)
		if runs[index].image {
			blocks = append(blocks, AttentionBlock{Start: uint32(start), End: uint32(start + runs[index].count)})
		}
	}
	return MultimodalPrompt{
		TokenIDs: ids, Embeddings: embeddings,
		EmbeddingWidth: embeddingWidth, EmbeddingStart: starts[0],
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
	suffix := "<|im_end|>\n<|im_start|>assistant\n<think>\n"
	if !thinking {
		suffix += "\n</think>\n\n"
	}
	plan := qwenImagePlan("Qwen3-VL", false, r.spec.OutputHidden, func(text []string, items []imagePromptItem) string {
		return renderQwenImagePrompt(text, items, false, suffix)
	})
	return executeImagePromptPlan(ctx, tokenizer, sources, text, plan, func(ctx context.Context, source image.Image) (imagePromptItem, error) {
		output, err := r.EncodeImage(ctx, source, DefaultQwen3VLPreprocessOptions())
		if err != nil {
			return imagePromptItem{}, err
		}
		return qwenImagePromptItem(output)
	})
}

func (r *Qwen3VLRunner) BuildImagesHistoryPrompt(
	ctx context.Context,
	tokenizer ImageTokenizer,
	sources []image.Image,
	text []string,
) (MultimodalPrompt, error) {
	plan := qwenImagePlan("Qwen3-VL history", true, r.spec.OutputHidden, func(text []string, items []imagePromptItem) string {
		return renderQwenImagePrompt(text, items, true, "")
	})
	return executeImagePromptPlan(ctx, tokenizer, sources, text, plan, func(ctx context.Context, source image.Image) (imagePromptItem, error) {
		output, err := r.EncodeImage(ctx, source, DefaultQwen3VLPreprocessOptions())
		if err != nil {
			return imagePromptItem{}, err
		}
		return qwenImagePromptItem(output)
	})
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
	var deepstack [][]float32
	if err := appendQwen3VLDeepstack(&deepstack, output); err != nil {
		return Qwen3VLPrompt{}, err
	}
	return Qwen3VLPrompt{
		TokenIDs: ids, Embeddings: output.Embeddings.Data, DeepstackEmbeddings: deepstack,
		EmbeddingWidth:        int(output.Embeddings.Shape.Dims[0]),
		EmbeddingTokenIndices: indices, MultiAxisPositions: positions,
	}, nil
}

func appendQwen3VLDeepstack(target *[][]float32, output Qwen3VLOutput) error {
	if len(output.DeepstackEmbeddings) == 0 {
		if len(*target) != 0 {
			return errors.New("deepstack stream count changed")
		}
		return nil
	}
	if len(*target) == 0 {
		*target = make([][]float32, len(output.DeepstackEmbeddings))
	}
	if len(*target) != len(output.DeepstackEmbeddings) {
		return fmt.Errorf("deepstack streams = %d, want %d", len(output.DeepstackEmbeddings), len(*target))
	}
	for index, stream := range output.DeepstackEmbeddings {
		if !stream.Shape.Equal(output.Embeddings.Shape) {
			return fmt.Errorf("deepstack stream %d shape differs from base embeddings", index)
		}
		(*target)[index] = append((*target)[index], stream.Data...)
	}
	return nil
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

func orderedVariableTokenRuns(
	ids []tokenizer.TokenID,
	tokens []tokenizer.TokenID,
	counts []int,
) ([]int, error) {
	if len(tokens) == 0 || len(tokens) != len(counts) {
		return nil, errors.New("media token run specification is inconsistent")
	}
	mediaTokens := make(map[tokenizer.TokenID]struct{}, len(tokens))
	for _, token := range tokens {
		mediaTokens[token] = struct{}{}
	}
	starts := make([]int, 0, len(tokens))
	foundTokens := make([]tokenizer.TokenID, 0, len(tokens))
	foundCounts := make([]int, 0, len(tokens))
	for index := 0; index < len(ids); {
		if _, ok := mediaTokens[ids[index]]; !ok {
			index++
			continue
		}
		start, token := index, ids[index]
		for index < len(ids) && ids[index] == token {
			index++
		}
		starts = append(starts, start)
		foundTokens = append(foundTokens, token)
		foundCounts = append(foundCounts, index-start)
	}
	if !slices.Equal(foundTokens, tokens) || !slices.Equal(foundCounts, counts) {
		return nil, fmt.Errorf("placeholder runs tokens/counts = %v/%v, want %v/%v", foundTokens, foundCounts, tokens, counts)
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

type HunyuanVLPositionChunk struct {
	Start      int
	Rows       int
	Columns    int
	ImageIndex int
}

func HunyuanVLVariableChunkPositions(tokens int, chunks []HunyuanVLPositionChunk) ([4][]uint32, error) {
	if tokens <= 0 || len(chunks) == 0 {
		return [4][]uint32{}, errors.New("projector: invalid Hunyuan-VL media positions")
	}
	var positions [4][]uint32
	for axis := range positions {
		positions[axis] = make([]uint32, tokens)
	}
	next, physical := uint32(0), 0
	for _, chunk := range chunks {
		count := chunk.Rows*(chunk.Columns+1) + 2
		if chunk.Rows <= 0 || chunk.Columns <= 0 || chunk.ImageIndex < 0 || chunk.Start < physical || chunk.Start+count > tokens {
			return [4][]uint32{}, errors.New("projector: Hunyuan-VL media chunk is invalid")
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
			position := physical + index
			positions[0][position] = base + uint32(index)
			if index == 0 || index == count-1 {
				for axis := 1; axis < 4; axis++ {
					positions[axis][position] = base + uint32(index)
				}
				continue
			}
			offset := index - 1
			positions[1][position] = uint32(offset % (chunk.Columns + 1))
			positions[2][position] = uint32(offset / (chunk.Columns + 1))
			positions[3][position] = uint32(chunk.ImageIndex)
		}
		physical += count
		next = base + uint32(count)
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

func Qwen2VLVideoPositions(tokens, start, temporal, rows, columns int) ([4][]uint32, error) {
	count := temporal * rows * columns
	if tokens <= 0 || start < 0 || temporal <= 0 || rows <= 0 || columns <= 0 || start+count > tokens {
		return [4][]uint32{}, errors.New("projector: invalid Qwen2-VL video position geometry")
	}
	var positions [4][]uint32
	for axis := range positions {
		positions[axis] = make([]uint32, tokens)
	}
	next := uint32(0)
	for index := 0; index < start; index++ {
		for axis := range positions {
			positions[axis][index] = next
		}
		next++
	}
	base := next
	spatial := rows * columns
	for index := 0; index < count; index++ {
		positions[0][start+index] = base + uint32(index/spatial)
		positions[1][start+index] = base + uint32((index%spatial)/columns)
		positions[2][start+index] = base + uint32(index%columns)
		positions[3][start+index] = 0
	}
	next = base + uint32(max(temporal, rows, columns))
	for index := start + count; index < tokens; index++ {
		for axis := range positions {
			positions[axis][index] = next
		}
		next++
	}
	return positions, nil
}
