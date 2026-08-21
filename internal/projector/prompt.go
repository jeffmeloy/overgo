package projector

import (
	"context"
	"errors"
	"fmt"
	"image"
	"math"
	"slices"
	"strings"

	"overgo/internal/artifact"
	"overgo/internal/gguf"
	"overgo/internal/modelartifact"
	"overgo/internal/modelrecipe"
	"overgo/internal/recipe"
	"overgo/internal/tokenizer"
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

type promptRunTokenizer interface {
	TokenizeTextRuns(string, string, []int, bool) ([]tokenizer.TokenID, []int, error)
}

type promptMarkerTokenizer interface {
	TokenizeTextMarkers(string, string, []int, bool) ([]tokenizer.TokenID, []int, error)
}

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

type Projector interface {
	Close() error
}

type PromptOptions struct {
	Thinking bool
	History  bool
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

// NewImageMediaInput: validated image chunk construction.
func NewImageMediaInput(source image.Image) MediaInput {
	return MediaInput{Kind: MediaImage, Image: source}
}

// NewAudioMediaInput: validated audio chunk construction.
func NewAudioMediaInput(samples []float32) MediaInput {
	return MediaInput{Kind: MediaAudio, Audio: samples}
}

func (input MediaInput) validate(family string, index int) error {
	switch input.Kind {
	case MediaImage:
		if input.Image == nil {
			return fmt.Errorf("projector: %s image %d is nil", family, index)
		}
		if len(input.Audio) != 0 {
			return fmt.Errorf("projector: %s image %d contains audio", family, index)
		}
	case MediaAudio:
		if len(input.Audio) == 0 {
			return fmt.Errorf("projector: %s audio %d is empty", family, index)
		}
		if input.Image != nil {
			return fmt.Errorf("projector: %s audio %d contains an image", family, index)
		}
	default:
		return fmt.Errorf("projector: unsupported %s media kind %d", family, input.Kind)
	}
	return nil
}

type OpenOptions struct {
	CUDA                bool
	DeviceOrdinal       int
	DisableDynamicTiles bool
}

func validateImagePromptInputs(
	tokenizer ImageTokenizer,
	sources []image.Image,
	text []string,
	family string,
) error {
	return validatePromptSequence(tokenizer, len(sources), text, family+" image/text")
}

func validatePromptSequence(
	tokenizer ImageTokenizer,
	itemCount int,
	text []string,
	sequenceLabel string,
) error {
	if tokenizer == nil {
		return errors.New("projector: tokenizer is nil")
	}
	if itemCount == 0 || len(text) != itemCount+1 {
		return fmt.Errorf("projector: %s sequence is inconsistent", sequenceLabel)
	}
	return nil
}

func validateMediaHistoryInputs(
	tokenizer ImageTokenizer,
	media []MediaInput,
	text []string,
	family string,
) error {
	if err := validatePromptSequence(tokenizer, len(media), text, family+" media history"); err != nil {
		return err
	}
	for index, input := range media {
		if err := input.validate(family+" history", index); err != nil {
			return err
		}
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
	if optimized, ok := tokenizer.(promptRunTokenizer); ok {
		ids, starts, err := optimized.TokenizeTextRuns(prompt, placeholder, counts, history)
		if err != nil {
			return nil, nil, fmt.Errorf("projector: tokenize %s: %w", promptLabel, err)
		}
		return ids, starts, nil
	}
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
	total := 0
	for _, count := range counts {
		total += count
	}
	indices := make([]uint32, 0, total)
	for index, start := range starts {
		indices = append(indices, sequentialTokenIndices(start+offset, counts[index])...)
	}
	return indices
}

type projectorDescriptor struct {
	kind  string
	media []recipe.DataKind
	open  func(context.Context, *gguf.File, OpenOptions) (Projector, error)
}

func describeProjector[T Projector](kind string, open func(context.Context, *gguf.File, OpenOptions) (T, error), media ...recipe.DataKind) projectorDescriptor {
	return projectorDescriptor{kind: kind, media: slices.Clone(media), open: func(ctx context.Context, file *gguf.File, options OpenOptions) (Projector, error) {
		return open(ctx, file, options)
	}}
}

func describeCatalogProjector[S any, T Projector](
	kind, label string,
	excluded []string,
	read func(*gguf.File) (S, error),
	validate func(*gguf.File, S) ([]string, error),
	build func(*gguf.File, S, *projectorCUDA) T,
	media ...recipe.DataKind,
) projectorDescriptor {
	return describeProjector(kind, func(ctx context.Context, file *gguf.File, options OpenOptions) (T, error) {
		return buildCatalogProjector(ctx, file, options, label, excluded, read, validate, build)
	}, media...)
}

var projectorCatalog = []projectorDescriptor{
	describeProjector(deepSeekOCR2ProjectorType, openDeepSeekOCR2, recipe.DataImage),
	describeProjector(deepSeekOCRProjectorType, openDeepSeekOCR, recipe.DataImage),
	describeCatalogProjector(cogVLMProjectorType, "CogVLM", nil, ReadCogVLMVisionSpec, validateCogVLMVisionCatalog,
		func(file *gguf.File, spec CogVLMVisionSpec, cuda *projectorCUDA) *CogVLMVisionRunner {
			return &CogVLMVisionRunner{projectorResources: projectorResources{file: file, cuda: cuda}, spec: spec, attention: compileVisionAttention(spec.Hidden, spec.Heads)}
		}, recipe.DataImage),
	describeProjector(gemma3nVisionProjectorType, openGemma3nVision, recipe.DataImage),
	describeCatalogProjector(mimoVLProjectorType, "MiMo-VL", nil, ReadMiMoVLSpec, validateMiMoVLCatalog,
		func(file *gguf.File, spec MiMoVLSpec, cuda *projectorCUDA) *MiMoVLRunner {
			return &MiMoVLRunner{projectorResources: projectorResources{file: file, cuda: cuda}, spec: spec}
		}, recipe.DataImage),
	describeCatalogProjector(granite4VisionProjectorType, "Granite 4 Vision", nil, ReadGranite4VisionSpec, validateGranite4VisionCatalog,
		func(file *gguf.File, spec Granite4VisionSpec, cuda *projectorCUDA) *Granite4VisionRunner {
			return &Granite4VisionRunner{projectorResources: projectorResources{file: file, cuda: cuda}, spec: spec, attention: compileVisionAttention(spec.Hidden, spec.Heads)}
		}, recipe.DataImage),
	describeCatalogProjector(llama4ProjectorType, "Llama-4", nil, ReadLlama4VisionSpec, validateLlama4VisionCatalog,
		func(file *gguf.File, spec Llama4VisionSpec, cuda *projectorCUDA) *Llama4VisionRunner {
			return &Llama4VisionRunner{projectorResources: projectorResources{file: file, cuda: cuda}, spec: spec, attention: compileVisionAttention(spec.Hidden, spec.Heads)}
		}, recipe.DataImage),
	describeCatalogProjector(hunyuanVLProjectorType, "Hunyuan-VL", nil, ReadHunyuanVLSpec, validateHunyuanVLCatalog,
		func(file *gguf.File, spec HunyuanVLSpec, cuda *projectorCUDA) *HunyuanVLRunner {
			return &HunyuanVLRunner{projectorResources: projectorResources{file: file, cuda: cuda}, spec: spec, attention: compileVisionAttention(spec.Hidden, spec.Heads)}
		}, recipe.DataImage),
	describeCatalogProjector(paddleOCRProjectorType, "PaddleOCR", nil, ReadPaddleOCRSpec, validatePaddleOCRCatalog,
		func(file *gguf.File, spec PaddleOCRSpec, cuda *projectorCUDA) *PaddleOCRRunner {
			return &PaddleOCRRunner{projectorResources: projectorResources{file: file, cuda: cuda}, spec: spec, attention: compileVisionAttention(spec.Hidden, spec.Heads)}
		}, recipe.DataImage),
	describeCatalogProjector(qwen2VLProjectorType, "Qwen2-VL", nil, ReadQwen2VLSpec, validateQwen2VLCatalog,
		func(file *gguf.File, spec Qwen2VLSpec, cuda *projectorCUDA) *Qwen2VLRunner {
			return &Qwen2VLRunner{projectorResources: projectorResources{file: file, cuda: cuda}, spec: spec}
		}, recipe.DataImage, recipe.DataVideo),
	describeCatalogProjector(qwen3VLProjectorType, "Qwen3-VL", nil, ReadQwen3VLSpec, validateQwen3VLCatalog,
		func(file *gguf.File, spec Qwen3VLSpec, cuda *projectorCUDA) *Qwen3VLRunner {
			return &Qwen3VLRunner{projectorResources: projectorResources{file: file, cuda: cuda}, spec: spec}
		}, recipe.DataImage, recipe.DataVideo),
	describeCatalogProjector(gemma4UVProjectorType, "Gemma 4", []string{"mm.a.input_projection.weight"}, ReadGemma4Spec, validateGemma4Catalog,
		func(file *gguf.File, spec Gemma4Spec, cuda *projectorCUDA) *Gemma4Runner {
			return &Gemma4Runner{projectorResources: projectorResources{file: file, cuda: cuda}, spec: spec}
		}, recipe.DataImage, recipe.DataAudio, recipe.DataVideo),
	describeCatalogProjector(gemma4UAProjectorType, "Gemma 4", []string{"mm.a.input_projection.weight"}, ReadGemma4Spec, validateGemma4Catalog,
		func(file *gguf.File, spec Gemma4Spec, cuda *projectorCUDA) *Gemma4Runner {
			return &Gemma4Runner{projectorResources: projectorResources{file: file, cuda: cuda}, spec: spec}
		}, recipe.DataImage, recipe.DataAudio, recipe.DataVideo),
	describeCatalogProjector(gemma4VisionTowerProjectorType, "Gemma 4 tower", nil, ReadGemma4TowerSpec, validateGemma4TowerCatalog,
		func(file *gguf.File, spec Gemma4TowerSpec, cuda *projectorCUDA) *Gemma4TowerRunner {
			return &Gemma4TowerRunner{projectorResources: projectorResources{file: file, cuda: cuda}, spec: spec, audioPlan: newGemma4AudioFrontendPlan(spec.Audio)}
		}),
}

func OpenAs[T Projector](ctx context.Context, path string, options OpenOptions) (T, error) {
	return openAs[T](ctx, path, options, nil)
}

// OpenActiveAs: exact artifact and module admission before projector use.
func OpenActiveAs[T Projector](
	ctx context.Context,
	store artifact.Reader,
	modelID artifact.ID,
	path string,
	options OpenOptions,
) (T, error) {
	return openAs[T](ctx, path, options, func(file *gguf.File, descriptor projectorDescriptor) error {
		inventory, err := modelartifact.FromGGUF(file, artifact.KindProjector)
		if err != nil {
			return err
		}
		_, program, err := modelrecipe.ResolveActiveCapability(ctx, store, modelID, recipe.TaskProjection)
		if err != nil {
			return err
		}
		definition := program.Definition()
		bound, ok := definition.PrimaryDependency(recipe.DependencyProjector)
		if !ok || bound != inventory.Manifest.ID {
			return errors.New("projector: loaded artifact differs from active projection recipe")
		}
		expected, err := modelrecipe.ProjectionDefinition(
			modelID, inventory.Manifest.ID, descriptor.media...,
		)
		if err != nil {
			return err
		}
		if definition.ID != expected.ID {
			return errors.New("projector: active projection recipe differs from artifact capabilities")
		}
		return nil
	})
}

// InspectProjection: validated inventory and media contract.
func InspectProjection(ctx context.Context, path string) (modelartifact.Inventory, []recipe.DataKind, error) {
	if err := ctx.Err(); err != nil {
		return modelartifact.Inventory{}, nil, err
	}
	file, err := gguf.Open(path)
	if err != nil {
		return modelartifact.Inventory{}, nil, err
	}
	descriptor, descriptorErr := resolveProjectorDescriptor(file)
	inventory, inventoryErr := modelartifact.FromGGUF(file, artifact.KindProjector)
	if err := errors.Join(descriptorErr, inventoryErr, file.Close()); err != nil {
		return modelartifact.Inventory{}, nil, err
	}
	return inventory, slices.Clone(descriptor.media), nil
}

func openAs[T Projector](
	ctx context.Context,
	path string,
	options OpenOptions,
	admit func(*gguf.File, projectorDescriptor) error,
) (T, error) {
	var zero T
	selected, err := openProjectorResource(ctx, path, func(file *gguf.File) (Projector, error) {
		descriptor, err := resolveProjectorDescriptor(file)
		if err != nil {
			return nil, err
		}
		if admit != nil {
			if err := admit(file, descriptor); err != nil {
				return nil, err
			}
		}
		return descriptor.open(ctx, file, options)
	})
	if err != nil {
		return zero, err
	}
	projector, ok := selected.(T)
	if !ok {
		_ = selected.Close()
		return zero, errors.New("projector: selected artifact does not implement the requested contract")
	}
	return projector, nil
}

func resolveProjectorDescriptor(file *gguf.File) (projectorDescriptor, error) {
	projectorType := ""
	for _, key := range []string{"clip.projector_type", "clip.vision.projector_type", "clip.audio.projector_type"} {
		if value, ok := file.MetadataValue(key); ok && value.Type == gguf.ValueTypeString {
			projectorType, _ = value.Data.(string)
			if projectorType != "" {
				break
			}
		}
	}
	for _, descriptor := range projectorCatalog {
		if descriptor.kind == projectorType {
			return descriptor, nil
		}
	}
	return projectorDescriptor{}, fmt.Errorf("projector: artifact projector type %q is unsupported", projectorType)
}

func (r *Granite4VisionRunner) imagesPrompt(
	ctx context.Context,
	tokenizer ImageTokenizer,
	sources []image.Image,
	text []string,
	options PromptOptions,
) (MultimodalPrompt, error) {
	history := options.History
	plan := imagePromptPlan{
		Family: "Granite 4 Vision", Placeholder: Granite4VisionImageToken,
		PlaceholderLabel: "Granite 4 Vision image token", AddSpecial: history,
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

func (r *Llama4VisionRunner) imagesPrompt(
	ctx context.Context,
	tokenizer ImageTokenizer,
	sources []image.Image,
	text []string,
	options PromptOptions,
) (MultimodalPrompt, error) {
	history := options.History
	plan := imagePromptPlan{
		Family: "Llama-4", Placeholder: Llama4ImagePad, PlaceholderLabel: "Llama-4 placeholder",
		AddSpecial: history, EmbeddingWidth: r.spec.OutputHidden,
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

func (r *HunyuanVLRunner) imagesPrompt(
	ctx context.Context,
	tokenizer ImageTokenizer,
	sources []image.Image,
	text []string,
	options PromptOptions,
) (MultimodalPrompt, error) {
	history := options.History
	plan := imagePromptPlan{
		Family: "Hunyuan-VL", Placeholder: HunyuanVLImagePad, PlaceholderLabel: "Hunyuan-VL placeholder",
		AddSpecial: history, EmbeddingWidth: r.spec.OutputHidden, Positions: hunyuanImagePromptPositions,
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
		output, err := r.EncodeImage(ctx, source, RasterPatchOptions{})
		if err != nil {
			return imagePromptItem{}, err
		}
		return imagePromptItem{
			Embeddings: output.Embeddings.Data, Count: int(output.Embeddings.Shape.Dims[1]),
			Rows: output.GridH / output.MergeSize, Columns: output.GridW / output.MergeSize,
		}, nil
	})
}

func (r *PaddleOCRRunner) imagesPrompt(
	ctx context.Context,
	tokenizer ImageTokenizer,
	sources []image.Image,
	text []string,
	options PromptOptions,
) (MultimodalPrompt, error) {
	history := options.History
	plan := imagePromptPlan{
		Family: "PaddleOCR", Placeholder: PaddleOCRImagePad, PlaceholderLabel: "PaddleOCR placeholder",
		AddSpecial: history, EmbeddingWidth: r.spec.OutputHidden, Positions: qwenImagePromptPositions,
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
		output, err := r.EncodeImage(ctx, source, RasterPatchOptions{})
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

func (r *MiMoVLRunner) imagesPrompt(
	ctx context.Context,
	tokenizer ImageTokenizer,
	sources []image.Image,
	text []string,
	options PromptOptions,
) (MultimodalPrompt, error) {
	history := options.History
	plan := imagePromptPlan{
		Family: "MiMo-VL", Placeholder: Qwen3VLImagePad, PlaceholderLabel: "MiMo-VL placeholder",
		AddSpecial: history, EmbeddingWidth: r.spec.ProjectionDim,
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

func (r *Qwen2VLRunner) imagesPrompt(
	ctx context.Context,
	tokenizer ImageTokenizer,
	sources []image.Image,
	text []string,
	options PromptOptions,
) (MultimodalPrompt, error) {
	history := options.History
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

func (r *Qwen2VLRunner) videoPrompt(
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
	rows, columns := output.GridH/output.MergeSize, output.GridW/output.MergeSize
	return executeProjectedPromptPlan(tokenizer, projectedPromptPlan{
		mediaPromptRunPlan: mediaPromptRunPlan{
			Prompt: "<|im_start|>user\n" + beforeVideo + "<|vision_start|>" +
				strings.Repeat(Qwen3VLVideoPad, count) + "<|vision_end|>" + afterVideo +
				"<|im_end|>\n<|im_start|>assistant\n",
			Placeholder: Qwen3VLVideoPad, Runs: 1, TokensPerRun: count,
			PromptLabel: "Qwen2-VL video prompt", PlaceholderLabel: "video placeholder", RunsLabel: "Qwen2-VL video prompt",
		},
		Embeddings: output.Embeddings.Data, EmbeddingWidth: r.spec.OutputHidden,
		Positions: func(tokenCount int, starts []int) ([4][]uint32, error) {
			return Qwen2VLVideoPositions(tokenCount, starts[0], output.GridT, rows, columns)
		},
	})
}

func (r *Gemma4Runner) imagesPrompt(
	ctx context.Context,
	tokenizer ImageTokenizer,
	sources []image.Image,
	text []string,
	options PromptOptions,
) (MultimodalPrompt, error) {
	if err := validateImagePromptInputs(tokenizer, sources, text, "Gemma 4"); err != nil {
		return MultimodalPrompt{}, err
	}
	if !options.History {
		for _, segment := range text[:len(text)-1] {
			if strings.TrimSpace(segment) != "" {
				return MultimodalPrompt{}, errors.New("projector: Gemma 4 requires images before user text")
			}
		}
	}
	plan := imagePromptPlan{
		Family: "Gemma 4", Placeholder: "<|image|>", PlaceholderLabel: "Gemma 4 image placeholder",
		AddSpecial: options.History, AttentionBlocks: imagePromptBlocks,
		Render: func(text []string, items []imagePromptItem) string {
			var prompt strings.Builder
			if !options.History {
				prompt.WriteString("<bos><|turn>user\n")
			}
			for index, item := range items {
				if options.History {
					prompt.WriteString(text[index])
				}
				prompt.WriteString("<|image>")
				prompt.WriteString(strings.Repeat("<|image|>", item.RunCount))
				prompt.WriteString("<image|>")
			}
			if options.History {
				prompt.WriteString(text[len(text)-1])
			} else {
				prompt.WriteString(strings.TrimSpace(text[len(text)-1]))
				prompt.WriteString("<turn|>\n<|turn>model\n<|channel>thought\n<channel|>")
			}
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

func (r *Gemma4Runner) mediaHistoryPrompt(
	ctx context.Context,
	tokenizerAPI ImageTokenizer,
	media []MediaInput,
	text []string,
) (MultimodalPrompt, error) {
	plan := mixedMediaPromptPlan{
		Family: "Gemma 4", AddSpecial: true, PromptLabel: "Gemma 4 media history",
		Render: renderMixedMediaHistory,
		Kinds: map[MediaKind]mixedMediaKindPlan{
			MediaImage: {
				Placeholder: "<|image|>", PlaceholderLabel: "Gemma 4 image placeholder",
				Open: "<|image>", Close: "<image|>", Attention: true,
				Encode: func(ctx context.Context, input MediaInput) (imagePromptItem, error) {
					output, err := r.EncodeImage(ctx, input.Image)
					if err != nil {
						return imagePromptItem{}, err
					}
					return imagePromptItem{Embeddings: output.Embeddings.Data, Count: int(output.Embeddings.Shape.Dims[1]), Width: int(output.Embeddings.Shape.Dims[0])}, nil
				},
			},
			MediaAudio: {
				Placeholder: "<|audio|>", PlaceholderLabel: "Gemma 4 audio placeholder",
				Open: "<|audio>", Close: "<audio|>",
				Encode: func(ctx context.Context, input MediaInput) (imagePromptItem, error) {
					output, err := r.EncodeAudio(ctx, input.Audio)
					if err != nil {
						return imagePromptItem{}, err
					}
					return imagePromptItem{Embeddings: output.Embeddings.Data, Count: int(output.Embeddings.Shape.Dims[1]), Width: int(output.Embeddings.Shape.Dims[0])}, nil
				},
			},
		},
	}
	return executeMixedMediaPromptPlan(ctx, tokenizerAPI, media, text, plan)
}

func (r *Gemma4Runner) audioPrompt(
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
	return executeProjectedPromptPlan(tokenizer, projectedPromptPlan{
		mediaPromptRunPlan: mediaPromptRunPlan{
			Prompt: Gemma4AudioPromptText(afterAudio, audioTokens), Placeholder: "<|audio|>",
			Runs: 1, TokensPerRun: audioTokens,
			PromptLabel: "Gemma 4 audio prompt", PlaceholderLabel: "Gemma 4 audio placeholder", RunsLabel: "Gemma 4 audio prompt",
		},
		Embeddings: output.Embeddings.Data, EmbeddingWidth: int(output.Embeddings.Shape.Dims[0]),
	})
}

func Gemma4AudioPromptText(question string, audioTokens int) string {
	return "<bos><|turn>user\n<|audio>" + strings.Repeat("<|audio|>", audioTokens) +
		"<audio|>" + strings.TrimSpace(question) +
		"<turn|>\n<|turn>model\n<|channel>thought\n<channel|>"
}

func (r *Gemma4Runner) videoPrompt(
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
	return executeProjectedPromptPlan(tokenizer, projectedPromptPlan{
		mediaPromptRunPlan: mediaPromptRunPlan{
			Prompt: Gemma4VideoPromptText(afterVideo, output.Frames, output.TokensPerFrame, fps), Placeholder: "<|video|>",
			Runs: output.Frames, TokensPerRun: output.TokensPerFrame,
			PromptLabel: "Gemma 4 video prompt", PlaceholderLabel: "Gemma 4 video placeholder", RunsLabel: "Gemma 4 video prompt",
		},
		Embeddings: output.Embeddings.Data, EmbeddingWidth: int(output.Embeddings.Shape.Dims[0]),
		AttentionBlocks: mediaPromptAttentionBlocks,
	})
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

func (r *Qwen3VLRunner) imagesPrompt(
	ctx context.Context,
	tokenizer ImageTokenizer,
	sources []image.Image,
	text []string,
	options PromptOptions,
) (MultimodalPrompt, error) {
	suffix := ""
	if !options.History {
		suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n"
		if !options.Thinking {
			suffix += "\n</think>\n\n"
		}
	}
	plan := qwenImagePlan("Qwen3-VL", options.History, r.spec.OutputHidden, func(text []string, items []imagePromptItem) string {
		return renderQwenImagePrompt(text, items, options.History, suffix)
	})
	return executeImagePromptPlan(ctx, tokenizer, sources, text, plan, func(ctx context.Context, source image.Image) (imagePromptItem, error) {
		output, err := r.EncodeImage(ctx, source, DefaultQwen3VLPreprocessOptions())
		if err != nil {
			return imagePromptItem{}, err
		}
		return qwenImagePromptItem(output)
	})
}

func (r *Qwen3VLRunner) videoPrompt(
	ctx context.Context,
	tokenizer ImageTokenizer,
	frames []image.Image,
	beforeVideo, afterVideo string,
	fps float64,
	thinking bool,
) (MultimodalPrompt, error) {
	if tokenizer == nil {
		return MultimodalPrompt{}, errors.New("projector: tokenizer is nil")
	}
	if fps <= 0 || math.IsNaN(fps) || math.IsInf(fps, 0) {
		return MultimodalPrompt{}, errors.New("projector: video FPS must be positive and finite")
	}
	output, err := r.EncodeFrames(ctx, frames, DefaultQwen3VLVideoPreprocessOptions())
	if err != nil {
		return MultimodalPrompt{}, err
	}
	rows, columns := output.GridH/output.MergeSize, output.GridW/output.MergeSize
	perGroup := rows * columns
	item, err := qwenImagePromptItem(output)
	if err != nil {
		return MultimodalPrompt{}, err
	}
	return executeProjectedPromptPlan(tokenizer, projectedPromptPlan{
		mediaPromptRunPlan: mediaPromptRunPlan{
			Prompt:      Qwen35VideoPromptText(beforeVideo, afterVideo, output.GridT, perGroup, fps, thinking),
			Placeholder: Qwen3VLVideoPad, Runs: output.GridT, TokensPerRun: perGroup,
			PromptLabel: "video prompt", PlaceholderLabel: "video placeholder", RunsLabel: "video prompt",
		},
		Embeddings: output.Embeddings.Data, Deepstack: item.Deepstack,
		EmbeddingWidth: int(output.Embeddings.Shape.Dims[0]),
		Positions: func(tokenCount int, starts []int) ([4][]uint32, error) {
			return Qwen3VLMultiChunkPositions(tokenCount, starts, perGroup, rows, columns)
		},
	})
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
