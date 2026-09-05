package projector

import (
	"context"
	"errors"
	"fmt"
	"image"
	"slices"
	"strings"
	"time"

	"overgo/internal/artifact"
	"overgo/internal/checked"
	"overgo/internal/gguf"
	"overgo/internal/media"
	"overgo/internal/modelartifact"
	"overgo/internal/modelrecipe"
	"overgo/internal/recipe"
	"overgo/internal/tensor"
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
	MultiAxisPositions    [tensor.MaxDimensions][]uint32
	AttentionBlocks       []AttentionBlock
	VisualBlocks          []AttentionBlock
}

type AttentionBlock struct {
	Start uint32
	End   uint32
}

type Projector interface {
	Close() error
	setPrompt(promptDispatch)
	compiledPrompt() promptDispatch
}

type PromptOptions struct {
	Thinking bool
	History  bool
}

type MediaKind uint8

const (
	MediaImage MediaKind = iota + tensor.SingletonExtent
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
		if len(input.Audio) != tensor.FirstOffset {
			return fmt.Errorf("projector: %s image %d contains audio", family, index)
		}
	case MediaAudio:
		if len(input.Audio) == tensor.FirstOffset {
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
	MediaPreprocess     *MediaPreprocessProfile
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
	if itemCount == tensor.FirstOffset || len(text) != itemCount+tensor.SingletonExtent {
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
	if len(placeholderIDs) != tensor.SingletonExtent {
		return nil, nil, fmt.Errorf("projector: %s maps to %d tokens", placeholderLabel, len(placeholderIDs))
	}
	starts, err := variableTokenRuns(ids, placeholderIDs[tensor.FirstOffset], counts)
	if err != nil {
		return nil, nil, fmt.Errorf("projector: %s: %w", runsLabel, err)
	}
	return ids, starts, nil
}

func embeddingTokenIndices(starts, counts []int, offset int) []uint32 {
	total := tensor.FirstOffset
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
	kind   string
	media  []recipe.DataKind
	open   func(context.Context, *gguf.File, OpenOptions) (Projector, error)
	prompt func(Projector) promptDispatch
}

func describeProjector[T Projector](
	kind string,
	open func(context.Context, *gguf.File, OpenOptions) (T, error),
	prompt func(T) promptDispatch,
	media ...recipe.DataKind,
) projectorDescriptor {
	descriptor := projectorDescriptor{kind: kind, media: slices.Clone(media), open: func(ctx context.Context, file *gguf.File, options OpenOptions) (Projector, error) {
		opened, err := open(ctx, file, options)
		if err == nil && options.MediaPreprocess != nil {
			if target, ok := any(opened).(interface{ setMediaPreprocess(MediaPreprocessProfile) }); ok {
				target.setMediaPreprocess(*options.MediaPreprocess)
			}
		}
		if err == nil && prompt != nil {
			opened.setPrompt(prompt(opened))
		}
		return opened, err
	}}
	if prompt != nil {
		descriptor.prompt = func(source Projector) promptDispatch { return prompt(source.(T)) }
	}
	return descriptor
}

func describeCatalogProjector[S any, T Projector](
	kind, label string,
	excluded []string,
	read func(*gguf.File) (S, error),
	validate func(*gguf.File, S) ([]string, error),
	build func(*gguf.File, S, *projectorCUDA) T,
	prompt func(T) promptDispatch,
	media ...recipe.DataKind,
) projectorDescriptor {
	return describeProjector(kind, func(ctx context.Context, file *gguf.File, options OpenOptions) (T, error) {
		return buildCatalogProjector(ctx, file, options, label, excluded, read, validate, build)
	}, prompt, media...)
}

func promptPrograms[T Projector](
	imageProgram func(T) compiledImagePromptProgram,
	mediaProgram func(T) compiledMediaPromptProgram,
) func(T) promptDispatch {
	return func(source T) promptDispatch {
		dispatch := promptDispatch{}
		if imageProgram != nil {
			dispatch.images = imageProgram(source).execute
		}
		if mediaProgram != nil {
			program := mediaProgram(source)
			dispatch.video = program.Video
			dispatch.audio = program.Audio
			dispatch.audioSampleRate = program.AudioSampleRate
			dispatch.media = program.History
		}
		return dispatch
	}
}

var projectorCatalog = []projectorDescriptor{
	describeProjector(deepSeekOCR2ProjectorType, openDeepSeekOCR2, promptPrograms(func(r *DeepSeekOCR2Runner) compiledImagePromptProgram {
		return compileDelimitedImagePromptProgram("DeepSeek-OCR-2", DeepSeekOCRImagePad, r.spec.OutputHidden, referenceImageEncoder(r.EncodeImage))
	}, nil), recipe.DataImage),
	describeProjector(deepSeekOCRProjectorType, openDeepSeekOCR, promptPrograms(func(r *DeepSeekOCRRunner) compiledImagePromptProgram {
		return compileDelimitedImagePromptProgram("DeepSeek-OCR", DeepSeekOCRImagePad, r.spec.OutputHidden, referenceImageEncoder(r.EncodeImage))
	}, nil), recipe.DataImage),
	describeCatalogProjector(cogVLMProjectorType, "CogVLM", nil, ReadCogVLMVisionSpec, validateCogVLMVisionCatalog,
		func(file *gguf.File, spec CogVLMVisionSpec, cuda *projectorCUDA) *CogVLMVisionRunner {
			return &CogVLMVisionRunner{projectorResources: projectorResources{file: file, cuda: cuda}, spec: spec, attention: compileVisionAttention(spec.Hidden, spec.Heads)}
		}, promptPrograms(compileCogVLMImagePrompt, nil), recipe.DataImage),
	describeProjector(gemma3nVisionProjectorType, openGemma3nVision, promptPrograms(compileGemma3nVisionImagePrompt, nil), recipe.DataImage),
	describeCatalogProjector(mimoVLProjectorType, "MiMo-VL", nil, ReadMiMoVLSpec, validateMiMoVLCatalog,
		func(file *gguf.File, spec MiMoVLSpec, cuda *projectorCUDA) *MiMoVLRunner {
			return &MiMoVLRunner{projectorResources: projectorResources{file: file, cuda: cuda}, spec: spec}
		}, promptPrograms(compileMiMoVLImagePrompt, nil), recipe.DataImage),
	describeCatalogProjector(granite4VisionProjectorType, "Granite 4 Vision", nil, ReadGranite4VisionSpec, validateGranite4VisionCatalog,
		func(file *gguf.File, spec Granite4VisionSpec, cuda *projectorCUDA) *Granite4VisionRunner {
			return &Granite4VisionRunner{projectorResources: projectorResources{file: file, cuda: cuda}, spec: spec, attention: compileVisionAttention(spec.Hidden, spec.Heads)}
		}, promptPrograms(compileGranite4VisionImagePrompt, nil), recipe.DataImage),
	describeCatalogProjector(llama4ProjectorType, "Llama-4", nil, ReadLlama4VisionSpec, validateLlama4VisionCatalog,
		func(file *gguf.File, spec Llama4VisionSpec, cuda *projectorCUDA) *Llama4VisionRunner {
			return &Llama4VisionRunner{projectorResources: projectorResources{file: file, cuda: cuda}, spec: spec, attention: compileVisionAttention(spec.Hidden, spec.Heads)}
		}, promptPrograms(compileLlama4VisionImagePrompt, nil), recipe.DataImage),
	describeCatalogProjector(hunyuanVLProjectorType, "Hunyuan-VL", nil, ReadHunyuanVLSpec, validateHunyuanVLCatalog,
		func(file *gguf.File, spec HunyuanVLSpec, cuda *projectorCUDA) *HunyuanVLRunner {
			runner := &HunyuanVLRunner{projectorResources: projectorResources{file: file, cuda: cuda}, spec: spec, attention: compileVisionAttention(spec.Hidden, spec.Heads)}
			runner.rasterPatchEncoder = rasterPatchEncoder{
				resources: &runner.projectorResources,
				plan:      spec.visionBackboneSpec.rasterPlan(spec.MergeSize, spec.MinPixels, spec.MaxPixels, rasterBicubic),
				execute:   runner.encodeGraph,
			}
			return runner
		}, promptPrograms(compileHunyuanVLImagePrompt, nil), recipe.DataImage),
	describeCatalogProjector(paddleOCRProjectorType, "PaddleOCR", nil, ReadPaddleOCRSpec, validatePaddleOCRCatalog,
		func(file *gguf.File, spec PaddleOCRSpec, cuda *projectorCUDA) *PaddleOCRRunner {
			runner := &PaddleOCRRunner{projectorResources: projectorResources{file: file, cuda: cuda}, spec: spec, attention: compileVisionAttention(spec.Hidden, spec.Heads)}
			runner.rasterPatchEncoder = rasterPatchEncoder{
				resources: &runner.projectorResources,
				plan:      spec.visionBackboneSpec.rasterPlan(spec.MergeSize, spec.MinPixels, spec.MaxPixels, rasterBilinear),
				execute:   runner.encodeGraph,
			}
			return runner
		}, promptPrograms(compilePaddleOCRImagePrompt, nil), recipe.DataImage),
	describeCatalogProjector(qwen2VLProjectorType, "Qwen2-VL", nil, ReadQwen2VLSpec, validateQwen2VLCatalog,
		func(file *gguf.File, spec Qwen2VLSpec, cuda *projectorCUDA) *Qwen2VLRunner {
			return &Qwen2VLRunner{projectorResources: projectorResources{file: file, cuda: cuda}, spec: spec}
		}, promptPrograms(compileQwen2VLImagePrompt, compileQwen2VLMediaPrompt), recipe.DataImage, recipe.DataVideo),
	describeCatalogProjector(qwen3VLProjectorType, "Qwen3-VL", nil, ReadQwen3VLSpec, validateQwen3VLCatalog,
		func(file *gguf.File, spec Qwen3VLSpec, cuda *projectorCUDA) *Qwen3VLRunner {
			return &Qwen3VLRunner{projectorResources: projectorResources{file: file, cuda: cuda}, spec: spec}
		}, promptPrograms(compileQwen3VLImagePrompt, compileQwen3VLMediaPrompt), recipe.DataImage, recipe.DataVideo),
	describeCatalogProjector(gemma4UVProjectorType, "Gemma 4", []string{"mm.a.input_projection.weight"}, ReadGemma4Spec, validateGemma4Catalog,
		func(file *gguf.File, spec Gemma4Spec, cuda *projectorCUDA) *Gemma4Runner {
			return &Gemma4Runner{projectorResources: projectorResources{file: file, cuda: cuda}, spec: spec}
		}, promptPrograms(compileGemma4ImagePrompt, compileGemma4MediaPrompt), recipe.DataImage, recipe.DataAudio, recipe.DataVideo),
	describeCatalogProjector(gemma4UAProjectorType, "Gemma 4", []string{"mm.a.input_projection.weight"}, ReadGemma4Spec, validateGemma4Catalog,
		func(file *gguf.File, spec Gemma4Spec, cuda *projectorCUDA) *Gemma4Runner {
			return &Gemma4Runner{projectorResources: projectorResources{file: file, cuda: cuda}, spec: spec}
		}, promptPrograms(compileGemma4ImagePrompt, compileGemma4MediaPrompt), recipe.DataImage, recipe.DataAudio, recipe.DataVideo),
	describeCatalogProjector(gemma4VisionTowerProjectorType, "Gemma 4 tower", nil, ReadGemma4TowerSpec, validateGemma4TowerCatalog,
		func(file *gguf.File, spec Gemma4TowerSpec, cuda *projectorCUDA) *Gemma4TowerRunner {
			return &Gemma4TowerRunner{projectorResources: projectorResources{file: file, cuda: cuda}, spec: spec, audioPlan: newAudioFrontendPlan(spec.Audio)}
		}, promptPrograms(compileGemma4TowerImagePrompt, compileGemma4TowerMediaPrompt), recipe.DataImage, recipe.DataAudio, recipe.DataVideo),
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
	return openAs[T](ctx, path, options, func(file *gguf.File, descriptor projectorDescriptor) (*MediaPreprocessProfile, error) {
		inventory, err := modelartifact.FromGGUF(file, artifact.KindProjector)
		if err != nil {
			return nil, err
		}
		_, program, err := modelrecipe.ResolveActiveCapability(ctx, store, modelID, recipe.TaskProjection)
		if err != nil {
			return nil, err
		}
		definition := program.Definition()
		bound, ok := definition.PrimaryDependency(recipe.DependencyProjector)
		if !ok || bound != inventory.Manifest.ID {
			return nil, errors.New("projector: loaded artifact differs from active projection recipe")
		}
		declared, hasDeclared, err := catalogMediaPreprocessProfile(descriptor.kind)
		if err != nil {
			return nil, err
		}
		var processorID artifact.ID
		var resolved *MediaPreprocessProfile
		if hasDeclared {
			profile, err := modelrecipe.ResolveProfileDependency(
				ctx, store, definition, recipe.DependencyProcessorProfile, mediaPreprocessProfileCodec,
			)
			if err != nil {
				return nil, err
			}
			if profile.ID != declared.ID {
				return nil, errors.New("projector: active processor profile differs from artifact declaration")
			}
			processorID, resolved = profile.ID, &profile
		} else if _, bound := definition.PrimaryDependency(recipe.DependencyProcessorProfile); bound {
			return nil, errors.New("projector: processor profile bound to unsupported artifact")
		}
		expected, err := modelrecipe.ProjectionDefinition(
			modelID, inventory.Manifest.ID, processorID, descriptor.media...,
		)
		if err != nil {
			return nil, err
		}
		if definition.ID != expected.ID {
			return nil, errors.New("projector: active projection recipe differs from artifact capabilities")
		}
		return resolved, nil
	})
}

// InspectProjection: validated inventory and media contract.
func InspectProjection(ctx context.Context, path string) (modelartifact.Inventory, []recipe.DataKind, *MediaPreprocessProfile, error) {
	if err := ctx.Err(); err != nil {
		return modelartifact.Inventory{}, nil, nil, err
	}
	file, err := gguf.Open(path)
	if err != nil {
		return modelartifact.Inventory{}, nil, nil, err
	}
	descriptor, descriptorErr := resolveProjectorDescriptor(file)
	inventory, inventoryErr := modelartifact.FromGGUF(file, artifact.KindProjector)
	profile, found, profileErr := catalogMediaPreprocessProfile(descriptor.kind)
	if err := errors.Join(descriptorErr, inventoryErr, profileErr, file.Close()); err != nil {
		return modelartifact.Inventory{}, nil, nil, err
	}
	if !found {
		return inventory, slices.Clone(descriptor.media), nil, nil
	}
	return inventory, slices.Clone(descriptor.media), &profile, nil
}

func openAs[T Projector](
	ctx context.Context,
	path string,
	options OpenOptions,
	admit func(*gguf.File, projectorDescriptor) (*MediaPreprocessProfile, error),
) (T, error) {
	var zero T
	selected, err := openProjectorResource(ctx, path, func(file *gguf.File) (Projector, error) {
		descriptor, err := resolveProjectorDescriptor(file)
		if err != nil {
			return nil, err
		}
		if admit != nil {
			profile, err := admit(file, descriptor)
			if err != nil {
				return nil, err
			}
			options.MediaPreprocess = profile
		}
		if options.MediaPreprocess == nil {
			profile, found, err := catalogMediaPreprocessProfile(descriptor.kind)
			if err != nil {
				return nil, err
			}
			if found {
				options.MediaPreprocess = &profile
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
	for _, key := range []string{visionProjectorTypeKey, visionTowerTypeKey, audioProjectorTypeKey} {
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

func compileGranite4VisionImagePrompt(r *Granite4VisionRunner) compiledImagePromptProgram {
	encode := func(ctx context.Context, source image.Image) (imagePromptItem, error) {
		output, err := r.EncodeImage(ctx, source)
		if err != nil {
			return imagePromptItem{}, err
		}
		deepstack := make([][]float32, len(output.DeepstackEmbeddings))
		for index := range output.DeepstackEmbeddings {
			deepstack[index] = output.DeepstackEmbeddings[index].Data
		}
		count := int(output.Embeddings.Shape.Dims[tensor.SingletonExtent])
		return imagePromptItem{
			Embeddings: output.Embeddings.Data, Deepstack: deepstack, Count: count, RunCount: count + tensor.SingletonExtent,
		}, nil
	}
	return compileFramedImagePromptProgram(imagePromptPlan{
		Family: "Granite 4 Vision", Placeholder: Granite4VisionImageToken,
		PlaceholderLabel: "Granite 4 Vision image token", EmbeddingWidth: r.spec.ProjectionDim,
		EmbeddingOffset: tensor.SingletonExtent,
	}, promptDelimiters{
		Prefix: "<|start_of_role|>user<|end_of_role|>\n",
		Suffix: "<|end_of_text|>\n<|start_of_role|>assistant<|end_of_role|>\n",
	}, promptDelimiters{}, encode)
}

func compileLlama4VisionImagePrompt(r *Llama4VisionRunner) compiledImagePromptProgram {
	encode := func(ctx context.Context, source image.Image) (imagePromptItem, error) {
		output, err := r.EncodeImage(ctx, source)
		if err != nil {
			return imagePromptItem{}, err
		}
		return imagePromptItem{Embeddings: output.Embeddings.Data, Count: int(output.Embeddings.Shape.Dims[tensor.SingletonExtent])}, nil
	}
	return compileFramedImagePromptProgram(imagePromptPlan{
		Family: "Llama-4", Placeholder: Llama4ImagePad, PlaceholderLabel: "Llama-4 placeholder",
		EmbeddingWidth: r.spec.OutputHidden,
	}, promptDelimiters{
		Prefix: "<|begin_of_text|><|header_start|>user<|header_end|>\n\n",
		Suffix: "<|eot|><|header_start|>assistant<|header_end|>\n\n",
	}, promptDelimiters{Prefix: Llama4ImageStart, Suffix: Llama4ImageEnd}, encode)
}

func compileHunyuanVLImagePrompt(r *HunyuanVLRunner) compiledImagePromptProgram {
	return compileFramedImagePromptProgram(imagePromptPlan{
		Family: "Hunyuan-VL", Placeholder: HunyuanVLImagePad, PlaceholderLabel: "Hunyuan-VL placeholder",
		EmbeddingWidth: r.spec.OutputHidden, Positions: hunyuanImagePromptPositions,
	}, promptDelimiters{
		Prefix: "<｜hy_begin▁of▁sentence｜>", Suffix: "<｜hy_User｜>",
	}, promptDelimiters{Prefix: HunyuanVLImageStart, Suffix: HunyuanVLImageEnd}, gridImagePromptEncoder(r.EncodeImage))
}

func compilePaddleOCRImagePrompt(r *PaddleOCRRunner) compiledImagePromptProgram {
	compile := func(history bool) imagePromptPlan {
		return imagePromptPlan{
			Family: "PaddleOCR", Placeholder: PaddleOCRImagePad, PlaceholderLabel: "PaddleOCR placeholder",
			AddSpecial: history, EmbeddingWidth: r.spec.OutputHidden, Positions: grid2DImagePromptPositions,
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
	}
	return compiledImagePromptProgram{
		Default: compile(false),
		History: compile(true),
		Encode:  gridImagePromptEncoder(r.EncodeImage),
	}
}

func compileMiMoVLImagePrompt(r *MiMoVLRunner) compiledImagePromptProgram {
	encode := func(ctx context.Context, source image.Image) (imagePromptItem, error) {
		output, err := r.EncodeImage(ctx, source)
		if err != nil {
			return imagePromptItem{}, err
		}
		return imagePromptItem{Embeddings: output.Embeddings.Data, Count: int(output.Embeddings.Shape.Dims[tensor.SingletonExtent])}, nil
	}
	return compileFramedImagePromptProgram(imagePromptPlan{
		Family: "MiMo-VL", Placeholder: Qwen3VLImagePad, PlaceholderLabel: "MiMo-VL placeholder",
		EmbeddingWidth: r.spec.ProjectionDim,
	}, promptDelimiters{
		Prefix: "<|im_start|>system\n" + MiMoVLSystemPrompt + "<|im_end|>\n<|im_start|>user\n",
		Suffix: "<|im_end|>\n<|im_start|>assistant\n",
	}, promptDelimiters{Prefix: "<|vision_start|>", Suffix: "<|vision_end|>"}, encode)
}

func compileQwen2VLImagePrompt(r *Qwen2VLRunner) compiledImagePromptProgram {
	return compileSpatialChatImagePromptProgram(
		"Qwen2-VL", r.spec.OutputHidden, "<|im_end|>\n<|im_start|>assistant\n", "",
		spatialGridImagePromptEncoder(r.EncodeImage, r.preprocess.Image),
	)
}

func compileQwen2VLMediaPrompt(r *Qwen2VLRunner) compiledMediaPromptProgram {
	return compiledMediaPromptProgram{Video: func(
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
		output, err := r.EncodeFrames(ctx, frames, r.preprocess.Video)
		if err != nil {
			return MultimodalPrompt{}, err
		}
		count := int(output.Embeddings.Shape.Dims[tensor.SingletonExtent])
		rows, columns := output.GridH/output.MergeSize, output.GridW/output.MergeSize
		return executeProjectedPromptPlan(tokenizer, projectedPromptPlan{
			mediaPromptRunPlan: mediaPromptRunPlan{
				Prompt: "<|im_start|>user\n" + beforeVideo + "<|vision_start|>" +
					strings.Repeat(Qwen3VLVideoPad, count) + "<|vision_end|>" + afterVideo +
					"<|im_end|>\n<|im_start|>assistant\n",
				Placeholder: Qwen3VLVideoPad, Runs: tensor.SingletonExtent, TokensPerRun: count,
				PromptLabel: "Qwen2-VL video prompt", PlaceholderLabel: "video placeholder", RunsLabel: "Qwen2-VL video prompt",
			},
			Embeddings: output.Embeddings.Data, EmbeddingWidth: r.spec.OutputHidden,
			Positions: func(tokenCount int, starts []int) ([tensor.MaxDimensions][]uint32, error) {
				return compileSpatialPositions(tokenCount, positionGrid3D, []spatialPositionChunk{{
					Start: starts[0], Extents: [tensor.TripleExtent]int{rows, columns, output.GridT},
				}})
			},
		})
	}}
}

func compileGemma4ImagePrompt(r *Gemma4Runner) compiledImagePromptProgram {
	return compileGemma4ImageProgram(gemma4PromptSource(r))
}

func compileGemma4ImageProgram(r gemmaPromptSource) compiledImagePromptProgram {
	var attentionBlocks func([]int, []imagePromptItem) []AttentionBlock
	if r.attention {
		attentionBlocks = imagePromptBlocks
	}
	compile := func(history bool) imagePromptPlan {
		return imagePromptPlan{
			Family: "Gemma 4", Placeholder: "<|image|>", PlaceholderLabel: "Gemma 4 image placeholder",
			AddSpecial: history, EmbeddingWidth: r.width, AttentionBlocks: attentionBlocks,
			Render: func(text []string, items []imagePromptItem) string {
				var prompt strings.Builder
				if !history {
					prompt.WriteString("<bos><|turn>user\n")
				}
				for index, item := range items {
					if history {
						prompt.WriteString(text[index])
					}
					prompt.WriteString("<|image>")
					prompt.WriteString(strings.Repeat("<|image|>", item.RunCount))
					prompt.WriteString("<image|>")
				}
				if history {
					prompt.WriteString(text[len(text)-1])
				} else {
					prompt.WriteString(strings.TrimSpace(text[len(text)-1]))
					prompt.WriteString("<turn|>\n<|turn>model\n" + r.assistant)
				}
				return prompt.String()
			},
		}
	}
	return compiledImagePromptProgram{
		Default: compile(false),
		History: compile(true),
		Encode: func(ctx context.Context, source image.Image) (imagePromptItem, error) {
			output, err := r.image(ctx, source)
			if err != nil {
				return imagePromptItem{}, err
			}
			return imagePromptItem{
				Embeddings: output.Embeddings.Data,
				Count:      int(output.Embeddings.Shape.Dims[tensor.SingletonExtent]),
			}, nil
		},
		Prepare: func(text []string, options PromptOptions) ([]string, error) {
			if len(text) < tensor.SingletonExtent {
				return nil, errors.New("projector: Gemma 4 image/text sequence is inconsistent")
			}
			if !options.History {
				for _, segment := range text[:len(text)-tensor.SingletonExtent] {
					if strings.TrimSpace(segment) != "" {
						return nil, errors.New("projector: Gemma 4 requires images before user text")
					}
				}
			}
			return text, nil
		},
	}
}

func compileGemma4MediaPrompt(r *Gemma4Runner) compiledMediaPromptProgram {
	return compileGemma4MediaProgram(gemma4PromptSource(r))
}

func compileGemma4MediaProgram(r gemmaPromptSource) compiledMediaPromptProgram {
	var attentionBlocks func([]int, int) []AttentionBlock
	if r.attention {
		attentionBlocks = mediaPromptAttentionBlocks
	}
	history := mixedMediaPromptPlan{
		Family: "Gemma 4", AddSpecial: true, EmbeddingWidth: r.width, PromptLabel: "Gemma 4 media history",
		Render: renderMixedMediaHistory,
		Kinds: map[MediaKind]mixedMediaKindPlan{
			MediaImage: {
				Placeholder: "<|image|>", PlaceholderLabel: "Gemma 4 image placeholder",
				Open: "<|image>", Close: "<image|>", Attention: r.attention,
				Encode: func(ctx context.Context, input MediaInput) (imagePromptItem, error) {
					output, err := r.image(ctx, input.Image)
					if err != nil {
						return imagePromptItem{}, err
					}
					return imagePromptItem{Embeddings: output.Embeddings.Data, Count: int(output.Embeddings.Shape.Dims[tensor.SingletonExtent])}, nil
				},
			},
			MediaAudio: {
				Placeholder: "<|audio|>", PlaceholderLabel: "Gemma 4 audio placeholder",
				Open: "<|audio>", Close: "<audio|>",
				Encode: func(ctx context.Context, input MediaInput) (imagePromptItem, error) {
					output, err := r.audio(ctx, input.Audio)
					if err != nil {
						return imagePromptItem{}, err
					}
					return imagePromptItem{Embeddings: output.Embeddings.Data, Count: int(output.Embeddings.Shape.Dims[tensor.SingletonExtent])}, nil
				},
			},
		},
	}
	video := func(
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
		if !checked.PositiveFinite64(fps) {
			return MultimodalPrompt{}, errors.New("projector: video FPS must be positive and finite")
		}
		output, err := r.video(ctx, frames)
		if err != nil {
			return MultimodalPrompt{}, err
		}
		return executeProjectedPromptPlan(tokenizer, projectedPromptPlan{
			mediaPromptRunPlan: mediaPromptRunPlan{
				Prompt: gemma4VideoPromptText(afterVideo, output.Frames, output.TokensPerFrame, fps, r.assistant), Placeholder: "<|video|>",
				Runs: output.Frames, TokensPerRun: output.TokensPerFrame,
				PromptLabel: "Gemma 4 video prompt", PlaceholderLabel: "Gemma 4 video placeholder", RunsLabel: "Gemma 4 video prompt",
			},
			Embeddings: output.Embeddings.Data, EmbeddingWidth: int(output.Embeddings.Shape.Dims[tensor.FirstOffset]),
			AttentionBlocks: attentionBlocks,
		})
	}
	return compiledMediaPromptProgram{
		Video: video,
		History: func(ctx context.Context, tokenizer ImageTokenizer, media []MediaInput, text []string) (MultimodalPrompt, error) {
			return executeMixedMediaPromptPlan(ctx, tokenizer, media, text, history)
		},
		Audio: func(
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
			output, err := r.audio(ctx, samples)
			if err != nil {
				return MultimodalPrompt{}, err
			}
			audioTokens := int(output.Embeddings.Shape.Dims[tensor.SingletonExtent])
			return executeProjectedPromptPlan(tokenizer, projectedPromptPlan{
				mediaPromptRunPlan: mediaPromptRunPlan{
					Prompt: gemma4AudioPromptText(afterAudio, audioTokens, r.assistant), Placeholder: "<|audio|>",
					Runs: tensor.SingletonExtent, TokensPerRun: audioTokens,
					PromptLabel: "Gemma 4 audio prompt", PlaceholderLabel: "Gemma 4 audio placeholder", RunsLabel: "Gemma 4 audio prompt",
				},
				Embeddings: output.Embeddings.Data, EmbeddingWidth: int(output.Embeddings.Shape.Dims[tensor.FirstOffset]),
			})
		},
		AudioSampleRate: r.sampleRate,
	}
}

func Gemma4AudioPromptText(question string, audioTokens int) string {
	return gemma4AudioPromptText(question, audioTokens, gemma4ClosedThought)
}

func gemma4AudioPromptText(question string, audioTokens int, assistant string) string {
	return "<bos><|turn>user\n<|audio>" + strings.Repeat("<|audio|>", audioTokens) +
		"<audio|>" + strings.TrimSpace(question) +
		"<turn|>\n<|turn>model\n" + assistant
}

func Gemma4VideoPromptText(question string, frames, tokensPerFrame int, fps float64) string {
	return gemma4VideoPromptText(question, frames, tokensPerFrame, fps, gemma4ClosedThought)
}

func gemma4VideoPromptText(question string, frames, tokensPerFrame int, fps float64, assistant string) string {
	var prompt strings.Builder
	prompt.WriteString("<bos><|turn>user\n")
	for frame := range frames {
		if frame > tensor.FirstOffset {
			prompt.WriteByte(' ')
		}
		seconds := int(float64(frame) / fps)
		secondsPerMinute := int(time.Minute / time.Second)
		fmt.Fprintf(&prompt, "%02d:%02d <|image>", seconds/secondsPerMinute, seconds%secondsPerMinute)
		prompt.WriteString(strings.Repeat("<|video|>", tokensPerFrame))
		prompt.WriteString("<image|>")
	}
	prompt.WriteString(strings.TrimSpace(question))
	prompt.WriteString("<turn|>\n<|turn>model\n" + assistant)
	return prompt.String()
}

func compileQwen3VLImagePrompt(r *Qwen3VLRunner) compiledImagePromptProgram {
	baseSuffix := "<|im_end|>\n<|im_start|>assistant\n<think>\n"
	return compileSpatialChatImagePromptProgram(
		"Qwen3-VL", r.spec.OutputHidden, baseSuffix+"\n</think>\n\n", baseSuffix,
		spatialGridImagePromptEncoder(r.EncodeImage, r.preprocess.Image),
	)
}

func compileQwen3VLMediaPrompt(r *Qwen3VLRunner) compiledMediaPromptProgram {
	return compiledMediaPromptProgram{Video: func(
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
		if !checked.PositiveFinite64(fps) {
			return MultimodalPrompt{}, errors.New("projector: video FPS must be positive and finite")
		}
		output, err := r.EncodeFrames(ctx, frames, r.preprocess.Video)
		if err != nil {
			return MultimodalPrompt{}, err
		}
		rows, columns := output.GridH/output.MergeSize, output.GridW/output.MergeSize
		perGroup := rows * columns
		item, err := spatialGridImagePromptItem(output)
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
			EmbeddingWidth: int(output.Embeddings.Shape.Dims[tensor.FirstOffset]),
			Positions: func(tokenCount int, starts []int) ([tensor.MaxDimensions][]uint32, error) {
				chunks := make([]spatialPositionChunk, len(starts))
				for index, start := range starts {
					chunks[index] = spatialPositionChunk{Start: start, Extents: [tensor.TripleExtent]int{rows, columns}}
				}
				return compileSpatialPositions(tokenCount, positionGrid2D, chunks)
			},
		})
	}}
}

func Qwen35VideoPromptText(beforeVideo, afterVideo string, groups, tokensPerGroup int, fps float64, thinking bool) string {
	var prompt strings.Builder
	prompt.WriteString("<|im_start|>user\n")
	prompt.WriteString(beforeVideo)
	prompt.WriteString("<|vision_start|>")
	for group := range groups {
		timestamp := (float64(group*tensor.PairedExtent) + media.RasterSampleCenter) / fps
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
	if len(counts) == tensor.FirstOffset {
		return nil, errors.New("media token counts are empty")
	}
	starts := make([]int, tensor.FirstOffset, len(counts))
	for index := tensor.FirstOffset; index < len(ids); {
		if ids[index] != token {
			index++
			continue
		}
		end := index + tensor.SingletonExtent
		for end < len(ids) && ids[end] == token {
			end++
		}
		starts = append(starts, index)
		if len(starts) > len(counts) || end-index != counts[len(starts)-tensor.SingletonExtent] {
			return nil, fmt.Errorf("placeholder run %d has %d tokens", len(starts)-tensor.SingletonExtent, end-index)
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
	if len(tokens) == tensor.FirstOffset || len(tokens) != len(counts) {
		return nil, errors.New("media token run specification is inconsistent")
	}
	mediaTokens := make(map[tokenizer.TokenID]struct{}, len(tokens))
	for _, token := range tokens {
		mediaTokens[token] = struct{}{}
	}
	starts := make([]int, tensor.FirstOffset, len(tokens))
	foundTokens := make([]tokenizer.TokenID, tensor.FirstOffset, len(tokens))
	foundCounts := make([]int, tensor.FirstOffset, len(tokens))
	for index := tensor.FirstOffset; index < len(ids); {
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

type spatialPositionLayout uint8

const (
	positionGrid2D spatialPositionLayout = iota
	positionGrid3D
	positionDelimitedRows
)

type spatialPositionChunk struct {
	Start   int
	Extents [tensor.TripleExtent]int
	Index   int
}

func compileSpatialPositions(tokens int, layout spatialPositionLayout, chunks []spatialPositionChunk) ([tensor.MaxDimensions][]uint32, error) {
	var positions [tensor.MaxDimensions][]uint32
	if !checked.PositiveInts(tokens, len(chunks)) || layout > positionDelimitedRows {
		return positions, errors.New("projector: invalid spatial positions")
	}
	for axis := range positions {
		positions[axis] = make([]uint32, tokens)
	}
	next, physical := uint32(tensor.FirstOffset), tensor.FirstOffset
	fillText := func(end int) {
		for physical < end {
			for axis := range positions {
				positions[axis][physical] = next
			}
			next++
			physical++
		}
	}
	for _, chunk := range chunks {
		first := chunk.Extents[tensor.FirstOffset]
		second := chunk.Extents[tensor.SingletonExtent]
		third := chunk.Extents[tensor.PairedExtent]
		if !checked.PositiveInts(first, second) || layout == positionGrid3D && !checked.PositiveInts(third) ||
			layout == positionDelimitedRows && chunk.Index < tensor.FirstOffset || chunk.Start < physical {
			return [tensor.MaxDimensions][]uint32{}, errors.New("projector: spatial position chunk is invalid")
		}
		plane, ok := checked.MulInt(first, second)
		count := plane
		switch layout {
		case positionGrid3D:
			count, ok = checked.MulInt(plane, third)
		case positionDelimitedRows:
			columns, columnsOK := checked.Add64(uint64(second), tensor.SingletonExtent)
			count64, countOK := checked.Mul64(uint64(first), columns)
			count64, addOK := checked.Add64(count64, tensor.PairedExtent)
			count, ok = checked.Int(count64)
			ok = ok && columnsOK && countOK && addOK
		}
		end64, endOK := checked.Add64(uint64(chunk.Start), uint64(count))
		end, endOK := checked.Int(end64)
		if !ok || !endOK || end > tokens {
			return [tensor.MaxDimensions][]uint32{}, errors.New("projector: spatial position chunk is invalid")
		}
		fillText(chunk.Start)
		base := next
		for index := range count {
			position := physical + index
			switch layout {
			case positionGrid2D:
				positions[0][position] = base
				positions[1][position] = base + uint32(index/second)
				positions[2][position] = base + uint32(index%second)
			case positionGrid3D:
				positions[0][position] = base + uint32(index/plane)
				positions[1][position] = base + uint32((index%plane)/second)
				positions[2][position] = base + uint32(index%second)
			case positionDelimitedRows:
				if index == 0 || index == count-tensor.SingletonExtent {
					for axis := tensor.SingletonExtent; axis < tensor.MaxDimensions; axis++ {
						positions[axis][position] = base + uint32(index)
					}
					positions[0][position] = base + uint32(index)
					continue
				}
				offset := index - tensor.SingletonExtent
				positions[0][position] = base + uint32(index)
				positions[1][position] = uint32(offset % (second + tensor.SingletonExtent))
				positions[2][position] = uint32(offset / (second + tensor.SingletonExtent))
				positions[3][position] = uint32(chunk.Index)
			}
		}
		physical += count
		switch layout {
		case positionGrid2D:
			next = base + uint32(max(first, second))
		case positionGrid3D:
			next = base + uint32(max(first, second, third))
		case positionDelimitedRows:
			next = base + uint32(count)
		default:
			return [tensor.MaxDimensions][]uint32{}, errors.New("projector: unknown spatial position layout")
		}
	}
	fillText(tokens)
	return positions, nil
}
