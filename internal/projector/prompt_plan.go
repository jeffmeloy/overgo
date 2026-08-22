package projector

import (
	"context"
	"errors"
	"fmt"
	"image"
	"slices"
	"strings"

	"overgo/internal/checked"
	"overgo/internal/tensor"
	"overgo/internal/tensor/reference"
	"overgo/internal/tokenizer"
)

type imagePromptItem struct {
	Embeddings []float32
	Deepstack  [][]float32
	Count      int
	RunCount   int
	Rows       int
	Columns    int
}

type imagePromptPlan struct {
	Family           string
	Placeholder      string
	PlaceholderLabel string
	AddSpecial       bool
	EmbeddingWidth   int
	EmbeddingOffset  int
	Render           func([]string, []imagePromptItem) string
	Positions        func(int, []int, []imagePromptItem) ([tensor.MaxDimensions][]uint32, error)
	AttentionBlocks  func([]int, []imagePromptItem) []AttentionBlock
}

type imagePromptEncoder func(context.Context, image.Image) (imagePromptItem, error)

func delimitedImagePromptPlan(family, placeholder, label string, addSpecial bool, width int, prefix, suffix string) imagePromptPlan {
	return imagePromptPlan{
		Family: family, Placeholder: placeholder, PlaceholderLabel: label,
		AddSpecial: addSpecial, EmbeddingWidth: width,
		Render: func(text []string, items []imagePromptItem) string {
			return renderDelimitedImagePrompt(text, items, placeholder, prefix, suffix)
		},
	}
}

func referenceImageEncoder(encode func(context.Context, image.Image) (reference.Value, error)) imagePromptEncoder {
	return func(ctx context.Context, source image.Image) (imagePromptItem, error) {
		value, err := encode(ctx, source)
		return imagePromptItem{Embeddings: value.Data, Count: int(value.Shape.Dims[tensor.SingletonExtent])}, err
	}
}

func gridImagePromptEncoder(
	encode func(context.Context, image.Image, RasterPatchOptions) (gridOutput, error),
) imagePromptEncoder {
	return func(ctx context.Context, source image.Image) (imagePromptItem, error) {
		output, err := encode(ctx, source, RasterPatchOptions{})
		return imagePromptItem{
			Embeddings: output.Embeddings.Data, Count: int(output.Embeddings.Shape.Dims[tensor.SingletonExtent]),
			Rows: output.GridH / output.MergeSize, Columns: output.GridW / output.MergeSize,
		}, err
	}
}

func qwenImagePromptEncoder(
	encode func(context.Context, image.Image, Qwen3VLPreprocessOptions) (Qwen3VLOutput, error),
) imagePromptEncoder {
	return func(ctx context.Context, source image.Image) (imagePromptItem, error) {
		output, err := encode(ctx, source, DefaultQwen3VLPreprocessOptions())
		if err != nil {
			return imagePromptItem{}, err
		}
		return qwenImagePromptItem(output)
	}
}

type mixedMediaPromptItem struct {
	Kind        MediaKind
	Placeholder string
	Open        string
	Close       string
	Embeddings  []float32
	Token       tokenizer.TokenID
	Count       int
	Attention   bool
}

type mixedMediaKindPlan struct {
	Placeholder      string
	PlaceholderLabel string
	Open             string
	Close            string
	Attention        bool
	Encode           func(context.Context, MediaInput) (imagePromptItem, error)
}

type mixedMediaPromptPlan struct {
	Family         string
	AddSpecial     bool
	EmbeddingWidth int
	PromptLabel    string
	Kinds          map[MediaKind]mixedMediaKindPlan
	Render         func([]string, []mixedMediaPromptItem) string
}

type mediaPromptRunPlan struct {
	Prompt           string
	AddSpecial       bool
	Placeholder      string
	Runs             int
	TokensPerRun     int
	PromptLabel      string
	PlaceholderLabel string
	RunsLabel        string
}

type mediaPromptRuns struct {
	TokenIDs []tokenizer.TokenID
	Starts   []int
	Counts   []int
	Indices  []uint32
}

type projectedPromptOutput struct {
	TokenIDs        []tokenizer.TokenID
	Embeddings      []float32
	Deepstack       [][]float32
	EmbeddingWidth  int
	Starts          []int
	Counts          []int
	EmbeddingOffset int
	Positions       [tensor.MaxDimensions][]uint32
	AttentionBlocks []AttentionBlock
}

type projectedPromptPlan struct {
	mediaPromptRunPlan
	Embeddings      []float32
	Deepstack       [][]float32
	EmbeddingWidth  int
	Positions       func(int, []int) ([tensor.MaxDimensions][]uint32, error)
	AttentionBlocks func([]int, int) []AttentionBlock
}

func compileMediaPromptRuns(tokenizer ImageTokenizer, plan mediaPromptRunPlan) (mediaPromptRuns, error) {
	if tokenizer == nil {
		return mediaPromptRuns{}, errors.New("projector: tokenizer is nil")
	}
	if !checked.PositiveInts(plan.Runs, plan.TokensPerRun) {
		return mediaPromptRuns{}, errors.New("projector: media prompt run plan is invalid")
	}
	counts := make([]int, plan.Runs)
	for index := range counts {
		counts[index] = plan.TokensPerRun
	}
	ids, starts, err := tokenizePromptRuns(
		tokenizer, plan.Prompt, plan.AddSpecial, plan.Placeholder, counts,
		plan.PromptLabel, plan.PlaceholderLabel, plan.RunsLabel,
	)
	if err != nil {
		return mediaPromptRuns{}, err
	}
	return mediaPromptRuns{
		TokenIDs: ids, Starts: starts, Counts: counts,
		Indices: embeddingTokenIndices(starts, counts, tensor.FirstOffset),
	}, nil
}

func executeProjectedPromptPlan(tokenizer ImageTokenizer, plan projectedPromptPlan) (MultimodalPrompt, error) {
	runs, err := compileMediaPromptRuns(tokenizer, plan.mediaPromptRunPlan)
	if err != nil {
		return MultimodalPrompt{}, err
	}
	positions := [tensor.MaxDimensions][]uint32{}
	if plan.Positions != nil {
		positions, err = plan.Positions(len(runs.TokenIDs), runs.Starts)
		if err != nil {
			return MultimodalPrompt{}, err
		}
	}
	var blocks []AttentionBlock
	if plan.AttentionBlocks != nil {
		blocks = plan.AttentionBlocks(runs.Starts, plan.TokensPerRun)
	}
	return assembleProjectedPrompt(projectedPromptOutput{
		TokenIDs: runs.TokenIDs, Embeddings: plan.Embeddings, Deepstack: plan.Deepstack,
		EmbeddingWidth: plan.EmbeddingWidth, Starts: runs.Starts, Counts: runs.Counts,
		Positions: positions, AttentionBlocks: blocks,
	})
}

func assembleProjectedPrompt(output projectedPromptOutput) (MultimodalPrompt, error) {
	if len(output.Starts) == tensor.FirstOffset || len(output.Starts) != len(output.Counts) {
		return MultimodalPrompt{}, errors.New("projector: projected prompt runs are inconsistent")
	}
	if !checked.PositiveInts(output.EmbeddingWidth) {
		return MultimodalPrompt{}, errors.New("projector: projected prompt embedding width is invalid")
	}
	tokenCount := tensor.FirstOffset
	for _, count := range output.Counts {
		if !checked.PositiveInts(count) {
			return MultimodalPrompt{}, errors.New("projector: projected prompt run is empty")
		}
		tokenCount += count
	}
	if len(output.Embeddings) != tokenCount*output.EmbeddingWidth {
		return MultimodalPrompt{}, fmt.Errorf(
			"projector: projected prompt embeddings = %d, want %d",
			len(output.Embeddings), tokenCount*output.EmbeddingWidth,
		)
	}
	for index, stream := range output.Deepstack {
		if len(stream) != len(output.Embeddings) {
			return MultimodalPrompt{}, fmt.Errorf("projector: deepstack stream %d shape differs from base embeddings", index)
		}
	}
	return MultimodalPrompt{
		TokenIDs: output.TokenIDs, Embeddings: output.Embeddings, DeepstackEmbeddings: output.Deepstack,
		EmbeddingWidth: output.EmbeddingWidth, EmbeddingStart: output.Starts[tensor.FirstOffset] + output.EmbeddingOffset,
		EmbeddingTokenIndices: embeddingTokenIndices(output.Starts, output.Counts, output.EmbeddingOffset),
		MultiAxisPositions:    output.Positions, AttentionBlocks: output.AttentionBlocks,
	}, nil
}

func mediaPromptAttentionBlocks(starts []int, tokensPerRun int) []AttentionBlock {
	blocks := make([]AttentionBlock, len(starts))
	for index, start := range starts {
		blocks[index] = AttentionBlock{Start: uint32(start), End: uint32(start + tokensPerRun)}
	}
	return blocks
}

func executeImagePromptPlan(
	ctx context.Context,
	tokenizerAPI ImageTokenizer,
	sources []image.Image,
	text []string,
	plan imagePromptPlan,
	encode imagePromptEncoder,
) (MultimodalPrompt, error) {
	if err := validateImagePromptInputs(tokenizerAPI, sources, text, plan.Family); err != nil {
		return MultimodalPrompt{}, err
	}
	if encode == nil || plan.Render == nil || !checked.PositiveInts(plan.EmbeddingWidth) {
		return MultimodalPrompt{}, errors.New("projector: image prompt plan is incomplete")
	}
	items := make([]imagePromptItem, len(sources))
	counts := make([]int, len(sources))
	for index, source := range sources {
		item, err := encode(ctx, source)
		if err != nil {
			return MultimodalPrompt{}, fmt.Errorf("projector: encode %s image %d: %w", plan.Family, index, err)
		}
		if !checked.PositiveInts(item.Count) {
			return MultimodalPrompt{}, fmt.Errorf("projector: %s image %d produced no embedding tokens", plan.Family, index)
		}
		if err := validateRowStorage(item.Count, rowStorage{elements: len(item.Embeddings), width: plan.EmbeddingWidth}); err != nil {
			return MultimodalPrompt{}, fmt.Errorf("projector: %s image %d embedding shape: %w", plan.Family, index, err)
		}
		if item.RunCount == tensor.FirstOffset {
			item.RunCount = item.Count
		}
		if item.RunCount < item.Count+plan.EmbeddingOffset {
			return MultimodalPrompt{}, fmt.Errorf("projector: %s image %d placeholder run is too short", plan.Family, index)
		}
		items[index] = item
		counts[index] = item.RunCount
	}
	embeddingElements := tensor.FirstOffset
	deepstackStreams := len(items[tensor.FirstOffset].Deepstack)
	for index, item := range items {
		embeddingElements += len(item.Embeddings)
		if len(item.Deepstack) != deepstackStreams {
			return MultimodalPrompt{}, fmt.Errorf(
				"projector: collect %s image %d deepstack: stream count changed", plan.Family, index,
			)
		}
	}
	embeddings := make([]float32, tensor.FirstOffset, embeddingElements)
	deepstack := make([][]float32, deepstackStreams)
	for stream := range deepstack {
		capacity := tensor.FirstOffset
		for _, item := range items {
			capacity += len(item.Deepstack[stream])
		}
		deepstack[stream] = make([]float32, tensor.FirstOffset, capacity)
	}
	for _, item := range items {
		embeddings = append(embeddings, item.Embeddings...)
		for stream := range deepstack {
			deepstack[stream] = append(deepstack[stream], item.Deepstack[stream]...)
		}
	}
	promptItems := items
	markerTokenizer, compact := tokenizerAPI.(promptMarkerTokenizer)
	if compact {
		promptItems = slices.Clone(items)
		for index := range promptItems {
			promptItems[index].RunCount = tensor.SingletonExtent
		}
	}
	prompt := plan.Render(text, promptItems)
	var ids []tokenizer.TokenID
	var starts []int
	var err error
	if compact {
		ids, starts, err = markerTokenizer.TokenizeTextMarkers(
			prompt, plan.Placeholder, counts, plan.AddSpecial,
		)
		if err != nil {
			err = fmt.Errorf("projector: tokenize %s image prompt: %w", plan.Family, err)
		}
	} else {
		ids, starts, err = tokenizeImagePromptRuns(
			tokenizerAPI, prompt, plan.AddSpecial, plan.Placeholder, counts,
			plan.Family, plan.PlaceholderLabel,
		)
	}
	if err != nil {
		return MultimodalPrompt{}, err
	}
	positions := [tensor.MaxDimensions][]uint32{}
	if plan.Positions != nil {
		positions, err = plan.Positions(len(ids), starts, items)
		if err != nil {
			return MultimodalPrompt{}, err
		}
	}
	embeddingCounts := make([]int, len(items))
	for index := range items {
		embeddingCounts[index] = items[index].Count
	}
	return assembleProjectedPrompt(projectedPromptOutput{
		TokenIDs: ids, Embeddings: embeddings, Deepstack: deepstack, EmbeddingWidth: plan.EmbeddingWidth,
		Starts: starts, Counts: embeddingCounts, EmbeddingOffset: plan.EmbeddingOffset,
		Positions: positions, AttentionBlocks: imagePromptAttentionBlocks(plan, starts, items),
	})
}

func executeMixedMediaPromptPlan(
	ctx context.Context,
	tokenizerAPI ImageTokenizer,
	media []MediaInput,
	text []string,
	plan mixedMediaPromptPlan,
) (MultimodalPrompt, error) {
	if err := validateMediaHistoryInputs(tokenizerAPI, media, text, plan.Family); err != nil {
		return MultimodalPrompt{}, err
	}
	if plan.Render == nil || len(plan.Kinds) == tensor.FirstOffset || !checked.PositiveInts(plan.EmbeddingWidth) {
		return MultimodalPrompt{}, errors.New("projector: mixed-media prompt plan is incomplete")
	}
	items := make([]mixedMediaPromptItem, len(media))
	for index, input := range media {
		kind, ok := plan.Kinds[input.Kind]
		if !ok || kind.Encode == nil || kind.Placeholder == "" {
			return MultimodalPrompt{}, fmt.Errorf("projector: %s media kind %d is unsupported", plan.Family, input.Kind)
		}
		placeholderIDs, err := tokenizerAPI.TokenizeText(kind.Placeholder, false, true)
		if err != nil {
			return MultimodalPrompt{}, fmt.Errorf("projector: tokenize %s: %w", kind.PlaceholderLabel, err)
		}
		if len(placeholderIDs) != tensor.SingletonExtent {
			return MultimodalPrompt{}, fmt.Errorf("projector: %s maps to %d tokens", kind.PlaceholderLabel, len(placeholderIDs))
		}
		encoded, err := kind.Encode(ctx, input)
		if err != nil {
			return MultimodalPrompt{}, fmt.Errorf("projector: encode %s media %d: %w", plan.Family, index, err)
		}
		if err := validateRowStorage(encoded.Count, rowStorage{elements: len(encoded.Embeddings), width: plan.EmbeddingWidth}); err != nil {
			return MultimodalPrompt{}, fmt.Errorf("projector: %s media %d embedding shape is invalid", plan.Family, index)
		}
		items[index] = mixedMediaPromptItem{
			Kind: input.Kind, Placeholder: kind.Placeholder, Open: kind.Open, Close: kind.Close,
			Embeddings: encoded.Embeddings, Token: placeholderIDs[tensor.FirstOffset], Count: encoded.Count,
			Attention: kind.Attention,
		}
	}
	embeddingElements := tensor.FirstOffset
	for _, item := range items {
		embeddingElements += len(item.Embeddings)
	}
	embeddings := make([]float32, tensor.FirstOffset, embeddingElements)
	for _, item := range items {
		embeddings = append(embeddings, item.Embeddings...)
	}
	ids, err := tokenizerAPI.TokenizeText(plan.Render(text, items), plan.AddSpecial, true)
	if err != nil {
		return MultimodalPrompt{}, fmt.Errorf("projector: tokenize %s: %w", plan.PromptLabel, err)
	}
	expectedTokens := make([]tokenizer.TokenID, len(items))
	counts := make([]int, len(items))
	for index, item := range items {
		expectedTokens[index], counts[index] = item.Token, item.Count
	}
	starts, err := orderedVariableTokenRuns(ids, expectedTokens, counts)
	if err != nil {
		return MultimodalPrompt{}, fmt.Errorf("projector: %s: %w", plan.PromptLabel, err)
	}
	blocks := make([]AttentionBlock, tensor.FirstOffset, len(items))
	for index, item := range items {
		if item.Attention {
			blocks = append(blocks, AttentionBlock{Start: uint32(starts[index]), End: uint32(starts[index] + item.Count)})
		}
	}
	return assembleProjectedPrompt(projectedPromptOutput{
		TokenIDs: ids, Embeddings: embeddings, EmbeddingWidth: plan.EmbeddingWidth,
		Starts: starts, Counts: counts, AttentionBlocks: blocks,
	})
}

func renderMixedMediaHistory(text []string, items []mixedMediaPromptItem) string {
	var prompt strings.Builder
	for index, item := range items {
		prompt.WriteString(text[index])
		prompt.WriteString(item.Open)
		prompt.WriteString(strings.Repeat(item.Placeholder, item.Count))
		prompt.WriteString(item.Close)
	}
	prompt.WriteString(text[len(text)-tensor.SingletonExtent])
	return prompt.String()
}

func imagePromptAttentionBlocks(plan imagePromptPlan, starts []int, items []imagePromptItem) []AttentionBlock {
	if plan.AttentionBlocks == nil {
		return nil
	}
	return plan.AttentionBlocks(starts, items)
}

func qwenImagePromptPositions(tokenCount int, starts []int, items []imagePromptItem) ([tensor.MaxDimensions][]uint32, error) {
	chunks := make([]spatialPositionChunk, len(starts))
	for index, start := range starts {
		chunks[index] = spatialPositionChunk{Start: start, Extents: [tensor.TripleExtent]int{items[index].Rows, items[index].Columns}}
	}
	return compileSpatialPositions(tokenCount, positionGrid2D, chunks)
}

func hunyuanImagePromptPositions(tokenCount int, starts []int, items []imagePromptItem) ([tensor.MaxDimensions][]uint32, error) {
	chunks := make([]spatialPositionChunk, len(starts))
	for index, start := range starts {
		chunks[index] = spatialPositionChunk{
			Start: start, Extents: [tensor.TripleExtent]int{items[index].Rows, items[index].Columns}, Index: index,
		}
	}
	return compileSpatialPositions(tokenCount, positionDelimitedRows, chunks)
}

func qwenImagePlan(
	family string,
	history bool,
	width int,
	render func([]string, []imagePromptItem) string,
) imagePromptPlan {
	return imagePromptPlan{
		Family: family, Placeholder: Qwen3VLImagePad, PlaceholderLabel: "image placeholder",
		AddSpecial: history, EmbeddingWidth: width, Render: render, Positions: qwenImagePromptPositions,
	}
}

func renderQwenImagePrompt(text []string, items []imagePromptItem, history bool, suffix string) string {
	var prompt strings.Builder
	if !history {
		prompt.WriteString("<|im_start|>user\n")
	}
	for index, item := range items {
		prompt.WriteString(text[index])
		prompt.WriteString("<|vision_start|>")
		prompt.WriteString(strings.Repeat(Qwen3VLImagePad, item.RunCount))
		prompt.WriteString("<|vision_end|>")
	}
	prompt.WriteString(text[len(text)-1])
	if !history {
		prompt.WriteString(suffix)
	}
	return prompt.String()
}

func renderDelimitedImagePrompt(text []string, items []imagePromptItem, placeholder, prefix, suffix string) string {
	var prompt strings.Builder
	for index, item := range items {
		prompt.WriteString(text[index])
		prompt.WriteString(prefix)
		prompt.WriteString(strings.Repeat(placeholder, item.RunCount))
		prompt.WriteString(suffix)
	}
	prompt.WriteString(text[len(text)-1])
	return prompt.String()
}

func qwenImagePromptItem(output Qwen3VLOutput) (imagePromptItem, error) {
	deepstack := make([][]float32, len(output.DeepstackEmbeddings))
	for index, stream := range output.DeepstackEmbeddings {
		if !stream.Shape.Equal(output.Embeddings.Shape) {
			return imagePromptItem{}, fmt.Errorf("deepstack stream %d shape differs from base embeddings", index)
		}
		deepstack[index] = stream.Data
	}
	return imagePromptItem{
		Embeddings: output.Embeddings.Data, Deepstack: deepstack,
		Count: int(output.Embeddings.Shape.Dims[tensor.SingletonExtent]),
		Rows:  output.GridH / output.MergeSize, Columns: output.GridW / output.MergeSize,
	}, nil
}

func imagePromptBlocks(starts []int, items []imagePromptItem) []AttentionBlock {
	blocks := make([]AttentionBlock, len(starts))
	for index, start := range starts {
		blocks[index] = AttentionBlock{Start: uint32(start), End: uint32(start + items[index].Count)}
	}
	return blocks
}
