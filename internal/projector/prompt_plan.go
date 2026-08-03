package projector

import (
	"context"
	"errors"
	"fmt"
	"image"
	"strings"
)

type imagePromptItem struct {
	Embeddings []float32
	Deepstack  [][]float32
	Count      int
	RunCount   int
	Rows       int
	Columns    int
	Width      int
}

type imagePromptPlan struct {
	Family           string
	Placeholder      string
	PlaceholderLabel string
	History          bool
	EmbeddingWidth   int
	EmbeddingOffset  int
	Render           func([]string, []imagePromptItem) string
	Positions        func(int, []int, []imagePromptItem) ([4][]uint32, error)
	AttentionBlocks  func([]int, []imagePromptItem) []AttentionBlock
}

type imagePromptEncoder func(context.Context, image.Image) (imagePromptItem, error)

func executeImagePromptPlan(
	ctx context.Context,
	tokenizer ImageTokenizer,
	sources []image.Image,
	text []string,
	plan imagePromptPlan,
	encode imagePromptEncoder,
) (MultimodalPrompt, error) {
	if err := validateImagePromptInputs(tokenizer, sources, text, plan.Family); err != nil {
		return MultimodalPrompt{}, err
	}
	if encode == nil || plan.Render == nil {
		return MultimodalPrompt{}, errors.New("projector: image prompt plan is incomplete")
	}
	items := make([]imagePromptItem, len(sources))
	counts := make([]int, len(sources))
	var embeddings []float32
	var deepstack [][]float32
	for index, source := range sources {
		item, err := encode(ctx, source)
		if err != nil {
			return MultimodalPrompt{}, fmt.Errorf("projector: encode %s image %d: %w", plan.Family, index, err)
		}
		if item.Count <= 0 {
			return MultimodalPrompt{}, fmt.Errorf("projector: %s image %d produced no embedding tokens", plan.Family, index)
		}
		if item.RunCount == 0 {
			item.RunCount = item.Count
		}
		if item.RunCount < item.Count+plan.EmbeddingOffset {
			return MultimodalPrompt{}, fmt.Errorf("projector: %s image %d placeholder run is too short", plan.Family, index)
		}
		items[index] = item
		counts[index] = item.RunCount
		embeddings = append(embeddings, item.Embeddings...)
		if err := appendPromptDeepstack(&deepstack, item.Deepstack); err != nil {
			return MultimodalPrompt{}, fmt.Errorf("projector: collect %s image %d deepstack: %w", plan.Family, index, err)
		}
	}
	ids, starts, err := tokenizeImagePromptRuns(
		tokenizer, plan.Render(text, items), plan.History, plan.Placeholder, counts,
		plan.Family, plan.PlaceholderLabel,
	)
	if err != nil {
		return MultimodalPrompt{}, err
	}
	positions := [4][]uint32{}
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
	width := plan.EmbeddingWidth
	if width == 0 {
		width = items[0].Width
	}
	for index, item := range items {
		if item.Width != 0 && item.Width != width {
			return MultimodalPrompt{}, fmt.Errorf("projector: %s image %d embedding width changed", plan.Family, index)
		}
	}
	return MultimodalPrompt{
		TokenIDs:              ids,
		Embeddings:            embeddings,
		DeepstackEmbeddings:   deepstack,
		EmbeddingWidth:        width,
		EmbeddingStart:        starts[0] + plan.EmbeddingOffset,
		EmbeddingTokenIndices: embeddingTokenIndices(starts, embeddingCounts, plan.EmbeddingOffset),
		MultiAxisPositions:    positions,
		AttentionBlocks:       imagePromptAttentionBlocks(plan, starts, items),
	}, nil
}

func imagePromptAttentionBlocks(plan imagePromptPlan, starts []int, items []imagePromptItem) []AttentionBlock {
	if plan.AttentionBlocks == nil {
		return nil
	}
	return plan.AttentionBlocks(starts, items)
}

func appendPromptDeepstack(target *[][]float32, streams [][]float32) error {
	if len(streams) == 0 {
		if len(*target) != 0 {
			return errors.New("deepstack stream count changed")
		}
		return nil
	}
	if len(*target) == 0 {
		*target = make([][]float32, len(streams))
	}
	if len(*target) != len(streams) {
		return fmt.Errorf("deepstack streams = %d, want %d", len(streams), len(*target))
	}
	for index := range streams {
		(*target)[index] = append((*target)[index], streams[index]...)
	}
	return nil
}

func qwenImagePromptPositions(tokenCount int, starts []int, items []imagePromptItem) ([4][]uint32, error) {
	chunks := make([]Qwen3VLPositionChunk, len(starts))
	for index, start := range starts {
		chunks[index] = Qwen3VLPositionChunk{Start: start, Rows: items[index].Rows, Columns: items[index].Columns}
	}
	return Qwen3VLVariableChunkPositions(tokenCount, chunks)
}

func hunyuanImagePromptPositions(tokenCount int, starts []int, items []imagePromptItem) ([4][]uint32, error) {
	chunks := make([]HunyuanVLPositionChunk, len(starts))
	for index, start := range starts {
		chunks[index] = HunyuanVLPositionChunk{
			Start: start, Rows: items[index].Rows, Columns: items[index].Columns, ImageIndex: index,
		}
	}
	return HunyuanVLVariableChunkPositions(tokenCount, chunks)
}

func qwenImagePlan(
	family string,
	history bool,
	width int,
	render func([]string, []imagePromptItem) string,
) imagePromptPlan {
	return imagePromptPlan{
		Family: family, Placeholder: Qwen3VLImagePad, PlaceholderLabel: "image placeholder",
		History: history, EmbeddingWidth: width, Render: render, Positions: qwenImagePromptPositions,
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
		Count: int(output.Embeddings.Shape.Dims[1]),
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
