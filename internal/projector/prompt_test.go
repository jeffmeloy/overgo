package projector

import (
	"context"
	"image"
	"slices"
	"testing"

	"overgo/internal/tensor"
)

func TestMediaHistoryChunkContract(t *testing.T) {
	imageChunk := NewImageMediaInput(image.NewRGBA(image.Rect(0, 0, 1, 1)))
	audioChunk := NewAudioMediaInput([]float32{0})
	if err := validateMediaHistoryInputs(
		gemma4PromptTokenizer{}, []MediaInput{imageChunk, audioChunk}, []string{"a", "b", "c"}, "test",
	); err != nil {
		t.Fatal(err)
	}
	invalid := []struct {
		name  string
		media []MediaInput
		text  []string
	}{
		{name: "empty", text: []string{"x"}},
		{name: "text count", media: []MediaInput{imageChunk}, text: []string{"x"}},
		{name: "nil image", media: []MediaInput{{Kind: MediaImage}}, text: []string{"", ""}},
		{name: "image audio", media: []MediaInput{{Kind: MediaImage, Image: imageChunk.Image, Audio: []float32{0}}}, text: []string{"", ""}},
		{name: "empty audio", media: []MediaInput{{Kind: MediaAudio}}, text: []string{"", ""}},
		{name: "audio image", media: []MediaInput{{Kind: MediaAudio, Audio: []float32{0}, Image: imageChunk.Image}}, text: []string{"", ""}},
		{name: "kind", media: []MediaInput{{Kind: 99}}, text: []string{"", ""}},
	}
	for _, test := range invalid {
		t.Run(test.name, func(t *testing.T) {
			if err := validateMediaHistoryInputs(
				gemma4PromptTokenizer{}, test.media, test.text, "test",
			); err == nil {
				t.Fatal("invalid media history accepted")
			}
		})
	}
}

func TestMixedMediaPromptPlan(t *testing.T) {
	media := []MediaInput{
		NewImageMediaInput(image.NewRGBA(image.Rect(0, 0, 1, 1))),
		NewAudioMediaInput([]float32{1}),
	}
	encode := func(count int) func(context.Context, MediaInput) (imagePromptItem, error) {
		return func(context.Context, MediaInput) (imagePromptItem, error) {
			return imagePromptItem{Embeddings: make([]float32, count*2), Count: count}, nil
		}
	}
	prompt, err := executeMixedMediaPromptPlan(
		context.Background(), gemma4PromptTokenizer{}, media, []string{"a", "b", "c"},
		mixedMediaPromptPlan{
			Family: "test", AddSpecial: true, EmbeddingWidth: 2, PromptLabel: "test history", Render: renderMixedMediaHistory,
			Kinds: map[MediaKind]mixedMediaKindPlan{
				MediaImage: {Placeholder: "<|image|>", PlaceholderLabel: "image placeholder", Open: "<i>", Close: "</i>", Attention: true, Encode: encode(2)},
				MediaAudio: {Placeholder: "<|audio|>", PlaceholderLabel: "audio placeholder", Open: "<a>", Close: "</a>", Encode: encode(1)},
			},
		},
	)
	if err != nil {
		t.Fatal(err)
	}
	if prompt.EmbeddingWidth != 2 || prompt.EmbeddingStart != 4 ||
		!slices.Equal(prompt.EmbeddingTokenIndices, []uint32{4, 5, 14}) ||
		!slices.Equal(prompt.AttentionBlocks, []AttentionBlock{{Start: 4, End: 6}}) {
		t.Fatalf("mixed-media prompt = %+v", prompt)
	}
}

func TestCompiledImagePromptPrograms(t *testing.T) {
	plan := delimitedImagePromptPlan(
		"fixture", "<|image|>", "fixture image placeholder", true, 2, "<i>", "</i>",
	)
	prompt, err := executeImagePromptPlan(
		context.Background(), gemma4PromptTokenizer{},
		[]image.Image{image.NewRGBA(image.Rect(0, 0, 1, 1))}, []string{"a", "b"}, plan,
		func(context.Context, image.Image) (imagePromptItem, error) {
			return imagePromptItem{Embeddings: make([]float32, 4), Count: 2, RunCount: 2}, nil
		},
	)
	if err != nil {
		t.Fatal(err)
	}
	if prompt.EmbeddingWidth != 2 || len(prompt.EmbeddingTokenIndices) != 2 || len(prompt.Embeddings) != 4 {
		t.Fatalf("compiled image prompt = %+v", prompt)
	}
}

func TestQwen35VideoPromptText(t *testing.T) {
	prompt := Qwen35VideoPromptText("", "Question", 2, 2, 24, true)
	want := "<|im_start|>user\n<|vision_start|>" +
		"<0.0 seconds><|vision_start|><|video_pad|><|video_pad|><|vision_end|>" +
		"<0.1 seconds><|vision_start|><|video_pad|><|video_pad|><|vision_end|>" +
		"<|vision_end|>Question<|im_end|>\n<|im_start|>assistant\n<think>\n"
	if prompt != want {
		t.Fatalf("video prompt = %q, want %q", prompt, want)
	}
}

func TestCompileMediaPromptRuns(t *testing.T) {
	runs, err := compileMediaPromptRuns(gemma4PromptTokenizer{}, mediaPromptRunPlan{
		Prompt: "a<|video|><|video|>b<|video|><|video|>", Placeholder: "<|video|>",
		Runs: 2, TokensPerRun: 2, PromptLabel: "video prompt",
		PlaceholderLabel: "video placeholder", RunsLabel: "video prompt",
	})
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(runs.Starts, []int{1, 4}) ||
		!slices.Equal(runs.Indices, []uint32{1, 2, 4, 5}) {
		t.Fatalf("media runs = starts %v indices %v", runs.Starts, runs.Indices)
	}
}

func TestSpatialPositionsGrid2D(t *testing.T) {
	positions, err := compileSpatialPositions(10, positionGrid2D, []spatialPositionChunk{{
		Start: 2, Extents: [tensor.TripleExtent]int{2, 2},
	}})
	if err != nil {
		t.Fatal(err)
	}
	want := [4][]uint32{
		{0, 1, 2, 2, 2, 2, 4, 5, 6, 7},
		{0, 1, 2, 2, 3, 3, 4, 5, 6, 7},
		{0, 1, 2, 3, 2, 3, 4, 5, 6, 7},
		{0, 1, 0, 0, 0, 0, 4, 5, 6, 7},
	}
	for axis := range positions {
		for index := range positions[axis] {
			if positions[axis][index] != want[axis][index] {
				t.Fatalf("axis %d position %d = %d, want %d", axis, index, positions[axis][index], want[axis][index])
			}
		}
	}
}

func TestSpatialPositionsMultipleGrid2D(t *testing.T) {
	positions, err := compileSpatialPositions(12, positionGrid2D, []spatialPositionChunk{
		{Start: 2, Extents: [tensor.TripleExtent]int{1, 2}},
		{Start: 8, Extents: [tensor.TripleExtent]int{1, 2}},
	})
	if err != nil {
		t.Fatal(err)
	}
	want := [4][]uint32{
		{0, 1, 2, 2, 4, 5, 6, 7, 8, 8, 10, 11},
		{0, 1, 2, 2, 4, 5, 6, 7, 8, 8, 10, 11},
		{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11},
		{0, 1, 0, 0, 4, 5, 6, 7, 0, 0, 10, 11},
	}
	for axis := range positions {
		for index := range positions[axis] {
			if positions[axis][index] != want[axis][index] {
				t.Fatalf("axis %d position %d = %d, want %d", axis, index, positions[axis][index], want[axis][index])
			}
		}
	}
}

func TestSpatialPositionsVariableGrid2D(t *testing.T) {
	positions, err := compileSpatialPositions(11, positionGrid2D, []spatialPositionChunk{
		{Start: 2, Extents: [tensor.TripleExtent]int{2, 2}},
		{Start: 7, Extents: [tensor.TripleExtent]int{1, 2}},
	})
	if err != nil {
		t.Fatal(err)
	}
	want := [4][]uint32{
		{0, 1, 2, 2, 2, 2, 4, 5, 5, 7, 8},
		{0, 1, 2, 2, 3, 3, 4, 5, 5, 7, 8},
		{0, 1, 2, 3, 2, 3, 4, 5, 6, 7, 8},
		{0, 1, 0, 0, 0, 0, 4, 0, 0, 7, 8},
	}
	for axis := range positions {
		if !slices.Equal(positions[axis], want[axis]) {
			t.Fatalf("axis %d positions = %v, want %v", axis, positions[axis], want[axis])
		}
	}
	if _, err := compileSpatialPositions(4, positionGrid2D, []spatialPositionChunk{{
		Start: 2, Extents: [tensor.TripleExtent]int{2, 2},
	}}); err == nil {
		t.Fatal("out-of-range variable chunk accepted")
	}
}
