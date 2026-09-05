package projector

import (
	"context"
	"image"
	"testing"

	"overgo/internal/tensor"
	"overgo/internal/tensor/reference"
)

func TestGemma4TowerCausalMediaPrompts(t *testing.T) {
	for _, test := range []struct {
		name   string
		source gemmaPromptSource
		blocks bool
	}{
		{"tower", gemma4TowerPromptSource(&Gemma4TowerRunner{}), false},
		{"linear", gemma4PromptSource(&Gemma4Runner{}), true},
	} {
		t.Run(test.name, func(t *testing.T) {
			source := test.source
			source.width = 2
			value := reference.Value{Shape: tensor.MustShape(2, 2), Data: []float32{1, 2, 3, 4}}
			source.image = func(context.Context, image.Image) (Gemma4Output, error) { return Gemma4Output{Embeddings: value}, nil }
			source.audio = func(context.Context, []float32) (Gemma4AudioOutput, error) {
				return Gemma4AudioOutput{Embeddings: value}, nil
			}
			source.video = func(context.Context, []image.Image) (Gemma4VideoOutput, error) {
				return Gemma4VideoOutput{Embeddings: value, Frames: 2, TokensPerFrame: 1}, nil
			}
			picture := image.NewRGBA(image.Rect(0, 0, 1, 1))
			program := compileGemma4ImageProgram(source)
			images, err := executeImagePromptPlan(t.Context(), gemma4PromptTokenizer{}, []image.Image{picture, picture}, []string{"", "", "Count."}, program.Default, program.Encode)
			if err != nil {
				t.Fatal(err)
			}
			media := compileGemma4MediaProgram(source)
			mixed, err := media.History(t.Context(), gemma4PromptTokenizer{}, []MediaInput{{Kind: MediaImage, Image: picture}, {Kind: MediaAudio, Audio: []float32{1}}}, []string{"before", "between", "after"})
			if err != nil {
				t.Fatal(err)
			}
			video, err := media.Video(t.Context(), gemma4PromptTokenizer{}, []image.Image{picture, picture}, "", "Describe.", 1, false)
			if err != nil {
				t.Fatal(err)
			}
			for name, prompt := range map[string]MultimodalPrompt{"images": images, "mixed": mixed, "video": video} {
				if len(prompt.Embeddings) == 0 || len(prompt.TokenIDs) == 0 {
					t.Fatalf("%s omitted media", name)
				}
				if (len(prompt.AttentionBlocks) > 0) != test.blocks {
					t.Errorf("%s blocks=%v want bidirectional=%t", name, prompt.AttentionBlocks, test.blocks)
				}
			}
		})
	}
}
