package projector

import (
	"context"
	"image"
	"slices"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/modelartifact"
	"overgo/internal/tensor"
	"overgo/internal/tensor/reference"
	"overgo/internal/testutil"
)

func TestGemma4TowerCausalMediaPrompts(t *testing.T) {
	causal, vision := fixtureMediaAttention(t, "causal"), fixtureMediaAttention(t, "vision")
	for _, test := range []struct {
		name   string
		source gemmaPromptSource
		blocks bool
	}{
		{"tower causal", gemma4TowerPromptSource(&Gemma4TowerRunner{mediaPreprocessOwner: causal}), false},
		{"tower vision", gemma4TowerPromptSource(&Gemma4TowerRunner{mediaPreprocessOwner: vision}), true},
		{"linear causal", gemma4PromptSource(&Gemma4Runner{mediaPreprocessOwner: causal}), false},
		{"linear vision", gemma4PromptSource(&Gemma4Runner{mediaPreprocessOwner: vision}), true},
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
				if test.blocks {
					indices := prompt.EmbeddingTokenIndices
					var expected []AttentionBlock
					switch name {
					case "images":
						expected = []AttentionBlock{{Start: indices[0], End: indices[1] + 1}, {Start: indices[2], End: indices[3] + 1}}
					case "mixed":
						expected = []AttentionBlock{{Start: indices[0], End: indices[1] + 1}}
					case "video":
						expected = []AttentionBlock{{Start: indices[0], End: indices[0] + 1}, {Start: indices[1], End: indices[1] + 1}}
					}
					if !slices.Equal(prompt.AttentionBlocks, expected) {
						t.Errorf("%s attention crosses image boundaries or includes text/audio: got %v want %v", name, prompt.AttentionBlocks, expected)
					}
				}
			}
		})
	}
	for _, source := range []gemmaPromptSource{gemma4TowerPromptSource(&Gemma4TowerRunner{}), gemma4PromptSource(&Gemma4Runner{})} {
		if source.policyErr == nil {
			t.Fatal("undeclared attention silently selected a mask")
		}
		program := compileGemma4ImageProgram(source)
		if _, err := program.Encode(t.Context(), image.NewRGBA(image.Rect(0, 0, 1, 1))); err == nil {
			t.Fatal("missing declaration reached image encoder")
		}
	}
}

func fixtureMediaAttention(t *testing.T, attention string) mediaPreprocessOwner {
	t.Helper()
	config, err := modelartifact.NewModelConfigDocument(
		testutil.ArtifactID(t, artifact.KindModel, "prompt-attention-model"), nil,
		&modelartifact.GenerationEssentials{ImageAttention: attention},
		[]modelartifact.ConfigSource{{Name: "config.json", SHA256: strings.Repeat("a", 64)}},
	)
	if err != nil {
		t.Fatal(err)
	}
	profile, err := (MediaPreprocessProfile{Version: artifact.InitialDocumentVersion}).BindModelConfig(config)
	if err != nil {
		t.Fatal(err)
	}
	return mediaPreprocessOwner{preprocess: profile}
}
