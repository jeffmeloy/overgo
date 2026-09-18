package speechsynth

import (
	"math"
	"overgo/internal/jsonfile"
	"path/filepath"
	"testing"
)

func testSpeechCompletionBoundary(t *testing.T) {
	m := loadArtifactModel(t)
	g6 := loadFixture[g6Golden](t, "g6_backbone.json")
	g7 := loadFixture[g7Golden](t, "g7_generation.json")
	g1 := loadFixture[struct {
		IDs []int `json:"ids"`
	}](t, "g1_tokenizer.json")
	// Force EOS from the first step to independently exercise the reference
	// post-EOS boundary. Reaching that boundary requires observing its next step;
	// a budget that ends just before it must still report incomplete.
	for _, test := range []struct {
		name      string
		frames    int
		threshold float64
		want      bool
	}{
		{"budget before stopping boundary", 2, math.Inf(-1), false},
		{"observed stopping boundary", 3, math.Inf(-1), true},
		{"budget with no EOS", 2, math.Inf(1), false},
	} {
		t.Run(test.name, func(t *testing.T) {
			noises := 0
			latents, logits, err := m.GenerateLatents(t.Context(), f32of(g6.TransformerCalls[0].In.Values), g6.TransformerCalls[0].In.Shape[1], g1.IDs, GenerateParams{
				MaxFrames: test.frames, EOSThreshold: test.threshold, FramesAfterEOS: 2,
				NoiseAt: func(step int, dst []float32) { noises++; copy(dst, f32of(g7.FlowCalls[2+step].Noise.Values)) },
			})
			if err != nil {
				t.Fatal(err)
			}
			if latents.Complete != test.want || latents.Frames != 2 || len(logits) != 2 || noises != 2 {
				t.Fatalf("complete=%v frames=%d logits=%d noises=%d", latents.Complete, latents.Frames, len(logits), noises)
			}
		})
	}
}

func TestSpeechCompletion(t *testing.T) {
	t.Run("post-EOS boundary", testSpeechCompletionBoundary)
	t.Run("native output", testSpeechCompletionMetadata)
}
func testSpeechCompletionMetadata(t *testing.T) {
	synth := loadArtifactSynthesizer(t)
	var config struct {
		Mimi struct {
			FrameRate float64 `json:"frame_rate"`
		} `json:"mimi"`
	}
	if err := jsonfile.Decode(filepath.Join(artifactDir(t), "pockettts_config.json"), &config); err != nil {
		t.Fatal(err)
	}
	request := SynthesisRequest{Text: "Hello. This speech was generated locally.", Voice: "alba", Seed: 7}
	tokens, err := synth.tokenizer.Encode(request.Text)
	if err != nil {
		t.Fatal(err)
	}
	// Pinned local reference estimate used only by this fixture.
	budget := int(math.Ceil((float64(len(tokens))/3 + 2) * config.Mimi.FrameRate))
	for _, frames := range []int{1, budget} {
		request.MaxFrames = frames
		plan, err := synth.tokenize(request)
		if err != nil {
			t.Fatal(err)
		}
		latents, err := synth.generate(t.Context(), plan)
		if err != nil {
			t.Fatal(err)
		}
		audio, err := synth.decode(t.Context(), latents)
		if err != nil {
			t.Fatal(err)
		}
		if audio.Complete != latents.Complete || audio.Complete != (frames == budget) || len(audio.PCM) == 0 {
			t.Fatalf("completion metadata lost: frames=%d complete=%v", frames, audio.Complete)
		}
	}
	t.Log("bounded preview stays playable and explicitly incomplete; native EOS completion survives decoding as typed output metadata; no new request/UI setting")
}
