package speechsynth

import (
	"context"
	"fmt"
	"math"
	"testing"

	"overgo/internal/tensor"
	"overgo/internal/testutil"
)

// generateFromConditioning prompts already-projected voice conditioning rows, [voiceFrames][d] as the reference's
// trace records them, and the text, then runs the live seeded loop the exported voice state's generation runs.
func generateFromConditioning(m *Model, ctx context.Context, voiceCond []float32, voiceFrames int, textIDs []int, p GenerateParams) (LatentBatch, []float64, error) {
	d := m.Dims.DModel
	if len(voiceCond) != voiceFrames*d {
		return LatentBatch{}, nil, fmt.Errorf("speechsynth: voice conditioning %d values != %d rows x %d", len(voiceCond), voiceFrames, d)
	}
	st := m.NewDecodeState(voiceFrames + len(textIDs) + p.MaxFrames + tensor.SingletonExtent)
	prompt := make([]float32, (voiceFrames+len(textIDs))*d)
	copy(prompt, voiceCond)
	if err := m.TextEmbedInto(prompt[voiceFrames*d:], textIDs); err != nil {
		return LatentBatch{}, nil, err
	}
	m.AppendForward(st, prompt, voiceFrames+len(textIDs))
	return m.generateFromState(ctx, st, p)
}

// TestGenerationGolden gates the backbone-scoped generation loop (ladder
// g7): with the reference's recorded noise injected per frame, the loop
// must reproduce every generated latent (noise + flow out) and EOS logit.
// Flow calls 0 and 1 belong to the voice-state pass; generation consumes
// calls 2..2+frames-1, mirroring the reference port's gate.
func TestGenerationGolden(t *testing.T) {
	m := loadArtifactModel(t)
	g6 := loadFixture[g6Golden](t, "g6_backbone.json")
	g7 := loadFixture[g7Golden](t, "g7_generation.json")
	g1 := loadFixture[struct {
		IDs []int `json:"ids"`
	}](t, "g1_tokenizer.json")

	voiceCond := f32of(g6.TransformerCalls[0].In.Values)
	tv := g6.TransformerCalls[0].In.Shape[1]
	nFrames := g7.NQuantizerCalls
	if len(g7.FlowCalls) < 2+nFrames || len(g7.EOSLogits) < 2+nFrames {
		t.Fatalf("golden has %d flow calls / %d eos for %d frames", len(g7.FlowCalls), len(g7.EOSLogits), nFrames)
	}
	noises := make([][]float32, nFrames)
	for i := range noises {
		noises[i] = f32of(g7.FlowCalls[2+i].Noise.Values)
	}

	latents, eos, err := generateFromConditioning(m, t.Context(), voiceCond, tv, g1.IDs, GenerateParams{
		MaxFrames:    nFrames,
		EOSThreshold: math.Inf(1),
		NoiseAt:      func(step int, dst []float32) { copy(dst, noises[step]) },
	})
	if err != nil {
		t.Fatal(err)
	}
	if latents.Frames != nFrames {
		t.Fatalf("generated %d frames, want %d", latents.Frames, nFrames)
	}

	worstLatent, worstEOS := 0.0, 0.0
	for i := range nFrames {
		fc := g7.FlowCalls[2+i]
		want := make([]float64, len(fc.Out.Values))
		for j := range want {
			want[j] = fc.Noise.Values[j] + fc.Out.Values[j]
		}
		got := latents.Values[i*latents.Width : (i+1)*latents.Width]
		if diff := testutil.MaxAbsDiff(got, want); diff > worstLatent {
			worstLatent = diff
		}
		if diff := testutil.MaxAbsDiff(got, want); diff > tolGenLatent {
			t.Fatalf("frame %d latent max abs diff %g > %g", i, diff, tolGenLatent)
		}
		if d := math.Abs(eos[i] - g7.EOSLogits[2+i]); d > worstEOS {
			worstEOS = d
		}
		if d := math.Abs(eos[i] - g7.EOSLogits[2+i]); d > tolEOSLogit {
			t.Fatalf("frame %d eos logit %g vs %g (diff %g > %g)", i, eos[i], g7.EOSLogits[2+i], d, tolEOSLogit)
		}
	}
	t.Logf("generation: %d frames, worst latent max abs diff %.6e (gate %.0e), worst eos diff %.6e (gate %.0e)",
		nFrames, worstLatent, tolGenLatent, worstEOS, tolEOSLogit)
}

// The codec-boundary refusal test is GONE by design: the mimi decoder is
// ported, so latent-to-pcm now has real golden gates — see
// TestSpeakE2EProducesReferenceAudio and TestLatentsToPCMContracts
// (codec_golden_test.go), mirroring rung-1's refusal-to-golden turn.

// TestGenerateLatentsInputContracts pins the named refusals.
func TestGenerateLatentsInputContracts(t *testing.T) {
	m := loadArtifactModel(t)
	noise := func(int, []float32) {}
	voice := &VoiceState{Frames: 1, K: make([][]float32, m.Dims.Layers), V: make([][]float32, m.Dims.Layers)}
	if _, _, err := m.GenerateLatentsWithVoice(t.Context(), voice, []int{0}, GenerateParams{MaxFrames: 0, NoiseAt: noise}); err == nil {
		t.Fatal("want refusal on MaxFrames <= 0")
	}
	if _, _, err := m.GenerateLatentsWithVoice(t.Context(), voice, []int{0}, GenerateParams{MaxFrames: 1}); err == nil {
		t.Fatal("want refusal on missing noise source")
	}
	if _, _, err := m.GenerateLatentsWithVoice(t.Context(), voice, nil, GenerateParams{MaxFrames: 1, NoiseAt: noise}); err == nil {
		t.Fatal("want refusal on voiced generation without text")
	}
	if _, _, err := m.GenerateLatentsWithVoice(t.Context(), &VoiceState{Frames: 1}, []int{0}, GenerateParams{MaxFrames: 1, NoiseAt: noise}); err == nil {
		t.Fatal("want refusal on a voice state of other layers")
	}
	if _, _, err := generateFromConditioning(m, t.Context(), make([]float32, m.Dims.DModel), 2, nil, GenerateParams{MaxFrames: 1, NoiseAt: noise}); err == nil {
		t.Fatal("want refusal on voice conditioning size mismatch")
	}
	if _, _, err := generateFromConditioning(m, t.Context(), nil, 0, []int{-1}, GenerateParams{MaxFrames: 1, NoiseAt: noise}); err == nil {
		t.Fatal("want refusal on out-of-table text id")
	}
}
