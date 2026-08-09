package speechsynth

import (
	"math"
	"strings"
	"testing"
)

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

	latents, eos, err := m.GenerateLatents(voiceCond, tv, g1.IDs, GenerateParams{
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
	for i := 0; i < nFrames; i++ {
		fc := g7.FlowCalls[2+i]
		want := make([]float64, len(fc.Out.Values))
		for j := range want {
			want[j] = fc.Noise.Values[j] + fc.Out.Values[j]
		}
		got := latents.Values[i*latents.Width : (i+1)*latents.Width]
		if diff := maxAbsDiff(got, want); diff > worstLatent {
			worstLatent = diff
		}
		if diff := maxAbsDiff(got, want); diff > tolGenLatent {
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

// TestLatentsToPCMRefusesAtCodecBoundary pins the codec-boundary refusal:
// until the mimi decoder slice lands, latent-to-pcm must refuse loudly.
func TestLatentsToPCMRefusesAtCodecBoundary(t *testing.T) {
	var m Model
	pcm, err := m.LatentsToPCM(LatentBatch{Values: make([]float32, 32), Frames: 1, Width: 32})
	if err == nil || pcm != nil {
		t.Fatal("want loud refusal at the codec boundary")
	}
	if !strings.Contains(err.Error(), "codec") || !strings.Contains(err.Error(), "refusing") {
		t.Fatalf("refusal must name the missing codec slice: %v", err)
	}
}

// TestGenerateLatentsInputContracts pins the named refusals.
func TestGenerateLatentsInputContracts(t *testing.T) {
	m := loadArtifactModel(t)
	noise := func(int, []float32) {}
	if _, _, err := m.GenerateLatents(make([]float32, m.Dims.DModel), 2, nil, GenerateParams{MaxFrames: 1, NoiseAt: noise}); err == nil {
		t.Fatal("want refusal on voice conditioning size mismatch")
	}
	if _, _, err := m.GenerateLatents(nil, 0, nil, GenerateParams{MaxFrames: 0, NoiseAt: noise}); err == nil {
		t.Fatal("want refusal on MaxFrames <= 0")
	}
	if _, _, err := m.GenerateLatents(nil, 0, nil, GenerateParams{MaxFrames: 1}); err == nil {
		t.Fatal("want refusal on missing noise source")
	}
	if _, _, err := m.GenerateLatents(nil, 0, []int{-1}, GenerateParams{MaxFrames: 1, NoiseAt: noise}); err == nil {
		t.Fatal("want refusal on out-of-table text id")
	}
}
