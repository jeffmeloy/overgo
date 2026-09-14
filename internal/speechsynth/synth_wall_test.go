package speechsynth

import (
	"math"
	"slices"
	"testing"
	"time"
)

// TestSynthWall: matched-protocol wall measurement against the reference's
// research-tagged e2e test (adaptive extmodel TestSpeechFlowGenerateE2E).
// That test's reported elapsed covers artifact resolve+load, the golden-text
// generation loop, latent->PCM codec decode, and parity asserts; the parity
// match here is: load once (reported separately), then median-of-5 warm
// walls for the SAME golden synth — voice conditioning + golden text ids +
// recorded noise -> latents -> PCM. Numbers land in docs/plan.json
// (rung7-pockettts performance-leg); this test keeps the protocol
// reproducible.
const synthWallRuns = 5

func TestSynthWall(t *testing.T) {
	loadStart := time.Now()
	m := loadArtifactModel(t)
	t.Logf("load %.3fs", time.Since(loadStart).Seconds())

	g6 := loadFixture[g6Golden](t, "g6_backbone.json")
	g7 := loadFixture[g7Golden](t, "g7_generation.json")
	g1 := loadFixture[struct {
		IDs []int `json:"ids"`
	}](t, "g1_tokenizer.json")

	voiceCond := f32of(g6.TransformerCalls[0].In.Values)
	tv := g6.TransformerCalls[0].In.Shape[1]
	nFrames := g7.NQuantizerCalls
	noises := make([][]float32, nFrames)
	for i := range noises {
		noises[i] = f32of(g7.FlowCalls[2+i].Noise.Values)
	}

	backbone := make([]float64, synthWallRuns)
	codec := make([]float64, synthWallRuns)
	total := make([]float64, synthWallRuns)
	for run := range total {
		start := time.Now()
		latents, _, err := m.GenerateLatents(t.Context(), voiceCond, tv, g1.IDs, GenerateParams{
			MaxFrames:    nFrames,
			EOSThreshold: math.Inf(1),
			NoiseAt:      func(step int, dst []float32) { copy(dst, noises[step]) },
		})
		if err != nil {
			t.Fatal(err)
		}
		backbone[run] = time.Since(start).Seconds()
		codecStart := time.Now()
		pcm, err := m.LatentsToPCM(t.Context(), latents)
		if err != nil {
			t.Fatal(err)
		}
		codec[run] = time.Since(codecStart).Seconds()
		total[run] = time.Since(start).Seconds()
		if len(pcm) == 0 {
			t.Fatal("empty pcm")
		}
	}
	report := func(name string, walls []float64) {
		sorted := append([]float64(nil), walls...)
		slices.Sort(sorted)
		t.Logf("%s: median %.4fs over %d runs (min %.4f max %.4f)",
			name, sorted[len(sorted)/2], synthWallRuns, sorted[0], sorted[len(sorted)-1])
	}
	report("backbone generate", backbone)
	report("codec decode", codec)
	report("synth total", total)
}
