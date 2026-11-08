//go:build integration

package latentvideo

import (
	"math"
	"os"
	"path/filepath"
	"testing"

	"overgo/internal/testcheck"
	"overgo/internal/testskip"
)

// TestTextConditioningGoldenParity: full pipeline vs the committed g1 cond
// and uncond contexts (fox prompt / negative prompt).
func TestTextConditioningGoldenParity(t *testing.T) {
	testskip.Short(t, "streams the 11.4GB encoder checkpoint twice")
	g, fixtureDir := loadG1(t)
	modelDir := wanModelDir(t)
	spec := TextConditioningSpec{
		TokenizerDir:      filepath.Join(modelDir, "google", "umt5-xxl"),
		EncoderCheckpoint: filepath.Join(modelDir, "models_t5_umt5-xxl-enc-bf16.pth"),
		ProjectionDir:     modelDir,
		SequenceLength:    g.TextLen,
		// Published encoder-config facts (relative_attention_max_distance,
		// layer_norm_epsilon) — the checkpoint carries no config JSON.
		RelativeMaxDistance: referenceEncoderPolicy.RelativeMaxDistance,
		NormEps:             referenceEncoderPolicy.NormEps,
	}
	if _, err := os.Stat(spec.EncoderCheckpoint); err != nil {
		t.Skipf("UNAVAILABLE: encoder checkpoint absent: %v", err)
	}
	weights, sourceBytes, err := loadProjectionWeights(spec.ProjectionDir)
	testcheck.NoError(t, err)
	// The streaming measures below are of a pass over the checkpoint: a prompt
	// another test of this process conditioned would be given its kept rows.
	forgetPromptEncoders(t)
	for _, branch := range []struct {
		name   string
		prompt string
		golden g1Branch
	}{
		{"conditional", g.Prompt, g.Conditional},
		{"unconditional", g.NegativePrompt, g.Unconditional},
	} {
		want := loadG1Tensor(t, fixtureDir, branch.golden.Tensor)
		got, err := textConditioningWithWeights(t.Context(), spec, branch.prompt, weights, sourceBytes)
		if err != nil {
			t.Fatalf("%s: %v", branch.name, err)
		}
		if got.Rows != g.TextLen || got.Dim != g.Dim || len(got.Context) != len(want) {
			t.Fatalf("%s shape rows=%d dim=%d len=%d want %dx%d", branch.name, got.Rows, got.Dim, len(got.Context), g.TextLen, g.Dim)
		}
		if got.TokenCount != branch.golden.Stats.TokenCount {
			t.Fatalf("%s token count=%d want %d", branch.name, got.TokenCount, branch.golden.Stats.TokenCount)
		}
		if got.Encoder.Layers != branch.golden.Stats.EncoderLayers {
			t.Fatalf("%s encoder layers=%d want %d", branch.name, got.Encoder.Layers, branch.golden.Stats.EncoderLayers)
		}
		// Streaming evidence: whole-checkpoint residency would show as peak
		// heap at checkpoint scale (11.4GB); the bound is one F32-decoded
		// block plus activations.
		if got.Encoder.MaxBlockWeightBytes <= 0 || got.Encoder.BlockWeightBytes <= got.Encoder.MaxBlockWeightBytes {
			t.Fatalf("%s streaming stats degenerate: %+v", branch.name, got.Encoder)
		}
		if got.Encoder.PeakHeapAllocBytes > 4<<30 {
			t.Fatalf("%s peak heap %d exceeds streamed bound", branch.name, got.Encoder.PeakHeapAllocBytes)
		}
		var maxDiff, sumDiff, maxTokenDiff, maxPadDiff float64
		for i := range want {
			d := math.Abs(float64(got.Context[i]) - float64(want[i]))
			sumDiff += d
			maxDiff = max(maxDiff, d)
			if i < got.TokenCount*g.Dim {
				maxTokenDiff = max(maxTokenDiff, d)
			} else {
				maxPadDiff = max(maxPadDiff, d)
			}
		}
		meanDiff := sumDiff / float64(len(want))
		t.Logf("%s tokens=%d engine=%s wall=%.1fs max_abs_diff=%.6g (token_rows=%.6g pad_rows=%.6g) mean_abs_diff=%.6g peak_heap_alloc_bytes=%d max_block_weight_bytes=%d total_block_weight_bytes=%d",
			branch.name, got.TokenCount, got.Encoder.Engine, got.Encoder.EncoderWallSec, maxDiff, maxTokenDiff, maxPadDiff, meanDiff, got.Encoder.PeakHeapAllocBytes, got.Encoder.MaxBlockWeightBytes, got.Encoder.BlockWeightBytes)
		if maxDiff > goldenCrossEngineMaxTolerance {
			t.Fatalf("%s max abs diff=%g > %g", branch.name, maxDiff, goldenCrossEngineMaxTolerance)
		}
		if meanDiff > goldenCrossEngineMeanTolerance {
			t.Fatalf("%s mean abs diff=%g > %g", branch.name, meanDiff, goldenCrossEngineMeanTolerance)
		}
	}
}
