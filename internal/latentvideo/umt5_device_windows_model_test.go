//go:build windows && modeltest

package latentvideo

import (
	"math"
	"path/filepath"
	"testing"
	"time"

	cudatest "overgo/internal/cuda/testutil"
	"overgo/internal/testcheck"
)

// TestUMT5EncodesOnTheCardAsTheHostDoes holds the umT5 encoder on the card to the reference the host encoder is held
// to: on 2026-11-08 every new prompt streamed the 11 GB checkpoint through the CPU. Both g1 contexts, the fox prompt's
// and the negative prompt's, conditioned through the card meet the cross-engine gates the host meets against the
// reference CUDA BF16 run (goldenCrossEngineMaxTolerance, goldenCrossEngineMeanTolerance, declared 2026-08-09); the
// card holds the encoder from the first prompt to the second, and no pass streams the checkpoint. One block on the card
// agrees with the host's block at cosine 0.99998 a token; over 24 blocks two BF16 engines drift by accumulation order,
// which is what those gates allow.
func TestUMT5EncodesOnTheCardAsTheHostDoes(t *testing.T) {
	cudatest.Require(t)
	g, fixtureDir := loadG1(t)
	modelDir := wanModelDir(t)
	spec := TextConditioningSpec{
		TokenizerDir:      filepath.Join(modelDir, "google", "umt5-xxl"),
		EncoderCheckpoint: filepath.Join(modelDir, "models_t5_umt5-xxl-enc-bf16.pth"),
		ProjectionDir:     modelDir, SequenceLength: g.TextLen,
		RelativeMaxDistance: referenceEncoderPolicy.RelativeMaxDistance, NormEps: referenceEncoderPolicy.NormEps,
		Card: true,
	}
	var sourceBytes int64
	weights := testcheck.Must(func() (projectionWeights, error) {
		weights, read, err := loadProjectionWeights(spec.ProjectionDir)
		sourceBytes = read
		return weights, err
	}())(t)
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
		started := time.Now()
		got := testcheck.Must(textConditioningWithWeights(t.Context(), spec, branch.prompt, weights, sourceBytes))(t)
		if got.Encoder.Engine != "card" || len(got.Context) != len(want) || got.TokenCount != branch.golden.Stats.TokenCount {
			t.Fatalf("%s: engine %q, %d values of %d tokens, want the card's %d values of %d", branch.name, got.Encoder.Engine, len(got.Context), got.TokenCount, len(want), branch.golden.Stats.TokenCount)
		}
		var worst, sum float64
		for index := range want {
			difference := math.Abs(float64(got.Context[index]) - float64(want[index]))
			worst, sum = max(worst, difference), sum+difference
		}
		mean := sum / float64(len(want))
		t.Logf("%s: %d tokens on the card in %s, max abs diff %.6g, mean abs diff %.6g", branch.name, got.TokenCount, time.Since(started), worst, mean)
		if worst > goldenCrossEngineMaxTolerance || mean > goldenCrossEngineMeanTolerance {
			t.Fatalf("%s: max abs diff %g and mean %g, want under %g and %g", branch.name, worst, mean, goldenCrossEngineMaxTolerance, goldenCrossEngineMeanTolerance)
		}
	}
	encoder := testcheck.Must(promptEncoderFor(spec))(t)
	if encoder.card == nil || encoder.passes != 0 {
		t.Fatalf("the encoder holds card %v after %d passes over the checkpoint, want the card held and none", encoder.card, encoder.passes)
	}
}
