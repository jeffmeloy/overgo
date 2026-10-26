//go:build windows

package inference

import (
	"slices"
	"testing"

	cudatest "overgo/internal/cuda/testutil"
	"overgo/internal/recipe"
	"overgo/internal/sampling"
	"overgo/internal/tokenizer"
)

// TestSampledVerifyWindowsKeepTheDistribution holds a sampled request's verify windows to plain sampled decoding
// (Strata's rule, src/core/verify.cpp run): the hermetic Flash-Next trunk with its draft layer, decoding at
// temperature one with the top-k and top-p cuts, speculating and decoding plainly from the same seed. Each draw is
// keyed by its position, so a seed draws the same uniforms either way, and a window's rows read the logits a plain
// step reads up to the span's rounding: a seed's text is the plain text unless a draw falls inside that rounding of a
// cut. The bound, written before the change: at least three seeds in four give the plain text token for token, the
// tokens drawn over every seed are distributed as the plain ones within a total variation of one tenth, and the
// windows drafted and kept drafts.
func TestSampledVerifyWindowsKeepTheDistribution(t *testing.T) {
	requireIntegration(t)
	cudatest.Require(t)
	const (
		seeds     = 32
		generated = 24
		agreeing  = 3
		of        = 4
		variation = 0.1
	)
	trunk := writeHermeticFlashNextGGUF(t)
	prompt := []tokenizer.TokenID{4, 9, 17}
	for _, copyTrunk := range []bool{false, true} {
		runner, err := openFixtureRunnerWithResidency(trunk, OpenOptions{Draft: writeHermeticFlashNextMTPGGUF(t, trunk, copyTrunk)}, recipe.ResidencyDeviceNative)
		if err != nil {
			t.Fatal(err)
		}
		var drafted, kept, same int
		counts := [2]map[tokenizer.TokenID]int{{}, {}}
		for seed := range int64(seeds) {
			generate := func(plain bool) []tokenizer.TokenID {
				t.Helper()
				sampler, err := sampling.New(sampling.Config{Temperature: 1, TopK: 20, TopP: 0.95, Seed: seed + 1})
				if err != nil {
					t.Fatal(err)
				}
				options := GenerateOptions{
					MaxNewTokens: generated, Sampler: sampler, PromptTokenIDs: prompt, ContinueAfterEOG: true, PlainDecode: plain,
				}
				if !plain {
					options.OnDraftSpan = func(count, accepted int) { drafted, kept = drafted+count, kept+accepted }
				}
				ids, _, err := runner.Generate(t.Context(), "", options)
				if err != nil {
					_ = runner.Close()
					t.Fatal(err)
				}
				return ids[len(prompt):]
			}
			speculated, plain := generate(false), generate(true)
			if slices.Equal(speculated, plain) {
				same++
			}
			for index, sequence := range [][]tokenizer.TokenID{speculated, plain} {
				for _, token := range sequence {
					counts[index][token]++
				}
			}
		}
		if err := runner.Close(); err != nil {
			t.Fatal(err)
		}
		distance := 0.0
		for token := range tokenizer.TokenID(flashNextVocab) {
			difference := float64(counts[0][token]-counts[1][token]) / float64(seeds*generated)
			distance += max(difference, -difference) / 2
		}
		t.Logf("copy=%t: %d of %d seeds give the plain text, the tokens %.3g apart in total variation, %d of %d drafts kept",
			copyTrunk, same, seeds, distance, kept, drafted)
		if same*of < seeds*agreeing || distance > variation || drafted == 0 || kept == 0 {
			t.Fatalf("copy=%t: %d of %d seeds give the plain text, the tokens %.3g apart, %d of %d drafts kept: past the bound",
				copyTrunk, same, seeds, distance, kept, drafted)
		}
	}
}
