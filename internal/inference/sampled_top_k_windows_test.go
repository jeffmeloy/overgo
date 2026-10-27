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

// TestCardTopKDrawsAsTheHostSampler holds a sampled request's draws from the card's top K of each row
// (GenerateOptions.DeviceTopK) to the host sampler's draws from the whole row: on the hermetic Flash-Next with its
// draft layer, which steps and verifies its spans, and on the fixture at Strata's geometry, which decodes through
// windows, a request at temperature one with the top-k and top-p cuts decodes the same text from the same seed either
// way, plainly and speculating, and draws from the top K with no row's logits brought to the host.
func TestCardTopKDrawsAsTheHostSampler(t *testing.T) {
	requireIntegration(t)
	cudatest.Require(t)
	const seeds, generated = 8, 24
	prompt := []tokenizer.TokenID{4, 9, 17}
	hermetic := writeHermeticFlashNextGGUF(t)
	windows := writeFlashNextFixture(t, strataFlashNextWindows)
	for _, fixture := range []struct {
		name, trunk, draft string
		windows            bool
	}{
		{"hermetic", hermetic, writeHermeticFlashNextMTPGGUF(t, hermetic, false), false},
		{"windows", windows, writeFlashNextMTPFixture(t, windows, false, strataFlashNextWindows), true},
	} {
		runner, err := openFixtureRunnerWithResidency(fixture.trunk, OpenOptions{Draft: fixture.draft}, recipe.ResidencyDeviceNative)
		if err != nil {
			t.Fatal(err)
		}
		if runner.windowsDecode() != fixture.windows {
			_ = runner.Close()
			t.Fatalf("%s: the runner decodes through windows %t", fixture.name, runner.windowsDecode())
		}
		drafted := 0
		for seed := range int64(seeds) {
			generate := func(plain, card bool) []tokenizer.TokenID {
				t.Helper()
				sampler, err := sampling.New(sampling.Config{Temperature: 1, TopK: 20, TopP: 0.95, Seed: seed + 1})
				if err != nil {
					t.Fatal(err)
				}
				rows := 0
				options := GenerateOptions{
					MaxNewTokens: generated, Sampler: sampler, PromptTokenIDs: prompt, ContinueAfterEOG: true,
					PlainDecode: plain, DeviceTopK: card,
					OnToken: func(event TokenEvent) error {
						if event.Logits != nil {
							rows++
						}
						return nil
					},
				}
				if !plain {
					options.OnDraftSpan = func(count, _ int) { drafted += count }
				}
				ids, _, err := runner.Generate(t.Context(), "", options)
				if err != nil {
					_ = runner.Close()
					t.Fatal(err)
				}
				if card == (rows != 0) {
					_ = runner.Close()
					t.Fatalf("%s seed %d plain=%t top-K=%t: %d of %d tokens drawn from a row brought to the host", fixture.name, seed, plain, card, rows, generated)
				}
				return ids[len(prompt):]
			}
			for _, plain := range []bool{true, false} {
				if whole, card := generate(plain, false), generate(plain, true); !slices.Equal(card, whole) {
					_ = runner.Close()
					t.Fatalf("%s seed %d plain=%t: the card's top K drew %v, the whole rows %v", fixture.name, seed, plain, card, whole)
				}
			}
		}
		if err := runner.Close(); err != nil {
			t.Fatal(err)
		}
		if drafted == 0 {
			t.Fatalf("%s: no window drafted", fixture.name)
		}
		t.Logf("%s: %d seeds draw the same text from the top K as from the whole rows, plainly and over %d drafts", fixture.name, seeds, drafted)
	}
}
