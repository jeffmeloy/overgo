//go:build windows

package inference

import (
	"fmt"
	"slices"
	"testing"

	cudatest "overgo/internal/cuda/testutil"
	"overgo/internal/recipe"
	"overgo/internal/sampling"
	"overgo/internal/tokenizer"
)

// TestSpeculationStopsAtTheContextLimit holds a speculating request to plain decoding's end at the context limit.
//   - a request whose prompt and output fill the context speculates to the plain text, its last spans drafting only the tokens left
//   - a request asking for more than the context holds ends where plain decoding ends, with the same tokens and error
//   - both for NextN's three drafts on the hermetic Qwen3.5 and for the attached MTP layer's one on the hermetic Flash-Next
func TestSpeculationStopsAtTheContextLimit(t *testing.T) {
	requireIntegration(t)
	cudatest.Require(t)
	greedy, err := sampling.New(sampling.Config{Temperature: 0})
	if err != nil {
		t.Fatal(err)
	}
	prompt := []tokenizer.TokenID{1, 4, 5}
	flashNext := writeHermeticFlashNextGGUF(t)
	for _, draft := range []struct {
		name    string
		path    string
		options OpenOptions
		context int
	}{
		{"nextn", writeHermeticQwen35GGUF(t), OpenOptions{}, int(hermeticContext)},
		{"mtp", flashNext, OpenOptions{Draft: writeHermeticFlashNextMTPGGUF(t, flashNext, true)}, int(flashNextContext)},
	} {
		runner, err := openFixtureRunnerWithResidency(draft.path, draft.options, recipe.ResidencyDeviceF32)
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(func() {
			if err := runner.Close(); err != nil {
				t.Error(err)
			}
		})
		fills := draft.context - len(prompt)
		for _, tokens := range []int{fills, draft.context} {
			// overdrafted counts the spans that drafted more tokens than the output had left after the span's own.
			emitted, drafted, overdrafted := 0, 0, 0
			generate := func(speculative bool) ([]tokenizer.TokenID, error) {
				ids, _, err := runner.Generate(t.Context(), "", GenerateOptions{
					MaxNewTokens: tokens, Sampler: greedy, DeviceGreedy: true, SpeculativeDecode: speculative, PlainDecode: !speculative,
					PromptTokenIDs: prompt, ContinueAfterEOG: true,
					OnToken: func(TokenEvent) error { emitted++; return nil },
					OnDraftSpan: func(count, _ int) {
						drafted += count
						if count > tokens-emitted-1 {
							overdrafted++
						}
					},
				})
				return ids, err
			}
			plain, plainErr := generate(false)
			if tokens == fills && (plainErr != nil || len(plain) != draft.context) {
				t.Fatalf("%s: plain decoding of %d tokens gave %d ids (%v), want the context's %d", draft.name, tokens, len(plain), plainErr, draft.context)
			}
			emitted = 0
			speculated, speculatedErr := generate(true)
			if !slices.Equal(speculated, plain) || fmt.Sprint(speculatedErr) != fmt.Sprint(plainErr) {
				t.Fatalf("%s: %d tokens speculated to %v (%v), plain decoding to %v (%v)", draft.name, tokens, speculated, speculatedErr, plain, plainErr)
			}
			if drafted == 0 || overdrafted != 0 {
				t.Fatalf("%s: %d tokens drafted %d, %d spans past the output left", draft.name, tokens, drafted, overdrafted)
			}
		}
	}
}
