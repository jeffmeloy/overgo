//go:build windows

package inference

import (
	"slices"
	"testing"

	"overgo/internal/sampling"
	"overgo/internal/tokenizer"
)

func TestHermeticCUDAFixedBudgetEOGPreservesDeviceGreedy(t *testing.T) {
	runner := openHermeticScoringRunner(t)
	prompt := []tokenizer.TokenID{1, 4, 5}
	const outputTokens = 4
	generate := func(continueAfterEOG bool) []tokenizer.TokenID {
		t.Helper()
		sampler, err := sampling.New(sampling.Config{Temperature: 0})
		if err != nil {
			t.Fatal(err)
		}
		ids, _, err := runner.Generate(t.Context(), "", GenerateOptions{
			MaxNewTokens: outputTokens, Sampler: sampler, DeviceGreedy: true,
			ContinueAfterEOG: continueAfterEOG, PromptTokenIDs: prompt,
			OnToken: func(event TokenEvent) error {
				if len(event.Logits) != 0 {
					t.Error("fixed budget fell back to host logits")
				}
				return nil
			},
		})
		if err != nil {
			t.Fatal(err)
		}
		return ids
	}
	reference := generate(true)
	if len(reference) != len(prompt)+outputTokens {
		t.Fatalf("incomplete reference: %v", reference)
	}
	// Declare the first greedy winner EOG in this hermetic vocabulary without
	// changing weights/logits. This reliably exercises stopping on the device.
	runner.vocab.EOS = reference[len(prompt)]
	natural := generate(false)
	if !slices.Equal(natural, reference[:len(prompt)+1]) {
		t.Fatalf("natural EOS ignored: %v", natural)
	}
	guard := generate(true)
	if !slices.Equal(guard, reference) {
		t.Fatalf("continued greedy tokens=%v want=%v", guard, reference)
	}
}
