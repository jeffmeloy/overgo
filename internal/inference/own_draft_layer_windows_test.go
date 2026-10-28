//go:build windows

package inference

import (
	"slices"
	"testing"

	cudatest "overgo/internal/cuda/testutil"
	"overgo/internal/sampling"
	"overgo/internal/tokenizer"
)

// TestOwnDraftLayerSpeculatesByDefault: a runner whose model's file appends its own next-token layer - the hermetic
// hybrid's, as Qwen3.8-27B's does - speculates by default, as an attached draft does: a request that says nothing
// drafts its greedy decoding, token for token what the request that decodes plainly emits.
func TestOwnDraftLayerSpeculatesByDefault(t *testing.T) {
	// Serial: device fixtures share the one card.
	requireIntegration(t)
	cudatest.Require(t)
	runner, err := openF32FixtureRunner(writeHermeticQwen35GGUF(t), OpenOptions{})
	if err != nil {
		t.Fatal(err)
	}
	defer runner.Close()
	if !runner.SpeculatesByDefault() {
		t.Fatal("the model's own next-token layer does not speculate by default")
	}
	greedy, err := sampling.New(sampling.Config{Temperature: 0})
	if err != nil {
		t.Fatal(err)
	}
	drafted := 0
	byDefault := GenerateOptions{
		MaxNewTokens: 8, Sampler: greedy, DeviceGreedy: true, PromptTokenIDs: []tokenizer.TokenID{1, 4, 5},
		OnDraftSpan: func(drafts, _ int) { drafted += drafts },
	}
	speculated, _, err := runner.Generate(t.Context(), "", byDefault)
	if err != nil {
		t.Fatal(err)
	}
	if drafted == 0 {
		t.Fatal("a request that says nothing drafted no token")
	}
	plain := byDefault
	plain.PlainDecode, plain.OnDraftSpan = true, nil
	decoded, _, err := runner.Generate(t.Context(), "", plain)
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(speculated, decoded) {
		t.Fatalf("the default request emits %v, the plain one %v", speculated, decoded)
	}
}
