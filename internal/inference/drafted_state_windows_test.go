//go:build windows

package inference

import (
	"maps"
	"slices"
	"testing"

	cudatest "overgo/internal/cuda/testutil"
	"overgo/internal/recipe"
	"overgo/internal/sampling"
	"overgo/internal/tokenizer"
)

// TestDraftedWindowStateBytesAreReported drafts with a hermetic Qwen3.5 runner and reads what its newest verify span
// wrote into its span slot: each recurrent layer's state at the span's positions, and the hidden rows and selections
// beside them, all within the slot's bytes. The table prices a drafted window's per-position state against replaying
// its kept tokens, as Strata's verifier does.
func TestDraftedWindowStateBytesAreReported(t *testing.T) {
	requireIntegration(t)
	cudatest.Require(t)
	runner, err := openFixtureRunnerWithResidency(writeHermeticQwen35GGUF(t), OpenOptions{}, recipe.ResidencyDeviceF32)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := runner.Close(); err != nil {
			t.Error(err)
		}
	})
	greedy, err := sampling.New(sampling.Config{Temperature: 0})
	if err != nil {
		t.Fatal(err)
	}
	if _, _, err := runner.Generate(t.Context(), "", GenerateOptions{
		MaxNewTokens: 12, Sampler: greedy, DeviceGreedy: true, SpeculativeDecode: true,
		PromptTokenIDs: []tokenizer.TokenID{1, 4, 5}, ContinueAfterEOG: true,
	}); err != nil {
		t.Fatal(err)
	}
	spanned := runner.spanned
	if spanned.positions < 1 || len(spanned.layers) == 0 || spanned.rows == 0 {
		t.Fatalf("the newest verify span reports %d positions, %d layers and %d bytes of rows, want a drafted window's state", spanned.positions, len(spanned.layers), spanned.rows)
	}
	total := spanned.rows
	for _, layer := range slices.Sorted(maps.Keys(spanned.layers)) {
		bytes := spanned.layers[layer]
		total += bytes
		t.Logf("layer %d: %d bytes, %d a position", layer, bytes, bytes/uint64(spanned.positions))
	}
	t.Logf("hidden rows and selections: %d bytes; %d positions", spanned.rows, spanned.positions)
	if !slices.ContainsFunc(runner.spanSlots[:], func(slot *spanSlot) bool { return slot != nil && slot.bytes >= total }) {
		t.Fatalf("the reported %d bytes fit no span slot", total)
	}
}
