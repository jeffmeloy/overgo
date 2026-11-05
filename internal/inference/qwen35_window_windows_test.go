//go:build windows

package inference

import (
	"slices"
	"testing"

	cudatest "overgo/internal/cuda/testutil"
	"overgo/internal/sampling"
	"overgo/internal/tensor/dtype"
	"overgo/internal/testcheck"
	"overgo/internal/tokenizer"
)

// TestQwen35DraftsVerifyInAWindow drafts with a qwen35 hybrid over attention cells, at the family's gated-delta geometry,
// and holds its drafts to Strata's window verifier (memory-bounds/cell-windows): the runner decodes through windows,
// which read the sequence's fixed state where it sits and replay the kept tokens at their commit; a drafted request
// emits the tokens the same model decodes plainly over its F32 cache, which no window serves; and its verify spans keep
// no layer's state at their positions, so its span slots hold the spans' rows alone, less than the slots of the same
// request verified in spans over the F32 cache, where two had held 1.17 GiB of Qwen3.8-27B. The hybrid's random weights
// keep its 128-wide state finite through about 28 tokens, so the prompt and its decode stay within them.
func TestQwen35DraftsVerifyInAWindow(t *testing.T) {
	// Serial: device fixtures share the one card.
	requireIntegration(t)
	cudatest.Require(t)
	// The family's gated-delta geometry, which Strata's verify window holds (TestQwen35GatedDeltaLayersHoldTheWindow).
	family := qwen35Geometry{embedding: 64, ffn: 64, inner: 256, state: 128, groups: 1, rank: 2, gates: dtype.Q8_0}
	path := writeQwen35Fixture(t, family, "w", qwen35CellsContext, qwen35CellsWidth, qwen35CellsHeads, qwen35CellsKVHeads)
	open := cacheRunnerOpener(t, path)
	runner, f32 := open(false), open(true)
	if !runner.windowsDecode() {
		t.Fatalf("the hybrid over cells %v decodes outside Strata's window verifier", runner.cellLayers)
	}
	greedy, err := sampling.New(sampling.Config{Temperature: 0})
	testcheck.NoError(t, err)
	const promptTokens = 8
	prompt := make([]tokenizer.TokenID, promptTokens)
	for index := range prompt {
		prompt[index] = tokenizer.TokenID(3 + (index*7+index/5)%5)
	}
	drafted := 0
	options := GenerateOptions{
		MaxNewTokens: 12, Sampler: greedy, DeviceGreedy: true, PromptTokenIDs: prompt, SpeculativeDecode: true, ContinueAfterEOG: true,
		OnDraftSpan: func(drafts, _ int) { drafted += drafts },
	}
	speculated, _, err := runner.Generate(t.Context(), "", options)
	testcheck.NoError(t, err)
	if len(runner.spanned.layers) != 0 {
		t.Fatalf("a verify window kept %d layers' state at its positions", len(runner.spanned.layers))
	}
	plain := options
	plain.SpeculativeDecode, plain.PlainDecode, plain.OnDraftSpan = false, true, nil
	decoded, _, err := f32.Generate(t.Context(), "", plain)
	testcheck.NoError(t, err)
	if f32.windowsDecode() || drafted == 0 || !slices.Equal(speculated, decoded) {
		t.Fatalf("in windows %d drafts emit %v, plain decoding over the F32 cache %v", drafted, speculated, decoded)
	}
	if _, _, err := f32.Generate(t.Context(), "", options); err != nil {
		t.Fatal(err)
	}
	largest := func(runner *Runner) uint64 {
		var bytes uint64
		for _, slot := range runner.spanSlots {
			if slot != nil {
				bytes = max(bytes, slot.bytes)
			}
		}
		return bytes
	}
	windowed, spanned := largest(runner), largest(f32)
	t.Logf("span slots: %d bytes verifying in windows, %d in spans", windowed, spanned)
	if windowed >= spanned {
		t.Fatalf("the windows' span slots take %d bytes, the spans' %d", windowed, spanned)
	}
}
