//go:build windows

package inference

import (
	"math"
	"slices"
	"testing"

	cudatest "overgo/internal/cuda/testutil"
	"overgo/internal/modelrecipe"
	"overgo/internal/recipe"
	"overgo/internal/servingtest"
	"overgo/internal/testutil"
	"overgo/internal/tokenizer"
)

// TestAttentionCellsMatchF32Decoding holds a runner over attention cells (widen-kv: Strata's int8 K/V cells for an
// attention layer's own geometry) to the same runner over its F32 cache, on a miniature Gemma 4 12B: a sliding layer of
// 8 K/V heads of 256 reading a 24-token window and a global layer of one K/V head of 512 whose keys are its values. A
// 40-token prompt is read past the window, then 32 tokens decode, the cells' runner fed the F32 runner's greedy tokens:
// each step's logits within 2% of the F32 logits' spread, the codes' rounding (an fp16 scale a 64-value group, codes of
// 127 steps) being all that moves them, and the greedy token the same wherever the F32 runner's lead over its second is
// more than twice the logits' difference; the decode steps replay captured graphs. The cells take about a quarter of
// the F32 cache's bytes.
func TestAttentionCellsMatchF32Decoding(t *testing.T) {
	requireIntegration(t)
	cudatest.Require(t)
	const promptTokens, steps = 40, 32
	path := testutil.HermeticGemma4CellsGGUF(t, 256)
	open := func(cells bool) *Runner {
		t.Helper()
		loaded, err := servingtest.ResolveActiveGGUFWithPolicy(path, recipe.PlacementHybrid, modelrecipe.DecodeSessionCapacity,
			recipe.ResidencyDeviceF32)
		if err != nil {
			t.Fatal(err)
		}
		runner, err := OpenWithProgram(t.Context(), &loaded, OpenOptions{})
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(func() { _ = runner.Close() })
		if !slices.Equal(runner.cellLayers, []bool{true, true}) {
			t.Fatalf("the fixture at gemma-4-12B's layer kinds holds cells %v, want both layers", runner.cellLayers)
		}
		if !cells {
			runner.cellLayers = nil
		}
		return runner
	}
	prompt := make([]tokenizer.TokenID, promptTokens)
	for index := range prompt {
		prompt[index] = tokenizer.TokenID(3 + (index*7+index/5)%20)
	}
	// run reads the prompt, then feeds each step the token feed gives it, keeping each step's logits and the bytes the
	// cache's token-aligned storage takes after the prompt.
	run := func(runner *Runner, feed func(step int, logits []float32) tokenizer.TokenID) ([][]float32, uint64) {
		t.Helper()
		runner.mu.Lock()
		defer runner.mu.Unlock()
		ctx := t.Context()
		var past *deviceKVCache
		defer func() {
			if past != nil {
				_ = past.Release(ctx)
			}
		}()
		tokens := prompt
		var logits [][]float32
		var bytes uint64
		for step := range steps + 1 {
			next, err := runner.forwardDeviceCachedModeLocked(ctx, deviceOutputLogits, tokens, past, nil)
			if err != nil {
				t.Fatalf("step %d: %v", step, err)
			}
			if past != nil {
				_ = past.Release(ctx)
			}
			past = next
			if step == 0 {
				if next.storage.cells != nil {
					bytes = runner.attentionCellBytes(next.storage.cells.capacity)
				}
				for _, value := range append(slices.Clone(next.storage.keys), next.storage.values...) {
					bytes += value.CapacityBytes
				}
			}
			logits = append(logits, slices.Clone(next.Logits))
			tokens = []tokenizer.TokenID{feed(step, next.Logits)}
		}
		return logits, bytes
	}
	argmax := func(logits []float32) tokenizer.TokenID {
		return tokenizer.TokenID(slices.Index(logits, slices.Max(logits)))
	}
	var fed []tokenizer.TokenID
	f32Logits, f32Bytes := run(open(false), func(_ int, logits []float32) tokenizer.TokenID {
		fed = append(fed, argmax(logits))
		return fed[len(fed)-1]
	})
	cellRunner := open(true)
	before, err := cellRunner.DeviceExecutionStats(t.Context())
	if err != nil {
		t.Fatal(err)
	}
	cellLogits, cellBytes := run(cellRunner, func(step int, _ []float32) tokenizer.TokenID { return fed[step] })
	// The decode steps replay their captured graphs, each step's cells read from its staged step record.
	if after, err := cellRunner.DeviceExecutionStats(t.Context()); err != nil || after.GraphLaunches == before.GraphLaunches {
		t.Fatalf("the cells' decode replayed no graph: %v", err)
	}
	worst := 0.0
	for step := range f32Logits {
		reference, got := f32Logits[step], cellLogits[step]
		spread := float64(slices.Max(reference) - slices.Min(reference))
		difference := 0.0
		for index := range reference {
			difference = max(difference, math.Abs(float64(got[index]-reference[index])))
		}
		worst = max(worst, difference/spread)
		if difference > 0.02*spread {
			t.Fatalf("step %d: the cells' logits stand %.3g from the F32 cache's, past 2%% of their spread %.3g", step, difference, spread)
		}
		sorted := slices.Sorted(slices.Values(reference))
		if lead := float64(sorted[len(sorted)-1] - sorted[len(sorted)-2]); lead > 2*difference && argmax(got) != argmax(reference) {
			t.Fatalf("step %d: the cells select %d where the F32 cache selects %d by a lead of %.3g", step, argmax(got), argmax(reference), lead)
		}
	}
	if 3*cellBytes > f32Bytes {
		t.Fatalf("the cells take %d bytes against the F32 cache's %d", cellBytes, f32Bytes)
	}
	t.Logf("%d steps: the cells' logits within %.2g of the F32 spread; %d bytes against %d", len(f32Logits), worst, cellBytes, f32Bytes)
}
