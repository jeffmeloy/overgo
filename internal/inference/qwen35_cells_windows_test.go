//go:build windows

package inference

import (
	"math"
	"slices"
	"testing"

	cudatest "overgo/internal/cuda/testutil"
	"overgo/internal/sampling"
	"overgo/internal/tokenizer"
)

// The miniature qwen35's attention geometry: heads of 256, four query heads over one K/V head, Qwen3.5-9B's
// instantiation (tensor.CellAttentionGeometries).
const (
	qwen35CellsWidth   = uint64(256)
	qwen35CellsHeads   = uint64(4)
	qwen35CellsKVHeads = uint64(1)
	// qwen35CellsContext holds the prompt and its decode.
	qwen35CellsContext = uint32(1024)
)

// TestQwen35CellsMatchF32Decoding holds a qwen35 runner over attention cells (cells-qwen35: its gated attention's K/V
// as Strata's int8 cells) to the same model over its F32 cache, on the hermetic hybrid at an instantiated geometry:
// three gated-delta layers and a gated attention layer of 256-wide heads, four query heads a K/V head. A 40-token
// prompt is read and 20 tokens decode, the cells' runner fed the F32 runner's greedy tokens: each step's logits within
// 2% of the F32 logits' spread, the codes' rounding being all that moves them. Over the cells the model's own
// next-token layer drafts losslessly: a drafted request emits the plain request's tokens.
func TestQwen35CellsMatchF32Decoding(t *testing.T) {
	// Serial: device fixtures share the one card.
	requireIntegration(t)
	cudatest.Require(t)
	// The hybrid's random weights keep its recurrent state finite through about 65 tokens; the prompt and its decode stay
	// within them.
	const promptTokens, steps = 40, 20
	path := writeHermeticQwen35GGUFShaped(t, "d", qwen35CellsContext, qwen35CellsWidth, qwen35CellsHeads, qwen35CellsKVHeads)
	open := func(f32 bool) *Runner {
		t.Helper()
		runner, err := openF32FixtureRunner(path, OpenOptions{F32Cache: f32})
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(func() { _ = runner.Close() })
		return runner
	}
	cells, f32 := open(false), open(true)
	if !slices.Equal(cells.cellLayers, []bool{false, false, false, true}) || f32.cellLayers != nil {
		t.Fatalf("the hybrid holds cells %v, its F32 runner %v; want its attention layer's alone", cells.cellLayers, f32.cellLayers)
	}
	prompt := make([]tokenizer.TokenID, promptTokens)
	for index := range prompt {
		prompt[index] = tokenizer.TokenID(3 + (index*7+index/5)%5)
	}
	// run reads the prompt, then feeds each step the token feed gives it, keeping each step's logits.
	run := func(runner *Runner, feed func(step int, logits []float32) tokenizer.TokenID) [][]float32 {
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
		for step := range steps + 1 {
			next, err := runner.forwardDeviceCachedModeLocked(ctx, deviceOutputLogits, tokens, past, nil)
			if err != nil {
				t.Fatalf("step %d: %v", step, err)
			}
			if past != nil {
				_ = past.Release(ctx)
			}
			past = next
			logits = append(logits, slices.Clone(next.Logits))
			tokens = []tokenizer.TokenID{feed(step, next.Logits)}
		}
		return logits
	}
	var fed []tokenizer.TokenID
	f32Logits := run(f32, func(_ int, logits []float32) tokenizer.TokenID {
		fed = append(fed, tokenizer.TokenID(slices.Index(logits, slices.Max(logits))))
		return fed[len(fed)-1]
	})
	cellLogits := run(cells, func(step int, _ []float32) tokenizer.TokenID { return fed[step] })
	for step := range f32Logits {
		reference, got := f32Logits[step], cellLogits[step]
		spread := float64(slices.Max(reference) - slices.Min(reference))
		difference := 0.0
		for index := range reference {
			difference = max(difference, math.Abs(float64(got[index]-reference[index])))
		}
		if difference > 0.02*spread {
			t.Fatalf("step %d: the cells' logits stand %.3g from the F32 cache's, past 2%% of their spread %.3g", step, difference, spread)
		}
	}

	greedy, err := sampling.New(sampling.Config{Temperature: 0})
	if err != nil {
		t.Fatal(err)
	}
	drafted := 0
	options := GenerateOptions{
		MaxNewTokens: 24, Sampler: greedy, DeviceGreedy: true, PromptTokenIDs: prompt[:8], SpeculativeDecode: true, ContinueAfterEOG: true,
		OnDraftSpan: func(drafts, _ int) { drafted += drafts },
	}
	speculated, _, err := cells.Generate(t.Context(), "", options)
	if err != nil {
		t.Fatal(err)
	}
	plain := options
	plain.SpeculativeDecode, plain.PlainDecode, plain.OnDraftSpan = false, true, nil
	decoded, _, err := cells.Generate(t.Context(), "", plain)
	if err != nil {
		t.Fatal(err)
	}
	if drafted == 0 || !slices.Equal(speculated, decoded) {
		t.Fatalf("over the cells %d drafts emit %v, plain decoding %v", drafted, speculated, decoded)
	}
}
