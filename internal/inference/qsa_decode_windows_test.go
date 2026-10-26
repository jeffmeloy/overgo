//go:build windows

package inference

import (
	"context"
	"math"
	"testing"

	cudatest "overgo/internal/cuda/testutil"
	"overgo/internal/tokenizer"
)

// strataFlashNext is the fixture's attention layer at the geometry Strata's QSA kernels are compiled for, the served
// file's: 24 heads over 2 K/V heads of 256 values, 64 rotated dims at base 1e7 over interleaved sections 11, 11 and
// 10, an indexer of 4 heads of 128 values with a budget of 2,048 cells in blocks of 4; a context past the budget.
var strataFlashNext = flashNextGeometry{
	context: 4096, heads: 24, kvHeads: 2, headWidth: 256, ropeDims: 64, sections: []int32{11, 11, 10, 0}, ropeBase: 1e7,
	indexHeads: 4, indexWidth: 128, budget: 2048, block: 4, decaying: true,
}

// TestInt8KVDecodingMatchesPlainDecoding holds Flash-Next's attention layer built as Strata's QSA layer - its int8
// K/V cells, the indexer's block selection and the attention over the selected cells, over the sequence's own cells
// - to the plain F32 decoding of the same fixture: a prompt read in chunks, the later ones past the indexer budget,
// then decode steps a kept session replays, each fed the plain decoding's greedy token, each step's greedy token the
// plain decoding's.
func TestInt8KVDecodingMatchesPlainDecoding(t *testing.T) {
	requireIntegration(t)
	cudatest.Require(t)
	const (
		promptTokens = 2600
		chunkTokens  = 768
		decodeSteps  = 24
	)
	path := writeFlashNextFixture(t, strataFlashNext)
	open := func() *Runner {
		t.Helper()
		runner, err := openF32FixtureRunner(path, OpenOptions{})
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(func() {
			if err := runner.Close(); err != nil {
				t.Error(err)
			}
		})
		runner.promptChunkTokens = chunkTokens
		return runner
	}
	qsa, plain := open(), open()
	if !qsa.holdsQSA(flashNextAttention) {
		t.Fatal("the fixture's attention layer is not built as Strata's QSA layer")
	}
	plain.qsaLayers = nil
	ctx := t.Context()
	qsa.mu.Lock()
	defer qsa.mu.Unlock()
	plain.mu.Lock()
	defer plain.mu.Unlock()

	prompt := make([]tokenizer.TokenID, promptTokens)
	for index := range prompt {
		prompt[index] = tokenizer.TokenID(3 + (index*7+index/5)%int(flashNextVocab-3))
	}
	var qsaPast, plainPast *deviceKVCache
	defer func() {
		for _, cache := range []*deviceKVCache{qsaPast, plainPast} {
			if err := cache.Release(context.Background()); err != nil {
				t.Error(err)
			}
		}
	}()
	step := func(runner *Runner, tokens []tokenizer.TokenID, past **deviceKVCache) {
		t.Helper()
		next, err := runner.forwardDeviceCachedModeLocked(ctx, deviceOutputLogits, tokens, *past, nil)
		if err != nil {
			t.Fatal(err)
		}
		if err := (*past).Release(ctx); err != nil {
			t.Fatal(err)
		}
		*past = next
	}
	argmax := func(logits []float32) (int, float64) {
		best, second := 0, -1
		for index := range logits {
			switch {
			case logits[index] > logits[best]:
				best, second = index, best
			case index != best && (second < 0 || logits[index] > logits[second]):
				second = index
			}
		}
		return best, float64(logits[best] - logits[second])
	}
	step(plain, prompt, &plainPast)
	step(qsa, prompt, &qsaPast)
	minimumMargin, largest := math.Inf(1), 0.0
	for decoded := 0; ; decoded++ {
		if len(qsaPast.Logits) != len(plainPast.Logits) || len(plainPast.Logits) == 0 {
			t.Fatalf("step %d: logits %d against %d", decoded, len(qsaPast.Logits), len(plainPast.Logits))
		}
		want, margin := argmax(plainPast.Logits)
		got, _ := argmax(qsaPast.Logits)
		difference := 0.0
		for index, value := range plainPast.Logits {
			difference = max(difference, math.Abs(float64(qsaPast.Logits[index]-value)))
		}
		t.Logf("cells %d: greedy %d against plain %d, plain margin %.4g, logits %.4g apart", qsaPast.Tokens, got, want, margin, difference)
		if got != want {
			t.Fatalf("cells %d: the QSA decoding selects %d, the plain decoding %d (its margin %.4g, the logits %.4g apart)",
				qsaPast.Tokens, got, want, margin, difference)
		}
		minimumMargin, largest = min(minimumMargin, margin), max(largest, difference)
		if decoded == decodeSteps {
			break
		}
		token := []tokenizer.TokenID{tokenizer.TokenID(want)}
		step(qsa, token, &qsaPast)
		step(plain, token, &plainPast)
	}
	t.Logf("%d steps from %d cells: the greedy tokens equal, the plain margins at least %.4g, the logits at most %.4g apart",
		decodeSteps, promptTokens, minimumMargin, largest)
}
