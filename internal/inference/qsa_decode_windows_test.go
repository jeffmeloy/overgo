//go:build windows

package inference

import (
	"context"
	"encoding/binary"
	"math"
	"slices"
	"testing"

	"overgo/internal/cuda/driver"
	"overgo/internal/cuda/executor"
	cudatest "overgo/internal/cuda/testutil"
	"overgo/internal/recipe"
	"overgo/internal/sampling"
	"overgo/internal/tensor"
	"overgo/internal/tokenizer"
)

// strataFlashNext is the fixture's attention layer at the geometry Strata's QSA kernels are compiled for, the served
// file's: 24 heads over 2 K/V heads of 256 values, 64 rotated dims at base 1e7 over interleaved sections 11, 11 and
// 10, an indexer of 4 heads of 128 values with a budget of 2,048 cells in blocks of 4; a context past the budget and
// past a small ring.
var strataFlashNext = flashNextGeometry{
	context: 8192, heads: 24, kvHeads: 2, headWidth: 256, ropeDims: 64, sections: []int32{11, 11, 10, 0}, ropeBase: 1e7,
	indexHeads: 4, indexWidth: 128, budget: 2048, block: 4, decaying: true,
}

// TestInt8KVDecodingMatchesPlainDecoding holds Flash-Next's attention layer built as Strata's QSA layer - its int8
// K/V cells, the indexer's block selection and the attention over the selected cells, over the sequence's own cells
// - to the plain F32 decoding of the same fixture: a prompt read in chunks, the later ones past the indexer budget,
// then decode steps a kept session replays, each fed the plain decoding's greedy token, each step's greedy token the
// plain decoding's. The cells sit on the card in one case and stream past a ring smaller than the sequence in the
// other (Strata's kv_stream.hpp), whose decode steps then find selected blocks off the card and resolve them.
func TestInt8KVDecodingMatchesPlainDecoding(t *testing.T) {
	requireIntegration(t)
	cudatest.Require(t)
	path := writeFlashNextFixture(t, strataFlashNext)
	t.Run("on the card", func(t *testing.T) { int8KVDecoding(t, path, 2600, 0) })
	// A ring of 4,096 cells, the fewest slots the resolve takes, under a sequence of more than 5,000.
	t.Run("streamed past the ring", func(t *testing.T) { int8KVDecoding(t, path, 5200, 4096) })
}

func int8KVDecoding(t *testing.T, path string, promptTokens int, ring uint32) {
	const (
		chunkTokens = 768
		decodeSteps = 24
	)
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
	qsa.qsaRing, plain.qsaLayers = ring, nil
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
	if streamed := qsaPast.storage.qsa.slots > 0; streamed != (ring != 0) {
		t.Fatalf("the cells stream %t with a ring of %d cells over %d", streamed, ring, qsaPast.storage.qsa.capacity)
	}
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
	if ring == 0 {
		return
	}
	// The resolve's counters: a streamed decode found selected blocks off the card and brought them in.
	cells := qsaPast.storage.qsa
	control := make([]byte, tensor.QSAStreamControlInts*i32Bytes)
	at := cells.layers[flashNextAttention].streamMap.Pointer + driver.DevicePtr(tensor.QSAStreamSlotArrays*uint64(cells.slots)*i32Bytes)
	if err := qsa.cuda.ReadDeviceBytes(ctx, []executor.DeviceBytes{{Pointer: at, Data: control}}); err != nil {
		t.Fatal(err)
	}
	misses, lookups, calls := binary.LittleEndian.Uint64(control[4*i32Bytes:]), binary.LittleEndian.Uint64(control[6*i32Bytes:]), binary.LittleEndian.Uint64(control[8*i32Bytes:])
	t.Logf("the attention layer's resolve: %d calls, %d block lookups, %d misses", calls, lookups, misses)
	if overflow := binary.LittleEndian.Uint32(control[3*i32Bytes:]); overflow != 0 || calls == 0 || misses == 0 {
		t.Fatalf("the streamed decode resolved %d calls with %d misses (overflow %d), want misses brought in", calls, misses, overflow)
	}
}

// TestQSACellsHoldOneCapacity holds a QSA runner's sequence to Strata's max_cells (kv_plan over --max-context): the
// cells a sequence takes at its first token are the ring's while it fits the ring and the context's past it, so the
// turns of a conversation and a fresh conversation, each asking for its own output, decode through the sessions the
// first turn compiled, and a prompt past the ring takes the context's cells, streamed.
func TestQSACellsHoldOneCapacity(t *testing.T) {
	requireIntegration(t)
	cudatest.Require(t)
	const ring = 4096
	trunk := writeFlashNextFixture(t, strataFlashNext)
	greedy, err := sampling.New(sampling.Config{Temperature: 0})
	if err != nil {
		t.Fatal(err)
	}
	runner, err := openFixtureRunnerWithResidency(trunk,
		OpenOptions{Draft: writeFlashNextMTPFixture(t, trunk, false, strataFlashNext), PromptCacheEntries: 1}, recipe.ResidencyDeviceNative)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := runner.Close(); err != nil {
			t.Error(err)
		}
	})
	runner.qsaRing = ring
	prompt := func(tokens int) []tokenizer.TokenID {
		ids := make([]tokenizer.TokenID, tokens)
		for index := range ids {
			ids[index] = tokenizer.TokenID(3 + (index*7+index/5)%int(flashNextVocab-3))
		}
		return ids
	}
	// capacity is the cells and VRAM slots of the sequence the prompt cache keeps after a turn.
	capacity := func() (uint32, uint32) {
		t.Helper()
		for _, entry := range runner.promptCaches {
			for _, cache := range entry.deviceCaches() {
				if cache.storage != nil && cache.storage.qsa != nil {
					return cache.storage.qsa.capacity, cache.storage.qsa.slots
				}
			}
		}
		t.Fatal("the prompt cache keeps no QSA cells")
		return 0, 0
	}
	var ids []tokenizer.TokenID
	sessions := 0
	for turn, generated := range []int{24, 40, 31, 57} {
		input := append(slices.Clone(ids), 21, 30, 7)
		if turn == 0 || turn == 3 {
			input = prompt(40)
			input[len(input)-1] = tokenizer.TokenID(3 + turn)
		}
		ids, _, err = runner.Generate(t.Context(), "", GenerateOptions{
			MaxNewTokens: generated, Sampler: greedy, DeviceGreedy: true, SpeculativeDecode: true, CachePrompt: true,
			PromptTokenIDs: input, ContinueAfterEOG: true,
		})
		if err != nil {
			t.Fatal(err)
		}
		if cells, slots := capacity(); cells != ring || slots != 0 {
			t.Fatalf("turn %d: the sequence holds %d cells (%d slots), want the ring's %d on the card", turn, cells, slots, ring)
		}
		if turn == 0 {
			sessions = len(runner.decodeSessions)
		} else if len(runner.decodeSessions) != sessions {
			t.Fatalf("turn %d: %d decode sessions compiled, the first turn's %d", turn, len(runner.decodeSessions), sessions)
		}
	}
	if _, _, err = runner.Generate(t.Context(), "", GenerateOptions{
		MaxNewTokens: 8, Sampler: greedy, DeviceGreedy: true, SpeculativeDecode: true, CachePrompt: true,
		PromptTokenIDs: prompt(ring + 100), ContinueAfterEOG: true,
	}); err != nil {
		t.Fatal(err)
	}
	if cells, slots := capacity(); cells != strataFlashNext.context || slots == 0 {
		t.Fatalf("a prompt past the ring holds %d cells (%d slots), want the context's %d streamed", cells, slots, strataFlashNext.context)
	}
	t.Logf("four turns decoded through %d sessions at %d cells", sessions, ring)
}

// TestQSABranchReadsOnFromAHeldBoundary: a turn that branches off before the conversation's last question reads on
// from a boundary whose fixed state its sequence still holds, to the tokens a cold read decodes. A decoded entry names
// its sequence's live state and stays a boundary of the entries after it; the next turn's decode moves that state on.
func TestQSABranchReadsOnFromAHeldBoundary(t *testing.T) {
	requireIntegration(t)
	cudatest.Require(t)
	greedy, err := sampling.New(sampling.Config{Temperature: 0})
	if err != nil {
		t.Fatal(err)
	}
	runner, err := openFixtureRunnerWithResidency(writeFlashNextFixture(t, strataFlashNextWindows), OpenOptions{PromptCacheEntries: 1},
		recipe.ResidencyDeviceNative)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := runner.Close(); err != nil {
			t.Error(err)
		}
	})
	generate := func(prompt []tokenizer.TokenID, cached bool) []tokenizer.TokenID {
		t.Helper()
		ids, _, err := runner.Generate(t.Context(), "", GenerateOptions{
			MaxNewTokens: 12, Sampler: greedy, DeviceGreedy: true, CachePrompt: cached, PromptTokenIDs: prompt,
			ContinueAfterEOG: true,
		})
		if err != nil {
			t.Fatal(err)
		}
		return ids
	}
	history := make([]tokenizer.TokenID, 40)
	for index := range history {
		history[index] = tokenizer.TokenID(3 + (index*11+index/3)%int(flashNextVocab-3))
	}
	// Serial: the turns share one runner, its prompt cache and its sequence's live state.
	first := generate(history, true)
	generate(append(slices.Clone(first), 21, 30, 7), true)
	if !runner.windowsDecode() || !slices.ContainsFunc(runner.promptCaches[0].Boundaries, (*deviceKVCache).staleLive) {
		t.Fatal("the second turn left no boundary naming a live state it moved on")
	}
	branch := append(slices.Clone(first), 22, 31, 8)
	warm := generate(branch, true)
	clearPromptCaches(t, runner)
	if cold := generate(branch, false); !slices.Equal(warm, cold) {
		t.Fatalf("the branch decoded %v, a cold read %v", warm[len(branch):], cold[len(branch):])
	}
}
