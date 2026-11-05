//go:build windows

package inference

import (
	"fmt"
	"math"
	"slices"
	"testing"

	cudatest "overgo/internal/cuda/testutil"
	"overgo/internal/gguf"
	"overgo/internal/tensor/dtype"
	"overgo/internal/testcheck"
	"overgo/internal/testutil"
	"overgo/internal/tokenizer"
)

// bf16CacheGeometry is a llama fixture whose attention the tensor-core prompt kernel takes: two layers of two 64-wide
// query heads over one key-value head.
const (
	bf16CacheLayers    = 2
	bf16CacheEmbedding = uint64(128)
	bf16CacheHeads     = uint64(2)
	bf16CacheKVHeads   = uint64(1)
	bf16CacheFFN       = uint64(64)
	bf16CacheVocab     = 32
	bf16CacheContext   = uint32(256)
	bf16CachePrompt    = 212
	// bf16CacheChunk reads the prompt in three chunks: the first over no past attends to its own F32 keys, the second
	// reads the cache on the tensor-core kernel, writing more of its capacity than is left at F32's width, and the
	// third, too short for that kernel, on the online kernel.
	bf16CacheChunk = 100
	bf16CacheSteps = 6
)

func writeBF16CacheFixture(t *testing.T) string {
	t.Helper()
	width := bf16CacheEmbedding / bf16CacheHeads
	tokens := []string{"<unk>", "<s>", "</s>"}
	types := []int32{2, 3, 3}
	for index := len(tokens); index < bf16CacheVocab; index++ {
		tokens, types = append(tokens, fmt.Sprintf("t%d", index)), append(types, 1)
	}
	metadata := []gguf.Metadata{
		testutil.GGUFScalar("general.architecture", gguf.ValueTypeString, "llama"),
		testutil.GGUFScalar("general.name", gguf.ValueTypeString, "bf16-cache"),
		testutil.GGUFScalar("llama.block_count", gguf.ValueTypeUint32, uint32(bf16CacheLayers)),
		testutil.GGUFScalar("llama.context_length", gguf.ValueTypeUint32, bf16CacheContext),
		testutil.GGUFScalar("llama.embedding_length", gguf.ValueTypeUint32, uint32(bf16CacheEmbedding)),
		testutil.GGUFScalar("llama.feed_forward_length", gguf.ValueTypeUint32, uint32(bf16CacheFFN)),
		testutil.GGUFScalar("llama.attention.head_count", gguf.ValueTypeUint32, uint32(bf16CacheHeads)),
		testutil.GGUFScalar("llama.attention.head_count_kv", gguf.ValueTypeUint32, uint32(bf16CacheKVHeads)),
		testutil.GGUFScalar("llama.rope.freq_base", gguf.ValueTypeFloat32, float32(10_000)),
		testutil.GGUFScalar("llama.attention.layer_norm_rms_epsilon", gguf.ValueTypeFloat32, float32(1e-5)),
		testutil.GGUFScalar("tokenizer.ggml.model", gguf.ValueTypeString, "llama"),
		testutil.GGUFArray("tokenizer.ggml.tokens", gguf.ValueTypeString, tokens),
		testutil.GGUFArray("tokenizer.ggml.scores", gguf.ValueTypeFloat32, make([]float32, bf16CacheVocab)),
		testutil.GGUFArray("tokenizer.ggml.token_type", gguf.ValueTypeInt32, types),
		testutil.GGUFScalar("tokenizer.ggml.bos_token_id", gguf.ValueTypeUint32, uint32(1)),
		testutil.GGUFScalar("tokenizer.ggml.eos_token_id", gguf.ValueTypeUint32, uint32(2)),
	}
	seed := 1
	tensor := func(name string, shape ...uint64) gguf.TensorData {
		seed++
		return testutil.GGUFTensorF32(name, shape, seed)
	}
	tensors := []gguf.TensorData{
		tensor("token_embd.weight", bf16CacheEmbedding, bf16CacheVocab),
		tensor("output_norm.weight", bf16CacheEmbedding),
		tensor("output.weight", bf16CacheEmbedding, bf16CacheVocab),
	}
	for layer := range bf16CacheLayers {
		name := func(suffix string) string { return fmt.Sprintf("blk.%d.%s", layer, suffix) }
		tensors = append(tensors,
			tensor(name("attn_norm.weight"), bf16CacheEmbedding),
			tensor(name("attn_q.weight"), bf16CacheEmbedding, bf16CacheEmbedding),
			tensor(name("attn_k.weight"), bf16CacheEmbedding, width*bf16CacheKVHeads),
			tensor(name("attn_v.weight"), bf16CacheEmbedding, width*bf16CacheKVHeads),
			tensor(name("attn_output.weight"), bf16CacheEmbedding, bf16CacheEmbedding),
			tensor(name("ffn_norm.weight"), bf16CacheEmbedding),
			tensor(name("ffn_gate.weight"), bf16CacheEmbedding, bf16CacheFFN),
			tensor(name("ffn_up.weight"), bf16CacheEmbedding, bf16CacheFFN),
			tensor(name("ffn_down.weight"), bf16CacheFFN, bf16CacheEmbedding),
		)
	}
	return testutil.TempGGUF(t, "bf16-cache.gguf", metadata, tensors)
}

// TestTheKeyValueCacheIsStoredInBF16 holds a dense runner whose attention the tensor-core prompt kernel takes to a BF16
// cache, against the same model held to the F32 cache (OpenOptions.F32Cache): its keys and values take half the F32
// cache's bytes; a prompt's first layer holds each of the F32 cache's values at its nearest BF16, and moves to host
// memory and back unchanged; and its prompt, read in chunks the tensor-core and online kernels read the cache in, and
// its decode steps, fed the F32 cache's greedy tokens, give logits within a hundredth of their spread of the F32
// cache's, selecting its tokens where they lead.
func TestTheKeyValueCacheIsStoredInBF16(t *testing.T) {
	// Serial: device fixtures share the one card.
	requireIntegration(t)
	cudatest.Require(t)
	path := writeBF16CacheFixture(t)
	open := func(f32 bool) *Runner {
		t.Helper()
		runner, err := openF32FixtureRunner(path, OpenOptions{F32Cache: f32})
		testcheck.NoError(t, err)
		t.Cleanup(func() { _ = runner.Close() })
		return runner
	}
	f32Runner, bf16Runner := open(true), open(false)
	if f32Runner.cacheType != dtype.F32 || bf16Runner.cacheType != dtype.BF16 {
		t.Fatalf("the runners keep their caches at %s and %s, want F32 and BF16", f32Runner.cacheType, bf16Runner.cacheType)
	}
	prompt := make([]tokenizer.TokenID, bf16CachePrompt)
	for index := range prompt {
		prompt[index] = tokenizer.TokenID(3 + (index*7+index/5)%(bf16CacheVocab-3))
	}
	argmax := func(logits []float32) tokenizer.TokenID {
		return tokenizer.TokenID(slices.Index(logits, slices.Max(logits)))
	}
	type run struct {
		logits [][]float32
		bytes  uint64
		first  []float32
	}
	// read reads the prompt, then feeds each step the token feed gives it, keeping each step's logits, the bytes the
	// prompt's keys and values take and its first layer's keys and values.
	read := func(runner *Runner, feed func(step int, logits []float32) tokenizer.TokenID) run {
		t.Helper()
		runner.mu.Lock()
		defer runner.mu.Unlock()
		runner.promptChunkTokens = bf16CacheChunk
		ctx := t.Context()
		var past *deviceKVCache
		defer func() {
			if past != nil {
				_ = past.Release(ctx)
			}
		}()
		var result run
		tokens := prompt
		for step := range bf16CacheSteps + 1 {
			next, err := runner.forwardDeviceCachedModeLocked(ctx, deviceOutputLogits, tokens, past, nil)
			if err != nil {
				t.Fatalf("step %d: %v", step, err)
			}
			if past != nil {
				_ = past.Release(ctx)
			}
			past = next
			if step == 0 {
				for _, value := range append(slices.Clone(next.storage.keys), next.storage.values...) {
					result.bytes += value.CapacityBytes
				}
				host, err := runner.downloadDeviceCache(ctx, next)
				testcheck.NoError(t, err)
				result.first = append(slices.Clone(host.Layers[0].Key.Data), host.Layers[0].Value.Data...)
				uploaded, err := runner.uploadDeviceCache(ctx, host)
				testcheck.NoError(t, err)
				back, err := runner.downloadDeviceCache(ctx, uploaded)
				_ = uploaded.Release(ctx)
				testcheck.NoError(t, err)
				if !slices.Equal(append(slices.Clone(back.Layers[0].Key.Data), back.Layers[0].Value.Data...), result.first) {
					t.Fatal("the cache moved to host memory and back holds other values")
				}
			}
			result.logits = append(result.logits, slices.Clone(next.Logits))
			tokens = []tokenizer.TokenID{feed(step, next.Logits)}
		}
		return result
	}
	var fed []tokenizer.TokenID
	f32 := read(f32Runner, func(_ int, logits []float32) tokenizer.TokenID {
		fed = append(fed, argmax(logits))
		return fed[len(fed)-1]
	})
	bf16 := read(bf16Runner, func(step int, _ []float32) tokenizer.TokenID { return fed[step] })
	if 2*bf16.bytes != f32.bytes {
		t.Fatalf("the BF16 cache's keys and values take %d bytes against the F32 cache's %d", bf16.bytes, f32.bytes)
	}
	if len(bf16.first) != len(f32.first) {
		t.Fatalf("the BF16 cache's first layer holds %d values against %d", len(bf16.first), len(f32.first))
	}
	for index, value := range f32.first {
		if bf16.first[index] != dtype.RoundBF16(value) {
			t.Fatalf("the BF16 cache holds %g where the F32 cache holds %g", bf16.first[index], value)
		}
	}
	worst := 0.0
	for step := range f32.logits {
		reference, got := f32.logits[step], bf16.logits[step]
		spread := float64(slices.Max(reference) - slices.Min(reference))
		difference := 0.0
		for index := range reference {
			difference = max(difference, math.Abs(float64(got[index]-reference[index])))
		}
		worst = max(worst, difference/spread)
		if !(difference <= 0.01*spread) {
			t.Fatalf("step %d: the BF16 cache's logits stand %.3g from the F32 cache's, past 1%% of their spread %.3g", step, difference, spread)
		}
		sorted := slices.Sorted(slices.Values(reference))
		if lead := float64(sorted[len(sorted)-1] - sorted[len(sorted)-2]); lead > 2*difference && argmax(got) != argmax(reference) {
			t.Fatalf("step %d: the BF16 cache selects %d where the F32 cache selects %d by a lead of %.3g", step, argmax(got), argmax(reference), lead)
		}
	}
	t.Logf("%d steps: the BF16 cache's logits within %.2g of the F32 spread; %d bytes against %d", len(f32.logits), worst, bf16.bytes, f32.bytes)
}
