//go:build windows

package inference

import (
	"context"
	"os"
	"path/filepath"
	"testing"

	cudatest "overgo/internal/cuda/testutil"
	"overgo/internal/gguf"
	"overgo/internal/testutil"
	"overgo/internal/tokenizer"
)

const (
	hybridEmbedding = uint64(8)
	hybridHeads     = uint64(2)
	hybridKVHeads   = uint64(1)
	hybridHeadWidth = uint64(4)
	hybridFFN       = uint64(12)
	hybridConv      = uint64(4)
	hybridInner     = uint64(8)
	hybridState     = uint64(4)
	hybridGroups    = uint64(2)
	hybridRank      = uint64(2)
	hybridKeyDim    = hybridState * hybridGroups
)

// writeHermeticQwen35GGUF emits a four-layer hybrid: three gated-delta
// recurrent layers followed by one gated-attention layer — the 27B's block
// pattern at hermetic scale.
func writeHermeticQwen35GGUF(t *testing.T) string {
	t.Helper()
	metadata := []gguf.Metadata{
		testutil.GGUFScalar("general.architecture", gguf.ValueTypeString, "qwen35"),
		testutil.GGUFScalar("general.name", gguf.ValueTypeString, "hermetic-hybrid"),
		testutil.GGUFScalar("qwen35.block_count", gguf.ValueTypeUint32, uint32(4)),
		testutil.GGUFScalar("qwen35.context_length", gguf.ValueTypeUint32, hermeticContext),
		testutil.GGUFScalar("qwen35.embedding_length", gguf.ValueTypeUint32, uint32(hybridEmbedding)),
		testutil.GGUFScalar("qwen35.feed_forward_length", gguf.ValueTypeUint32, uint32(hybridFFN)),
		testutil.GGUFScalar("qwen35.attention.head_count", gguf.ValueTypeUint32, uint32(hybridHeads)),
		testutil.GGUFScalar("qwen35.attention.head_count_kv", gguf.ValueTypeUint32, uint32(hybridKVHeads)),
		testutil.GGUFScalar("qwen35.attention.key_length", gguf.ValueTypeUint32, uint32(hybridHeadWidth)),
		testutil.GGUFScalar("qwen35.attention.value_length", gguf.ValueTypeUint32, uint32(hybridHeadWidth)),
		testutil.GGUFScalar("qwen35.rope.freq_base", gguf.ValueTypeFloat32, float32(10_000)),
		testutil.GGUFScalar("qwen35.rope.dimension_count", gguf.ValueTypeUint32, uint32(4)),
		testutil.GGUFArray("qwen35.rope.dimension_sections", gguf.ValueTypeInt32, []int32{1, 1, 0, 0}),
		testutil.GGUFScalar("qwen35.attention.layer_norm_rms_epsilon", gguf.ValueTypeFloat32, float32(1e-6)),
		testutil.GGUFScalar("qwen35.ssm.conv_kernel", gguf.ValueTypeUint32, uint32(hybridConv)),
		testutil.GGUFScalar("qwen35.ssm.inner_size", gguf.ValueTypeUint32, uint32(hybridInner)),
		testutil.GGUFScalar("qwen35.ssm.state_size", gguf.ValueTypeUint32, uint32(hybridState)),
		testutil.GGUFScalar("qwen35.ssm.time_step_rank", gguf.ValueTypeUint32, uint32(hybridRank)),
		testutil.GGUFScalar("qwen35.ssm.group_count", gguf.ValueTypeUint32, uint32(hybridGroups)),
		testutil.GGUFScalar("tokenizer.ggml.model", gguf.ValueTypeString, "llama"),
		testutil.GGUFArray("tokenizer.ggml.tokens", gguf.ValueTypeString,
			[]string{"<unk>", "<s>", "</s>", "â–", "a", "b", "c", "d"}),
		testutil.GGUFArray("tokenizer.ggml.scores", gguf.ValueTypeFloat32, make([]float32, hermeticVocab)),
		testutil.GGUFArray("tokenizer.ggml.token_type", gguf.ValueTypeInt32,
			[]int32{2, 3, 3, 1, 1, 1, 1, 1}),
		testutil.GGUFScalar("tokenizer.ggml.bos_token_id", gguf.ValueTypeUint32, uint32(1)),
		testutil.GGUFScalar("tokenizer.ggml.eos_token_id", gguf.ValueTypeUint32, uint32(2)),
	}
	tensors := []gguf.TensorData{
		testutil.GGUFTensorF32("token_embd.weight", []uint64{hybridEmbedding, hermeticVocab}, 1),
		testutil.GGUFTensorF32("output_norm.weight", []uint64{hybridEmbedding}, 2),
		testutil.GGUFTensorF32("output.weight", []uint64{hybridEmbedding, hermeticVocab}, 3),
	}
	seed := 4
	tensor := func(name string, shape ...uint64) {
		tensors = append(tensors, testutil.GGUFTensorF32(name, shape, seed))
		seed++
	}
	for _, block := range []string{"blk.0.", "blk.1.", "blk.2."} {
		tensor(block+"attn_norm.weight", hybridEmbedding)
		tensor(block+"attn_qkv.weight", hybridEmbedding, 2*hybridKeyDim+hybridInner)
		tensor(block+"attn_gate.weight", hybridEmbedding, hybridInner)
		tensor(block+"ssm_beta.weight", hybridEmbedding, hybridRank)
		tensor(block+"ssm_alpha.weight", hybridEmbedding, hybridRank)
		tensor(block+"ssm_conv1d.weight", hybridConv, 2*hybridKeyDim+hybridInner)
		tensor(block+"ssm_dt.bias", hybridRank)
		tensor(block+"ssm_a", hybridRank)
		tensor(block+"ssm_norm.weight", hybridState)
		tensor(block+"ssm_out.weight", hybridInner, hybridEmbedding)
		tensor(block+"post_attention_norm.weight", hybridEmbedding)
		tensor(block+"ffn_gate.weight", hybridEmbedding, hybridFFN)
		tensor(block+"ffn_up.weight", hybridEmbedding, hybridFFN)
		tensor(block+"ffn_down.weight", hybridFFN, hybridEmbedding)
	}
	attention := "blk.3."
	tensor(attention+"attn_norm.weight", hybridEmbedding)
	tensor(attention+"attn_q.weight", hybridEmbedding, hybridHeads*hybridHeadWidth*2)
	tensor(attention+"attn_k.weight", hybridEmbedding, hybridKVHeads*hybridHeadWidth)
	tensor(attention+"attn_v.weight", hybridEmbedding, hybridKVHeads*hybridHeadWidth)
	tensor(attention+"attn_output.weight", hybridHeads*hybridHeadWidth, hybridEmbedding)
	tensor(attention+"attn_q_norm.weight", hybridHeadWidth)
	tensor(attention+"attn_k_norm.weight", hybridHeadWidth)
	tensor(attention+"post_attention_norm.weight", hybridEmbedding)
	tensor(attention+"ffn_gate.weight", hybridEmbedding, hybridFFN)
	tensor(attention+"ffn_up.weight", hybridEmbedding, hybridFFN)
	tensor(attention+"ffn_down.weight", hybridFFN, hybridEmbedding)

	path := filepath.Join(t.TempDir(), "hermetic-hybrid.gguf")
	file, err := os.Create(path)
	if err != nil {
		t.Fatal(err)
	}
	if err = gguf.Write(file, metadata, tensors, gguf.WriteOptions{}); err != nil {
		_ = file.Close()
		t.Fatal(err)
	}
	if err := file.Close(); err != nil {
		t.Fatal(err)
	}
	return path
}

// TestRecurrentCheckpoint proves the draft-boundary contract the MTP loop
// needs on a hybrid model: a device cache holding gated-delta recurrent
// state is a checkpoint — appending a draft span yields a NEW cache and
// leaves the boundary cache intact, so rejection rolls back by releasing
// the speculative cache and decoding again from the boundary.
func TestRecurrentCheckpoint(t *testing.T) {
	requireIntegration(t)
	cudatest.Require(t)
	path := writeHermeticQwen35GGUF(t)
	runner, err := openF32FixtureRunner(path, OpenOptions{})
	if err != nil {
		t.Fatal(err)
	}
	defer runner.Close()
	ctx := context.Background()
	runner.mu.Lock()
	defer runner.mu.Unlock()
	release := func(cache *deviceKVCache) {
		if err := cache.Release(context.Background()); err != nil {
			t.Fatal(err)
		}
	}

	// Boundary: committed prefix with live recurrent state.
	boundary, err := runner.forwardDeviceCachedModeLocked(
		ctx, deviceOutputGreedySpan, []tokenizer.TokenID{1, 4}, nil,
	)
	if err != nil {
		t.Fatal(err)
	}
	defer release(boundary)

	// Draft span from the boundary, twice: identical selections prove the
	// append is a pure function of the boundary state.
	first, err := runner.forwardDeviceCachedModeLocked(
		ctx, deviceOutputGreedySpan, []tokenizer.TokenID{5, 6}, boundary,
	)
	if err != nil {
		t.Fatal(err)
	}
	firstSelected := append([]tokenizer.TokenID(nil), first.SpanSelected...)
	// Reject the draft: roll back by releasing the speculative cache.
	release(first)
	second, err := runner.forwardDeviceCachedModeLocked(
		ctx, deviceOutputGreedySpan, []tokenizer.TokenID{5, 6}, boundary,
	)
	if err != nil {
		t.Fatal(err)
	}
	defer release(second)
	if len(firstSelected) != 2 || len(second.SpanSelected) != 2 {
		t.Fatalf("span selections = %v/%v", firstSelected, second.SpanSelected)
	}
	for index := range firstSelected {
		if firstSelected[index] != second.SpanSelected[index] {
			t.Fatalf(
				"replay after rollback diverged at %d: %v vs %v",
				index, firstSelected, second.SpanSelected,
			)
		}
	}

	// A different continuation from the same boundary must match a fresh
	// decode of the full history — the boundary state survived both
	// speculative appends unchanged.
	alternative, err := runner.forwardDeviceCachedModeLocked(
		ctx, deviceOutputGreedySpan, []tokenizer.TokenID{7}, boundary,
	)
	if err != nil {
		t.Fatal(err)
	}
	defer release(alternative)
	fresh, err := runner.forwardDeviceCachedModeLocked(
		ctx, deviceOutputGreedySpan, []tokenizer.TokenID{1, 4, 7}, nil,
	)
	if err != nil {
		t.Fatal(err)
	}
	defer release(fresh)
	if len(alternative.SpanSelected) != 1 || len(fresh.SpanSelected) != 3 {
		t.Fatalf("selections = %v/%v", alternative.SpanSelected, fresh.SpanSelected)
	}
	if alternative.SpanSelected[0] != fresh.SpanSelected[2] {
		t.Fatalf(
			"boundary continuation selected %d, fresh history selected %d",
			alternative.SpanSelected[0], fresh.SpanSelected[2],
		)
	}
	if alternative.Tokens != fresh.Tokens || alternative.Position != fresh.Position {
		t.Fatalf(
			"cache tokens/position = %d/%d, fresh = %d/%d",
			alternative.Tokens, alternative.Position, fresh.Tokens, fresh.Position,
		)
	}
}
