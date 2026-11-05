//go:build windows

package inference

import (
	"bytes"
	"encoding/binary"
	"io"
	"os"
	"path/filepath"
	"testing"

	cudatest "overgo/internal/cuda/testutil"
	"overgo/internal/gguf"
	"overgo/internal/quant"
	"overgo/internal/tensor/dtype"
	"overgo/internal/testcheck"
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
	return writeHermeticQwen35GGUFLast(t, "d")
}

// writeHermeticQwen35GGUFLast is writeHermeticQwen35GGUF with last as its last token's text.
func writeHermeticQwen35GGUFLast(t *testing.T, last string) string {
	t.Helper()
	return writeHermeticQwen35GGUFShaped(t, last, hermeticContext, hybridHeadWidth, hybridHeads, hybridKVHeads)
}

// qwen35Geometry is a hybrid fixture's geometry: its embedding and feed-forward widths, its gated-delta layers' inner
// width, state width, key groups and value heads, and the type its alpha and beta are stored in.
type qwen35Geometry struct {
	embedding, ffn, inner, state, groups, rank uint64
	gates                                      dtype.Type
}

// hermeticQwen35 is the hermetic hybrid's geometry.
var hermeticQwen35 = qwen35Geometry{
	embedding: hybridEmbedding, ffn: hybridFFN, inner: hybridInner, state: hybridState, groups: hybridGroups, rank: hybridRank,
	gates: dtype.F32,
}

// writeHermeticQwen35GGUFShaped is the hybrid at a context of context tokens, its attention layers' heads of width under
// heads query heads over kvHeads K/V heads.
func writeHermeticQwen35GGUFShaped(t *testing.T, last string, context uint32, width, heads, kvHeads uint64) string {
	t.Helper()
	return writeQwen35Fixture(t, hermeticQwen35, last, context, width, heads, kvHeads)
}

// writeQwen35Fixture is the hybrid at geometry g, its context and attention layers as writeHermeticQwen35GGUFShaped
// takes them; alpha and beta of another type than F32 are quantized from the values an F32 fixture holds.
func writeQwen35Fixture(t *testing.T, g qwen35Geometry, last string, context uint32, width, heads, kvHeads uint64) string {
	t.Helper()
	metadata := []gguf.Metadata{
		testutil.GGUFScalar("general.architecture", gguf.ValueTypeString, "qwen35"),
		testutil.GGUFScalar("general.name", gguf.ValueTypeString, "hermetic-hybrid"),
		testutil.GGUFScalar("qwen35.block_count", gguf.ValueTypeUint32, uint32(5)),
		testutil.GGUFScalar("qwen35.nextn_predict_layers", gguf.ValueTypeUint32, uint32(1)),
		testutil.GGUFScalar("qwen35.context_length", gguf.ValueTypeUint32, context),
		testutil.GGUFScalar("qwen35.embedding_length", gguf.ValueTypeUint32, uint32(g.embedding)),
		testutil.GGUFScalar("qwen35.feed_forward_length", gguf.ValueTypeUint32, uint32(g.ffn)),
		testutil.GGUFScalar("qwen35.attention.head_count", gguf.ValueTypeUint32, uint32(heads)),
		testutil.GGUFScalar("qwen35.attention.head_count_kv", gguf.ValueTypeUint32, uint32(kvHeads)),
		testutil.GGUFScalar("qwen35.attention.key_length", gguf.ValueTypeUint32, uint32(width)),
		testutil.GGUFScalar("qwen35.attention.value_length", gguf.ValueTypeUint32, uint32(width)),
		testutil.GGUFScalar("qwen35.rope.freq_base", gguf.ValueTypeFloat32, float32(10_000)),
		testutil.GGUFScalar("qwen35.rope.dimension_count", gguf.ValueTypeUint32, uint32(4)),
		testutil.GGUFArray("qwen35.rope.dimension_sections", gguf.ValueTypeInt32, []int32{1, 1, 0, 0}),
		testutil.GGUFScalar("qwen35.attention.layer_norm_rms_epsilon", gguf.ValueTypeFloat32, float32(1e-6)),
		testutil.GGUFScalar("qwen35.ssm.conv_kernel", gguf.ValueTypeUint32, uint32(hybridConv)),
		testutil.GGUFScalar("qwen35.ssm.inner_size", gguf.ValueTypeUint32, uint32(g.inner)),
		testutil.GGUFScalar("qwen35.ssm.state_size", gguf.ValueTypeUint32, uint32(g.state)),
		testutil.GGUFScalar("qwen35.ssm.time_step_rank", gguf.ValueTypeUint32, uint32(g.rank)),
		testutil.GGUFScalar("qwen35.ssm.group_count", gguf.ValueTypeUint32, uint32(g.groups)),
		testutil.GGUFScalar("tokenizer.ggml.model", gguf.ValueTypeString, "llama"),
		testutil.GGUFArray("tokenizer.ggml.tokens", gguf.ValueTypeString,
			[]string{"<unk>", "<s>", "</s>", "â–", "a", "b", "c", last}),
		testutil.GGUFArray("tokenizer.ggml.scores", gguf.ValueTypeFloat32, make([]float32, hermeticVocab)),
		testutil.GGUFArray("tokenizer.ggml.token_type", gguf.ValueTypeInt32,
			[]int32{2, 3, 3, 1, 1, 1, 1, 1}),
		testutil.GGUFScalar("tokenizer.ggml.bos_token_id", gguf.ValueTypeUint32, uint32(1)),
		testutil.GGUFScalar("tokenizer.ggml.eos_token_id", gguf.ValueTypeUint32, uint32(2)),
	}
	tensors := []gguf.TensorData{
		testutil.GGUFTensorF32("token_embd.weight", []uint64{g.embedding, hermeticVocab}, 1),
		testutil.GGUFTensorF32("output_norm.weight", []uint64{g.embedding}, 2),
		testutil.GGUFTensorF32("output.weight", []uint64{g.embedding, hermeticVocab}, 3),
	}
	seed := 4
	tensor := func(name string, shape ...uint64) {
		tensors = append(tensors, testutil.GGUFTensorF32(name, shape, seed))
		seed++
	}
	gates := func(name string, shape ...uint64) {
		if g.gates == dtype.F32 {
			tensor(name, shape...)
			return
		}
		data := testutil.GGUFTensorF32(name, shape, seed)
		seed++
		raw, err := io.ReadAll(data.Data)
		testcheck.NoError(t, err)
		values := make([]float32, len(raw)/4)
		testcheck.NoError(t, binary.Read(bytes.NewReader(raw), binary.LittleEndian, values))
		quantized, err := quant.Quantize(g.gates, values)
		testcheck.NoError(t, err)
		tensors = append(tensors, gguf.TensorData{Name: name, Shape: shape, Type: g.gates, Data: bytes.NewReader(quantized)})
	}
	for _, block := range []string{"blk.0.", "blk.1.", "blk.2."} {
		tensor(block+"attn_norm.weight", g.embedding)
		tensor(block+"attn_qkv.weight", g.embedding, 2*g.state*g.groups+g.inner)
		tensor(block+"attn_gate.weight", g.embedding, g.inner)
		gates(block+"ssm_beta.weight", g.embedding, g.rank)
		gates(block+"ssm_alpha.weight", g.embedding, g.rank)
		tensor(block+"ssm_conv1d.weight", hybridConv, 2*g.state*g.groups+g.inner)
		tensor(block+"ssm_dt.bias", g.rank)
		tensor(block+"ssm_a", g.rank)
		tensor(block+"ssm_norm.weight", g.state)
		tensor(block+"ssm_out.weight", g.inner, g.embedding)
		tensor(block+"post_attention_norm.weight", g.embedding)
		tensor(block+"ffn_gate.weight", g.embedding, g.ffn)
		tensor(block+"ffn_up.weight", g.embedding, g.ffn)
		tensor(block+"ffn_down.weight", g.ffn, g.embedding)
	}
	// blk.4 is the NextN draft layer: beyond the trunk the cadence falls
	// back to attention, so it carries the gated-attention set plus the
	// nextn projection tensors.
	for _, attention := range []string{"blk.3.", "blk.4."} {
		tensor(attention+"attn_norm.weight", g.embedding)
		tensor(attention+"attn_q.weight", g.embedding, heads*width*2)
		tensor(attention+"attn_k.weight", g.embedding, kvHeads*width)
		tensor(attention+"attn_v.weight", g.embedding, kvHeads*width)
		tensor(attention+"attn_output.weight", heads*width, g.embedding)
		tensor(attention+"attn_q_norm.weight", width)
		tensor(attention+"attn_k_norm.weight", width)
		tensor(attention+"post_attention_norm.weight", g.embedding)
		tensor(attention+"ffn_gate.weight", g.embedding, g.ffn)
		tensor(attention+"ffn_up.weight", g.embedding, g.ffn)
		tensor(attention+"ffn_down.weight", g.ffn, g.embedding)
	}
	tensor("blk.4.nextn.eh_proj.weight", 2*g.embedding, g.embedding)
	tensor("blk.4.nextn.enorm.weight", g.embedding)
	tensor("blk.4.nextn.hnorm.weight", g.embedding)

	path := filepath.Join(t.TempDir(), "hermetic-hybrid.gguf")
	file, err := os.Create(path)
	testcheck.NoError(t, err)
	if err = gguf.Write(file, metadata, tensors, gguf.WriteOptions{}); err != nil {
		_ = file.Close()
		t.Fatal(err)
	}
	testcheck.NoError(t, file.Close())
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
	testcheck.NoError(t, err)
	defer runner.Close()
	ctx := t.Context()
	runner.mu.Lock()
	defer runner.mu.Unlock()
	release := func(cache *deviceKVCache) {
		testcheck.NoError(t, cache.Release(t.Context()))
	}

	// Boundary: committed prefix with live recurrent state.
	boundary, err := runner.forwardDeviceCachedModeLocked(
		ctx, deviceOutputGreedySpan, []tokenizer.TokenID{1, 4}, nil, nil,
	)
	testcheck.NoError(t, err)
	defer release(boundary)

	// Draft span from the boundary, twice: identical selections prove the
	// append is a pure function of the boundary state.
	first, err := runner.forwardDeviceCachedModeLocked(
		ctx, deviceOutputGreedySpan, []tokenizer.TokenID{5, 6}, boundary, nil,
	)
	testcheck.NoError(t, err)
	firstSelected := append([]tokenizer.TokenID(nil), first.SpanSelected...)
	// Reject the draft: roll back by releasing the speculative cache.
	release(first)
	second, err := runner.forwardDeviceCachedModeLocked(
		ctx, deviceOutputGreedySpan, []tokenizer.TokenID{5, 6}, boundary, nil,
	)
	testcheck.NoError(t, err)
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
		ctx, deviceOutputGreedySpan, []tokenizer.TokenID{7}, boundary, nil,
	)
	testcheck.NoError(t, err)
	defer release(alternative)
	fresh, err := runner.forwardDeviceCachedModeLocked(
		ctx, deviceOutputGreedySpan, []tokenizer.TokenID{1, 4, 7}, nil, nil,
	)
	testcheck.NoError(t, err)
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
