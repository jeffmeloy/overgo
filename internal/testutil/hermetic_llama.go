package testutil

import (
	"overgo/internal/gguf"
	"testing"
)

// HermeticLlamaGGUF writes the shared dense CPU/device parity fixture.
func HermeticLlamaGGUF(t testing.TB, context uint32) string {
	t.Helper()
	const (
		hermeticEmbedding = uint64(8)
		hermeticHeads     = uint64(2)
		hermeticHeadWidth = hermeticEmbedding / hermeticHeads
		hermeticKVHeads   = uint64(1)
		hermeticFFN       = uint64(12)
		hermeticVocab     = uint64(8)
	)
	metadata := []gguf.Metadata{
		GGUFScalar("general.architecture", gguf.ValueTypeString, "llama"),
		GGUFScalar("general.name", gguf.ValueTypeString, "hermetic-cuda"),
		GGUFScalar("llama.block_count", gguf.ValueTypeUint32, uint32(1)),
		GGUFScalar("llama.context_length", gguf.ValueTypeUint32, context),
		GGUFScalar("llama.embedding_length", gguf.ValueTypeUint32, uint32(hermeticEmbedding)),
		GGUFScalar("llama.feed_forward_length", gguf.ValueTypeUint32, uint32(hermeticFFN)),
		GGUFScalar("llama.attention.head_count", gguf.ValueTypeUint32, uint32(hermeticHeads)),
		GGUFScalar("llama.attention.head_count_kv", gguf.ValueTypeUint32, uint32(hermeticKVHeads)),
		GGUFScalar("llama.rope.freq_base", gguf.ValueTypeFloat32, float32(10_000)),
		GGUFScalar("llama.attention.layer_norm_rms_epsilon", gguf.ValueTypeFloat32, float32(1e-5)),
		GGUFScalar("tokenizer.ggml.model", gguf.ValueTypeString, "llama"),
		GGUFArray("tokenizer.ggml.tokens", gguf.ValueTypeString,
			[]string{"<unk>", "<s>", "</s>", "â–", "a", "b", "c", "d"}),
		GGUFArray("tokenizer.ggml.scores", gguf.ValueTypeFloat32, make([]float32, hermeticVocab)),
		GGUFArray("tokenizer.ggml.token_type", gguf.ValueTypeInt32,
			[]int32{2, 3, 3, 1, 1, 1, 1, 1}),
		GGUFScalar("tokenizer.ggml.bos_token_id", gguf.ValueTypeUint32, uint32(1)),
		GGUFScalar("tokenizer.ggml.eos_token_id", gguf.ValueTypeUint32, uint32(2)),
	}
	tensors := []gguf.TensorData{
		GGUFTensorF32("token_embd.weight", []uint64{hermeticEmbedding, hermeticVocab}, 1),
		GGUFTensorF32("output_norm.weight", []uint64{hermeticEmbedding}, 2),
		GGUFTensorF32("output.weight", []uint64{hermeticEmbedding, hermeticVocab}, 3),
		GGUFTensorF32("blk.0.attn_norm.weight", []uint64{hermeticEmbedding}, 4),
		GGUFTensorF32("blk.0.attn_q.weight", []uint64{hermeticEmbedding, hermeticEmbedding}, 5),
		GGUFTensorF32("blk.0.attn_k.weight", []uint64{hermeticEmbedding, hermeticHeadWidth}, 6),
		GGUFTensorF32("blk.0.attn_v.weight", []uint64{hermeticEmbedding, hermeticHeadWidth}, 7),
		GGUFTensorF32("blk.0.attn_output.weight", []uint64{hermeticEmbedding, hermeticEmbedding}, 8),
		GGUFTensorF32("blk.0.ffn_norm.weight", []uint64{hermeticEmbedding}, 9),
		GGUFTensorF32("blk.0.ffn_gate.weight", []uint64{hermeticEmbedding, hermeticFFN}, 10),
		GGUFTensorF32("blk.0.ffn_up.weight", []uint64{hermeticEmbedding, hermeticFFN}, 11),
		GGUFTensorF32("blk.0.ffn_down.weight", []uint64{hermeticFFN, hermeticEmbedding}, 12),
	}
	return TempGGUF(t, "hermetic.gguf", metadata, tensors)
}
