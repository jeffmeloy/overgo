package model

import (
	"errors"
	"math"
	"strings"
	"testing"

	"llamacpp2go/internal/gguf"
)

func TestReadQwen3Spec(t *testing.T) {
	file := &gguf.File{Metadata: []gguf.Metadata{
		metadata("general.architecture", gguf.ValueTypeString, "qwen3"),
		metadata("general.name", gguf.ValueTypeString, "fixture"),
		metadata("qwen3.block_count", gguf.ValueTypeUint32, uint32(36)),
		metadata("qwen3.context_length", gguf.ValueTypeUint32, uint32(40960)),
		metadata("qwen3.embedding_length", gguf.ValueTypeUint32, uint32(2560)),
		metadata("qwen3.feed_forward_length", gguf.ValueTypeUint32, uint32(9728)),
		metadata("qwen3.attention.head_count", gguf.ValueTypeUint32, uint32(32)),
		metadata("qwen3.attention.head_count_kv", gguf.ValueTypeUint32, uint32(8)),
		metadata("qwen3.attention.key_length", gguf.ValueTypeUint32, uint32(128)),
		metadata("qwen3.attention.value_length", gguf.ValueTypeUint32, uint32(128)),
		metadata("qwen3.rope.freq_base", gguf.ValueTypeFloat32, float32(1_000_000)),
		metadata("qwen3.attention.layer_norm_rms_epsilon", gguf.ValueTypeFloat32, float32(1e-6)),
		{
			Key: "tokenizer.ggml.tokens",
			Value: gguf.Value{
				Type:      gguf.ValueTypeArray,
				ArrayType: gguf.ValueTypeString,
				Data:      []string{"a", "b", "c"},
			},
		},
	}}
	spec, err := ReadSpec(file)
	if err != nil {
		t.Fatal(err)
	}
	if spec.Architecture != "qwen3" || spec.BlockCount != 36 || spec.VocabularySize != 3 {
		t.Fatalf("unexpected spec: %+v", spec)
	}
}

func TestReadQwen3MoESpec(t *testing.T) {
	file := &gguf.File{Metadata: []gguf.Metadata{
		metadata("general.architecture", gguf.ValueTypeString, "qwen3moe"),
		metadata("qwen3moe.block_count", gguf.ValueTypeUint32, uint32(2)),
		metadata("qwen3moe.context_length", gguf.ValueTypeUint32, uint32(4096)),
		metadata("qwen3moe.embedding_length", gguf.ValueTypeUint32, uint32(8)),
		metadata("qwen3moe.feed_forward_length", gguf.ValueTypeUint32, uint32(24)),
		metadata("qwen3moe.expert_feed_forward_length", gguf.ValueTypeUint32, uint32(12)),
		metadata("qwen3moe.expert_count", gguf.ValueTypeUint32, uint32(8)),
		metadata("qwen3moe.expert_used_count", gguf.ValueTypeUint32, uint32(2)),
		metadata("qwen3moe.expert_weights_scale", gguf.ValueTypeFloat32, float32(1.5)),
		metadata("qwen3moe.attention.head_count", gguf.ValueTypeUint32, uint32(2)),
		metadata("qwen3moe.attention.head_count_kv", gguf.ValueTypeUint32, uint32(1)),
		metadata("qwen3moe.attention.key_length", gguf.ValueTypeUint32, uint32(4)),
		metadata("qwen3moe.attention.value_length", gguf.ValueTypeUint32, uint32(4)),
		metadata("qwen3moe.rope.freq_base", gguf.ValueTypeFloat32, float32(1_000_000)),
		metadata("qwen3moe.attention.layer_norm_rms_epsilon", gguf.ValueTypeFloat32, float32(1e-6)),
	}}
	spec, err := ReadSpec(file)
	if err != nil {
		t.Fatal(err)
	}
	if spec.Architecture != "qwen3moe" || spec.ExpertCount != 8 ||
		spec.ExpertUsedCount != 2 || spec.ExpertFeedForward != 12 ||
		spec.ExpertWeightsScale != 1.5 {
		t.Fatalf("unexpected Qwen3-MoE spec: %+v", spec)
	}
}

func TestReadArcticSpec(t *testing.T) {
	file := &gguf.File{Metadata: []gguf.Metadata{
		metadata("general.architecture", gguf.ValueTypeString, "arctic"),
		metadata("arctic.block_count", gguf.ValueTypeUint32, uint32(2)),
		metadata("arctic.context_length", gguf.ValueTypeUint32, uint32(4096)),
		metadata("arctic.embedding_length", gguf.ValueTypeUint32, uint32(8)),
		metadata("arctic.feed_forward_length", gguf.ValueTypeUint32, uint32(12)),
		metadata("arctic.expert_count", gguf.ValueTypeUint32, uint32(128)),
		metadata("arctic.expert_used_count", gguf.ValueTypeUint32, uint32(3)),
		metadata("arctic.attention.head_count", gguf.ValueTypeUint32, uint32(2)),
		metadata("arctic.attention.head_count_kv", gguf.ValueTypeUint32, uint32(1)),
		metadata("arctic.attention.key_length", gguf.ValueTypeUint32, uint32(4)),
		metadata("arctic.attention.value_length", gguf.ValueTypeUint32, uint32(4)),
		metadata("arctic.rope.freq_base", gguf.ValueTypeFloat32, float32(10000)),
		metadata("arctic.attention.layer_norm_rms_epsilon", gguf.ValueTypeFloat32, float32(1e-5)),
	}}
	spec, err := ReadSpec(file)
	if err != nil {
		t.Fatal(err)
	}
	if spec.Architecture != "arctic" || spec.ExpertCount != 128 ||
		spec.ExpertUsedCount != 3 || spec.ExpertFeedForward != 12 ||
		spec.ExpertWeightsScale != 1 || !usesNormalRoPE(spec.Architecture) {
		t.Fatalf("unexpected Arctic spec: %+v", spec)
	}
}

func TestReadBailingMoESpec(t *testing.T) {
	file := &gguf.File{Metadata: []gguf.Metadata{
		metadata("general.architecture", gguf.ValueTypeString, "bailingmoe"),
		metadata("bailingmoe.block_count", gguf.ValueTypeUint32, uint32(2)),
		metadata("bailingmoe.context_length", gguf.ValueTypeUint32, uint32(4096)),
		metadata("bailingmoe.embedding_length", gguf.ValueTypeUint32, uint32(8)),
		metadata("bailingmoe.feed_forward_length", gguf.ValueTypeUint32, uint32(16)),
		metadata("bailingmoe.expert_feed_forward_length", gguf.ValueTypeUint32, uint32(6)),
		metadata("bailingmoe.expert_count", gguf.ValueTypeUint32, uint32(8)),
		metadata("bailingmoe.expert_used_count", gguf.ValueTypeUint32, uint32(2)),
		metadata("bailingmoe.expert_shared_count", gguf.ValueTypeUint32, uint32(2)),
		metadata("bailingmoe.expert_weights_scale", gguf.ValueTypeFloat32, float32(1.25)),
		metadata("bailingmoe.expert_weights_norm", gguf.ValueTypeBool, true),
		metadata("bailingmoe.leading_dense_block_count", gguf.ValueTypeUint32, uint32(1)),
		metadata("bailingmoe.attention.head_count", gguf.ValueTypeUint32, uint32(2)),
		metadata("bailingmoe.attention.head_count_kv", gguf.ValueTypeUint32, uint32(1)),
		metadata("bailingmoe.attention.key_length", gguf.ValueTypeUint32, uint32(4)),
		metadata("bailingmoe.attention.value_length", gguf.ValueTypeUint32, uint32(4)),
		metadata("bailingmoe.rope.freq_base", gguf.ValueTypeFloat32, float32(10000)),
		metadata("bailingmoe.attention.layer_norm_rms_epsilon", gguf.ValueTypeFloat32, float32(1e-6)),
	}}
	spec, err := ReadSpec(file)
	if err != nil {
		t.Fatal(err)
	}
	if spec.Architecture != "bailingmoe" || spec.ExpertCount != 8 ||
		spec.ExpertUsedCount != 2 || spec.ExpertFeedForward != 6 ||
		spec.SharedExpertCount != 2 || spec.SharedExpertFF != 12 ||
		spec.ExpertWeightsScale != 1.25 || !spec.ExpertWeightsNorm ||
		spec.LeadingDenseBlocks != 1 || !usesNormalRoPE(spec.Architecture) {
		t.Fatalf("unexpected BailingMoE spec: %+v", spec)
	}
}

func TestReadDeepSeekSpec(t *testing.T) {
	file := &gguf.File{Metadata: []gguf.Metadata{
		metadata("general.architecture", gguf.ValueTypeString, "deepseek"),
		metadata("deepseek.block_count", gguf.ValueTypeUint32, uint32(3)),
		metadata("deepseek.context_length", gguf.ValueTypeUint32, uint32(4096)),
		metadata("deepseek.embedding_length", gguf.ValueTypeUint32, uint32(8)),
		metadata("deepseek.feed_forward_length", gguf.ValueTypeUint32, uint32(16)),
		metadata("deepseek.expert_feed_forward_length", gguf.ValueTypeUint32, uint32(6)),
		metadata("deepseek.expert_count", gguf.ValueTypeUint32, uint32(8)),
		metadata("deepseek.expert_used_count", gguf.ValueTypeUint32, uint32(2)),
		metadata("deepseek.expert_shared_count", gguf.ValueTypeUint32, uint32(2)),
		metadata("deepseek.expert_weights_scale", gguf.ValueTypeFloat32, float32(1.3)),
		metadata("deepseek.leading_dense_block_count", gguf.ValueTypeUint32, uint32(1)),
		metadata("deepseek.attention.head_count", gguf.ValueTypeUint32, uint32(2)),
		metadata("deepseek.attention.head_count_kv", gguf.ValueTypeUint32, uint32(1)),
		metadata("deepseek.attention.key_length", gguf.ValueTypeUint32, uint32(4)),
		metadata("deepseek.attention.value_length", gguf.ValueTypeUint32, uint32(4)),
		metadata("deepseek.rope.freq_base", gguf.ValueTypeFloat32, float32(10000)),
		metadata("deepseek.attention.layer_norm_rms_epsilon", gguf.ValueTypeFloat32, float32(1e-6)),
	}}
	spec, err := ReadSpec(file)
	if err != nil {
		t.Fatal(err)
	}
	if spec.Architecture != "deepseek" || spec.ExpertCount != 8 ||
		spec.ExpertUsedCount != 2 || spec.ExpertFeedForward != 6 ||
		spec.SharedExpertCount != 2 || spec.SharedExpertFF != 12 ||
		spec.ExpertWeightsScale != 1.3 || spec.ExpertWeightsNorm ||
		spec.LeadingDenseBlocks != 1 || !usesNormalRoPE(spec.Architecture) {
		t.Fatalf("unexpected DeepSeek spec: %+v", spec)
	}
}

func TestReadDBRXSpec(t *testing.T) {
	file := &gguf.File{Metadata: []gguf.Metadata{
		metadata("general.architecture", gguf.ValueTypeString, "dbrx"),
		metadata("dbrx.block_count", gguf.ValueTypeUint32, uint32(2)),
		metadata("dbrx.context_length", gguf.ValueTypeUint32, uint32(4096)),
		metadata("dbrx.embedding_length", gguf.ValueTypeUint32, uint32(8)),
		metadata("dbrx.feed_forward_length", gguf.ValueTypeUint32, uint32(6)),
		metadata("dbrx.expert_count", gguf.ValueTypeUint32, uint32(8)),
		metadata("dbrx.expert_used_count", gguf.ValueTypeUint32, uint32(2)),
		metadata("dbrx.expert_weights_scale", gguf.ValueTypeFloat32, float32(1.25)),
		metadata("dbrx.attention.head_count", gguf.ValueTypeUint32, uint32(2)),
		metadata("dbrx.attention.head_count_kv", gguf.ValueTypeUint32, uint32(1)),
		metadata("dbrx.attention.key_length", gguf.ValueTypeUint32, uint32(4)),
		metadata("dbrx.attention.value_length", gguf.ValueTypeUint32, uint32(4)),
		metadata("dbrx.attention.clamp_kqv", gguf.ValueTypeFloat32, float32(8)),
		metadata("dbrx.rope.freq_base", gguf.ValueTypeFloat32, float32(10000)),
		metadata("dbrx.attention.layer_norm_epsilon", gguf.ValueTypeFloat32, float32(1e-5)),
	}}
	spec, err := ReadSpec(file)
	if err != nil {
		t.Fatal(err)
	}
	if spec.Architecture != "dbrx" || spec.ExpertCount != 8 || spec.ExpertUsedCount != 2 ||
		spec.ExpertFeedForward != 6 || spec.ExpertWeightsScale != 1.25 || !spec.ExpertWeightsNorm ||
		spec.AttentionClamp != 8 || !spec.UsesLayerNorm() || spec.RequiresLayerNormBias() ||
		usesNormalRoPE(spec.Architecture) {
		t.Fatalf("unexpected DBRX spec: %+v", spec)
	}
}

func TestReadBailingMoE2SpecTrimsNextNLayers(t *testing.T) {
	file := &gguf.File{Metadata: []gguf.Metadata{
		metadata("general.architecture", gguf.ValueTypeString, "bailingmoe2"),
		metadata("bailingmoe2.block_count", gguf.ValueTypeUint32, uint32(3)),
		metadata("bailingmoe2.nextn_predict_layers", gguf.ValueTypeUint32, uint32(1)),
		metadata("bailingmoe2.context_length", gguf.ValueTypeUint32, uint32(4096)),
		metadata("bailingmoe2.embedding_length", gguf.ValueTypeUint32, uint32(8)),
		metadata("bailingmoe2.feed_forward_length", gguf.ValueTypeUint32, uint32(16)),
		metadata("bailingmoe2.expert_feed_forward_length", gguf.ValueTypeUint32, uint32(6)),
		metadata("bailingmoe2.expert_shared_feed_forward_length", gguf.ValueTypeUint32, uint32(5)),
		metadata("bailingmoe2.expert_count", gguf.ValueTypeUint32, uint32(8)),
		metadata("bailingmoe2.expert_used_count", gguf.ValueTypeUint32, uint32(2)),
		metadata("bailingmoe2.expert_shared_count", gguf.ValueTypeUint32, uint32(2)),
		metadata("bailingmoe2.expert_weights_scale", gguf.ValueTypeFloat32, float32(1.25)),
		metadata("bailingmoe2.expert_weights_norm", gguf.ValueTypeBool, true),
		metadata("bailingmoe2.expert_gating_func", gguf.ValueTypeUint32, uint32(2)),
		metadata("bailingmoe2.leading_dense_block_count", gguf.ValueTypeUint32, uint32(1)),
		metadata("bailingmoe2.attention.head_count", gguf.ValueTypeUint32, uint32(2)),
		metadata("bailingmoe2.attention.head_count_kv", gguf.ValueTypeUint32, uint32(1)),
		metadata("bailingmoe2.attention.key_length", gguf.ValueTypeUint32, uint32(4)),
		metadata("bailingmoe2.attention.value_length", gguf.ValueTypeUint32, uint32(4)),
		metadata("bailingmoe2.rope.freq_base", gguf.ValueTypeFloat32, float32(10000)),
		metadata("bailingmoe2.rope.dimension_count", gguf.ValueTypeUint32, uint32(4)),
		metadata("bailingmoe2.attention.layer_norm_rms_epsilon", gguf.ValueTypeFloat32, float32(1e-6)),
	}}
	spec, err := ReadSpec(file)
	if err != nil {
		t.Fatal(err)
	}
	if spec.Architecture != "bailingmoe2" || spec.BlockCount != 2 ||
		spec.LeadingDenseBlocks != 1 || spec.ExpertCount != 8 || spec.ExpertUsedCount != 2 ||
		spec.ExpertFeedForward != 6 || spec.SharedExpertCount != 2 || spec.SharedExpertFF != 10 ||
		spec.ExpertGatingFunc != 2 || !spec.ExpertWeightsNorm || spec.ExpertWeightsScale != 1.25 ||
		spec.RopeDimensionCount != 4 || usesNormalRoPE(spec.Architecture) {
		t.Fatalf("unexpected BailingMoE2 spec: %+v", spec)
	}
}

func TestReadQwen2MoESpec(t *testing.T) {
	file := &gguf.File{Metadata: []gguf.Metadata{
		metadata("general.architecture", gguf.ValueTypeString, "qwen2moe"),
		metadata("qwen2moe.block_count", gguf.ValueTypeUint32, uint32(1)),
		metadata("qwen2moe.context_length", gguf.ValueTypeUint32, uint32(4096)),
		metadata("qwen2moe.embedding_length", gguf.ValueTypeUint32, uint32(8)),
		metadata("qwen2moe.feed_forward_length", gguf.ValueTypeUint32, uint32(16)),
		metadata("qwen2moe.expert_feed_forward_length", gguf.ValueTypeUint32, uint32(6)),
		metadata("qwen2moe.expert_shared_feed_forward_length", gguf.ValueTypeUint32, uint32(10)),
		metadata("qwen2moe.expert_count", gguf.ValueTypeUint32, uint32(4)),
		metadata("qwen2moe.expert_used_count", gguf.ValueTypeUint32, uint32(2)),
		metadata("qwen2moe.attention.head_count", gguf.ValueTypeUint32, uint32(2)),
		metadata("qwen2moe.attention.head_count_kv", gguf.ValueTypeUint32, uint32(1)),
		metadata("qwen2moe.attention.key_length", gguf.ValueTypeUint32, uint32(4)),
		metadata("qwen2moe.attention.value_length", gguf.ValueTypeUint32, uint32(4)),
		metadata("qwen2moe.rope.freq_base", gguf.ValueTypeFloat32, float32(1_000_000)),
		metadata("qwen2moe.attention.layer_norm_rms_epsilon", gguf.ValueTypeFloat32, float32(1e-6)),
	}}
	spec, err := ReadSpec(file)
	if err != nil {
		t.Fatal(err)
	}
	if spec.ExpertFeedForward != 6 || spec.SharedExpertFF != 10 || spec.ExpertWeightsScale != 1 {
		t.Fatalf("unexpected Qwen2-MoE spec: %+v", spec)
	}
}

func TestReadOLMoESpec(t *testing.T) {
	file := &gguf.File{Metadata: []gguf.Metadata{
		metadata("general.architecture", gguf.ValueTypeString, "olmoe"),
		metadata("olmoe.block_count", gguf.ValueTypeUint32, uint32(1)),
		metadata("olmoe.context_length", gguf.ValueTypeUint32, uint32(4096)),
		metadata("olmoe.embedding_length", gguf.ValueTypeUint32, uint32(8)),
		metadata("olmoe.feed_forward_length", gguf.ValueTypeUint32, uint32(12)),
		metadata("olmoe.expert_count", gguf.ValueTypeUint32, uint32(8)),
		metadata("olmoe.expert_used_count", gguf.ValueTypeUint32, uint32(2)),
		metadata("olmoe.attention.head_count", gguf.ValueTypeUint32, uint32(2)),
		metadata("olmoe.attention.head_count_kv", gguf.ValueTypeUint32, uint32(2)),
		metadata("olmoe.attention.key_length", gguf.ValueTypeUint32, uint32(4)),
		metadata("olmoe.attention.value_length", gguf.ValueTypeUint32, uint32(4)),
		metadata("olmoe.rope.freq_base", gguf.ValueTypeFloat32, float32(10000)),
		metadata("olmoe.attention.layer_norm_rms_epsilon", gguf.ValueTypeFloat32, float32(1e-6)),
	}}
	spec, err := ReadSpec(file)
	if err != nil {
		t.Fatal(err)
	}
	if spec.Architecture != "olmoe" || spec.ExpertFeedForward != 12 ||
		spec.ExpertCount != 8 || spec.ExpertUsedCount != 2 {
		t.Fatalf("unexpected OLMoE spec: %+v", spec)
	}
}

func TestReadPhiMoESpec(t *testing.T) {
	file := &gguf.File{Metadata: []gguf.Metadata{
		metadata("general.architecture", gguf.ValueTypeString, "phimoe"),
		metadata("phimoe.block_count", gguf.ValueTypeUint32, uint32(2)),
		metadata("phimoe.context_length", gguf.ValueTypeUint32, uint32(131072)),
		metadata("phimoe.embedding_length", gguf.ValueTypeUint32, uint32(8)),
		metadata("phimoe.feed_forward_length", gguf.ValueTypeUint32, uint32(12)),
		metadata("phimoe.expert_count", gguf.ValueTypeUint32, uint32(4)),
		metadata("phimoe.expert_used_count", gguf.ValueTypeUint32, uint32(2)),
		metadata("phimoe.attention.head_count", gguf.ValueTypeUint32, uint32(2)),
		metadata("phimoe.attention.head_count_kv", gguf.ValueTypeUint32, uint32(2)),
		metadata("phimoe.rope.freq_base", gguf.ValueTypeFloat32, float32(10000)),
		metadata("phimoe.rope.dimension_count", gguf.ValueTypeUint32, uint32(4)),
		metadata("phimoe.rope.scaling.type", gguf.ValueTypeString, "longrope"),
		metadata("phimoe.rope.scaling.original_context_length", gguf.ValueTypeUint32, uint32(4096)),
		metadata("phimoe.rope.scaling.attn_factor", gguf.ValueTypeFloat32, float32(1.19)),
		metadata("phimoe.attention.layer_norm_rms_epsilon", gguf.ValueTypeFloat32, float32(1e-5)),
	}}
	spec, err := ReadSpec(file)
	if err != nil {
		t.Fatal(err)
	}
	if spec.Architecture != "phimoe" || spec.ExpertFeedForward != 12 ||
		spec.ExpertCount != 4 || spec.ExpertUsedCount != 2 ||
		spec.RopeDimensionCount != 4 || spec.OriginalContextLength != 4096 ||
		!spec.RequiresLayerNormBias() {
		t.Fatalf("unexpected PhiMoE spec: %+v", spec)
	}
}

func TestReadEXAOneMoESpec(t *testing.T) {
	file := &gguf.File{Metadata: []gguf.Metadata{
		metadata("general.architecture", gguf.ValueTypeString, "exaone-moe"),
		metadata("exaone-moe.block_count", gguf.ValueTypeUint32, uint32(5)),
		metadata("exaone-moe.nextn_predict_layers", gguf.ValueTypeUint32, uint32(1)),
		metadata("exaone-moe.context_length", gguf.ValueTypeUint32, uint32(32768)),
		metadata("exaone-moe.embedding_length", gguf.ValueTypeUint32, uint32(8)),
		metadata("exaone-moe.feed_forward_length", gguf.ValueTypeUint32, uint32(16)),
		metadata("exaone-moe.expert_feed_forward_length", gguf.ValueTypeUint32, uint32(6)),
		metadata("exaone-moe.expert_shared_feed_forward_length", gguf.ValueTypeUint32, uint32(12)),
		metadata("exaone-moe.expert_shared_count", gguf.ValueTypeUint32, uint32(2)),
		metadata("exaone-moe.expert_count", gguf.ValueTypeUint32, uint32(4)),
		metadata("exaone-moe.expert_used_count", gguf.ValueTypeUint32, uint32(2)),
		metadata("exaone-moe.expert_gating_func", gguf.ValueTypeUint32, uint32(2)),
		metadata("exaone-moe.expert_weights_scale", gguf.ValueTypeFloat32, float32(1.5)),
		metadata("exaone-moe.expert_weights_norm", gguf.ValueTypeBool, true),
		metadata("exaone-moe.leading_dense_block_count", gguf.ValueTypeUint32, uint32(1)),
		metadata("exaone-moe.attention.head_count", gguf.ValueTypeUint32, uint32(2)),
		metadata("exaone-moe.attention.head_count_kv", gguf.ValueTypeUint32, uint32(1)),
		metadata("exaone-moe.rope.freq_base", gguf.ValueTypeFloat32, float32(10000)),
		metadata("exaone-moe.rope.freq_base_swa", gguf.ValueTypeFloat32, float32(500000)),
		metadata("exaone-moe.attention.sliding_window", gguf.ValueTypeUint32, uint32(128)),
		{Key: "exaone-moe.attention.sliding_window_pattern", Value: gguf.Value{Type: gguf.ValueTypeArray, ArrayType: gguf.ValueTypeBool, Data: []bool{true, true, false, true}}},
		metadata("exaone-moe.attention.layer_norm_rms_epsilon", gguf.ValueTypeFloat32, float32(1e-6)),
	}}
	spec, err := ReadSpec(file)
	if err != nil {
		t.Fatal(err)
	}
	if spec.BlockCount != 4 || spec.ExpertFeedForward != 6 || spec.SharedExpertFF != 12 ||
		spec.ExpertGatingFunc != 2 || !spec.ExpertWeightsNorm ||
		!spec.IsSlidingLayer(0) || spec.IsSlidingLayer(2) || spec.UsesRoPE(2) {
		t.Fatalf("unexpected EXAONE-MoE spec: %+v", spec)
	}
}

func TestReadDreamSpecIsNonCausal(t *testing.T) {
	file := &gguf.File{Metadata: []gguf.Metadata{
		metadata("general.architecture", gguf.ValueTypeString, "dream"),
		metadata("dream.block_count", gguf.ValueTypeUint32, uint32(28)),
		metadata("dream.context_length", gguf.ValueTypeUint32, uint32(4096)),
		metadata("dream.embedding_length", gguf.ValueTypeUint32, uint32(8)),
		metadata("dream.feed_forward_length", gguf.ValueTypeUint32, uint32(16)),
		metadata("dream.attention.head_count", gguf.ValueTypeUint32, uint32(2)),
		metadata("dream.attention.head_count_kv", gguf.ValueTypeUint32, uint32(1)),
		metadata("dream.attention.key_length", gguf.ValueTypeUint32, uint32(4)),
		metadata("dream.attention.value_length", gguf.ValueTypeUint32, uint32(4)),
		metadata("dream.rope.freq_base", gguf.ValueTypeFloat32, float32(10000)),
		metadata("dream.attention.layer_norm_rms_epsilon", gguf.ValueTypeFloat32, float32(1e-6)),
	}}
	spec, err := ReadSpec(file)
	if err != nil {
		t.Fatal(err)
	}
	if spec.Architecture != "dream" || !spec.NonCausalAttention {
		t.Fatalf("unexpected Dream spec: %+v", spec)
	}
}

func TestReadEuroBERTSpecIsNonCausalNeoX(t *testing.T) {
	file := &gguf.File{Metadata: []gguf.Metadata{
		metadata("general.architecture", gguf.ValueTypeString, "eurobert"),
		metadata("eurobert.block_count", gguf.ValueTypeUint32, uint32(12)),
		metadata("eurobert.context_length", gguf.ValueTypeUint32, uint32(4096)),
		metadata("eurobert.embedding_length", gguf.ValueTypeUint32, uint32(8)),
		metadata("eurobert.feed_forward_length", gguf.ValueTypeUint32, uint32(16)),
		metadata("eurobert.vocab_size", gguf.ValueTypeUint32, uint32(32)),
		metadata("eurobert.attention.head_count", gguf.ValueTypeUint32, uint32(2)),
		metadata("eurobert.attention.head_count_kv", gguf.ValueTypeUint32, uint32(1)),
		metadata("eurobert.rope.freq_base", gguf.ValueTypeFloat32, float32(10000)),
		metadata("eurobert.attention.layer_norm_rms_epsilon", gguf.ValueTypeFloat32, float32(1e-6)),
	}}
	spec, err := ReadSpec(file)
	if err != nil {
		t.Fatal(err)
	}
	if spec.Architecture != "eurobert" || !spec.NonCausalAttention ||
		spec.KeyLength != 4 || spec.ValueLength != 4 || usesNormalRoPE(spec.Architecture) {
		t.Fatalf("unexpected EuroBERT spec: %+v", spec)
	}
}

func TestReadBERTSpecUsesLearnedPositions(t *testing.T) {
	file := &gguf.File{Metadata: []gguf.Metadata{
		metadata("general.architecture", gguf.ValueTypeString, "bert"),
		metadata("bert.block_count", gguf.ValueTypeUint32, uint32(12)),
		metadata("bert.context_length", gguf.ValueTypeUint32, uint32(512)),
		metadata("bert.embedding_length", gguf.ValueTypeUint32, uint32(8)),
		metadata("bert.feed_forward_length", gguf.ValueTypeUint32, uint32(16)),
		metadata("bert.vocab_size", gguf.ValueTypeUint32, uint32(32)),
		metadata("bert.attention.head_count", gguf.ValueTypeUint32, uint32(2)),
		metadata("bert.attention.layer_norm_epsilon", gguf.ValueTypeFloat32, float32(1e-5)),
		metadata("tokenizer.ggml.token_type_count", gguf.ValueTypeUint32, uint32(2)),
	}}
	spec, err := ReadSpec(file)
	if err != nil {
		t.Fatal(err)
	}
	if spec.Architecture != "bert" || !spec.NonCausalAttention || !spec.RopeDisabled ||
		spec.HeadCountKV != 2 || spec.KeyLength != 4 || spec.ValueLength != 4 ||
		spec.TokenTypeCount != 2 || !spec.UsesLayerNorm() || !usesPostOnlyNorm(spec.Architecture) {
		t.Fatalf("unexpected BERT spec: %+v", spec)
	}
}

func TestReadNeoBERTSpecIsNonCausalNormalRoPE(t *testing.T) {
	file := &gguf.File{Metadata: []gguf.Metadata{
		metadata("general.architecture", gguf.ValueTypeString, "neo-bert"),
		metadata("neo-bert.block_count", gguf.ValueTypeUint32, uint32(28)),
		metadata("neo-bert.context_length", gguf.ValueTypeUint32, uint32(4096)),
		metadata("neo-bert.embedding_length", gguf.ValueTypeUint32, uint32(8)),
		metadata("neo-bert.feed_forward_length", gguf.ValueTypeUint32, uint32(16)),
		metadata("neo-bert.vocab_size", gguf.ValueTypeUint32, uint32(32)),
		metadata("neo-bert.attention.head_count", gguf.ValueTypeUint32, uint32(2)),
		metadata("neo-bert.attention.layer_norm_rms_epsilon", gguf.ValueTypeFloat32, float32(1e-6)),
	}}
	spec, err := ReadSpec(file)
	if err != nil {
		t.Fatal(err)
	}
	if spec.Architecture != "neo-bert" || !spec.NonCausalAttention || !spec.IsEncoderOnly() ||
		spec.HeadCountKV != 2 || spec.KeyLength != 4 || spec.RopeDimensionCount != 4 ||
		spec.RopeFrequencyBase != 10000 || !usesNormalRoPE(spec.Architecture) ||
		!usesFusedGateUp(spec.Architecture) {
		t.Fatalf("unexpected NeoBERT spec: %+v", spec)
	}
}

func TestReadNomicBERTSpecIsNonCausalNeoXRoPE(t *testing.T) {
	file := &gguf.File{Metadata: []gguf.Metadata{
		metadata("general.architecture", gguf.ValueTypeString, "nomic-bert"),
		metadata("nomic-bert.block_count", gguf.ValueTypeUint32, uint32(12)),
		metadata("nomic-bert.context_length", gguf.ValueTypeUint32, uint32(8192)),
		metadata("nomic-bert.embedding_length", gguf.ValueTypeUint32, uint32(8)),
		metadata("nomic-bert.feed_forward_length", gguf.ValueTypeUint32, uint32(16)),
		metadata("nomic-bert.vocab_size", gguf.ValueTypeUint32, uint32(32)),
		metadata("nomic-bert.attention.head_count", gguf.ValueTypeUint32, uint32(2)),
		metadata("nomic-bert.attention.layer_norm_epsilon", gguf.ValueTypeFloat32, float32(1e-5)),
		metadata("tokenizer.ggml.token_type_count", gguf.ValueTypeUint32, uint32(2)),
	}}
	spec, err := ReadSpec(file)
	if err != nil {
		t.Fatal(err)
	}
	if spec.Architecture != "nomic-bert" || !spec.NonCausalAttention || !spec.IsEncoderOnly() ||
		spec.HeadCountKV != 2 || spec.KeyLength != 4 || spec.ValueLength != 4 ||
		spec.RopeDimensionCount != 4 || spec.RopeFrequencyBase != 10000 ||
		spec.TokenTypeCount != 2 || !spec.UsesLayerNorm() ||
		usesNormalRoPE(spec.Architecture) || !usesPostOnlyNorm(spec.Architecture) {
		t.Fatalf("unexpected NomicBERT spec: %+v", spec)
	}
}

func TestReadNomicBERTSpecRejectsMoECadence(t *testing.T) {
	file := &gguf.File{Metadata: []gguf.Metadata{
		metadata("general.architecture", gguf.ValueTypeString, "nomic-bert"),
		metadata("nomic-bert.block_count", gguf.ValueTypeUint32, uint32(12)),
		metadata("nomic-bert.context_length", gguf.ValueTypeUint32, uint32(8192)),
		metadata("nomic-bert.embedding_length", gguf.ValueTypeUint32, uint32(8)),
		metadata("nomic-bert.feed_forward_length", gguf.ValueTypeUint32, uint32(16)),
		metadata("nomic-bert.vocab_size", gguf.ValueTypeUint32, uint32(32)),
		metadata("nomic-bert.attention.head_count", gguf.ValueTypeUint32, uint32(2)),
		metadata("nomic-bert.attention.layer_norm_epsilon", gguf.ValueTypeFloat32, float32(1e-5)),
		metadata("nomic-bert.moe_every_n_layers", gguf.ValueTypeUint32, uint32(2)),
		metadata("tokenizer.ggml.token_type_count", gguf.ValueTypeUint32, uint32(2)),
	}}
	if _, err := ReadSpec(file); err == nil {
		t.Fatal("expected NomicBERT MoE cadence rejection")
	}
}

func TestReadJinaBERTV2SpecUsesBidirectionalALiBi(t *testing.T) {
	file := &gguf.File{Metadata: []gguf.Metadata{
		metadata("general.architecture", gguf.ValueTypeString, "jina-bert-v2"),
		metadata("jina-bert-v2.block_count", gguf.ValueTypeUint32, uint32(12)),
		metadata("jina-bert-v2.context_length", gguf.ValueTypeUint32, uint32(8192)),
		metadata("jina-bert-v2.embedding_length", gguf.ValueTypeUint32, uint32(8)),
		metadata("jina-bert-v2.feed_forward_length", gguf.ValueTypeUint32, uint32(16)),
		metadata("jina-bert-v2.vocab_size", gguf.ValueTypeUint32, uint32(32)),
		metadata("jina-bert-v2.attention.head_count", gguf.ValueTypeUint32, uint32(2)),
		metadata("jina-bert-v2.attention.layer_norm_epsilon", gguf.ValueTypeFloat32, float32(1e-5)),
		metadata("tokenizer.ggml.token_type_count", gguf.ValueTypeUint32, uint32(2)),
	}}
	spec, err := ReadSpec(file)
	if err != nil {
		t.Fatal(err)
	}
	if spec.Architecture != "jina-bert-v2" || !spec.NonCausalAttention || !spec.IsEncoderOnly() ||
		!spec.RopeDisabled || spec.MaxALiBiBias != 8 || spec.HeadCountKV != 2 ||
		spec.KeyLength != 4 || spec.ValueLength != 4 || spec.TokenTypeCount != 2 ||
		!spec.UsesLayerNorm() || !usesPostOnlyNorm(spec.Architecture) {
		t.Fatalf("unexpected JinaBERT v2 spec: %+v", spec)
	}
}

func TestReadRND1SpecIsNonCausalMoE(t *testing.T) {
	file := &gguf.File{Metadata: []gguf.Metadata{
		metadata("general.architecture", gguf.ValueTypeString, "rnd1"),
		metadata("rnd1.block_count", gguf.ValueTypeUint32, uint32(48)),
		metadata("rnd1.context_length", gguf.ValueTypeUint32, uint32(4096)),
		metadata("rnd1.embedding_length", gguf.ValueTypeUint32, uint32(8)),
		metadata("rnd1.feed_forward_length", gguf.ValueTypeUint32, uint32(24)),
		metadata("rnd1.expert_feed_forward_length", gguf.ValueTypeUint32, uint32(12)),
		metadata("rnd1.expert_count", gguf.ValueTypeUint32, uint32(8)),
		metadata("rnd1.expert_used_count", gguf.ValueTypeUint32, uint32(2)),
		metadata("rnd1.attention.head_count", gguf.ValueTypeUint32, uint32(2)),
		metadata("rnd1.attention.head_count_kv", gguf.ValueTypeUint32, uint32(1)),
		metadata("rnd1.attention.key_length", gguf.ValueTypeUint32, uint32(4)),
		metadata("rnd1.attention.value_length", gguf.ValueTypeUint32, uint32(4)),
		metadata("rnd1.rope.freq_base", gguf.ValueTypeFloat32, float32(1_000_000)),
		metadata("rnd1.attention.layer_norm_rms_epsilon", gguf.ValueTypeFloat32, float32(1e-6)),
	}}
	spec, err := ReadSpec(file)
	if err != nil {
		t.Fatal(err)
	}
	if spec.Architecture != "rnd1" || !spec.NonCausalAttention ||
		spec.ExpertCount != 8 || spec.ExpertUsedCount != 2 || spec.ExpertFeedForward != 12 {
		t.Fatalf("unexpected RND1 spec: %+v", spec)
	}
}

func TestReadLLaDASpecsAreNonCausal(t *testing.T) {
	for _, architecture := range []string{"llada", "llada-moe"} {
		t.Run(architecture, func(t *testing.T) {
			metadataItems := []gguf.Metadata{
				metadata("general.architecture", gguf.ValueTypeString, architecture),
				metadata(architecture+".block_count", gguf.ValueTypeUint32, uint32(2)),
				metadata(architecture+".context_length", gguf.ValueTypeUint32, uint32(4096)),
				metadata(architecture+".embedding_length", gguf.ValueTypeUint32, uint32(8)),
				metadata(architecture+".feed_forward_length", gguf.ValueTypeUint32, uint32(12)),
				metadata(architecture+".attention.head_count", gguf.ValueTypeUint32, uint32(2)),
				metadata(architecture+".attention.head_count_kv", gguf.ValueTypeUint32, uint32(1)),
				metadata(architecture+".attention.key_length", gguf.ValueTypeUint32, uint32(4)),
				metadata(architecture+".attention.value_length", gguf.ValueTypeUint32, uint32(4)),
				metadata(architecture+".rope.freq_base", gguf.ValueTypeFloat32, float32(10000)),
				metadata(architecture+".attention.layer_norm_rms_epsilon", gguf.ValueTypeFloat32, float32(1e-6)),
			}
			if architecture == "llada-moe" {
				metadataItems = append(metadataItems,
					metadata("llada-moe.expert_count", gguf.ValueTypeUint32, uint32(4)),
					metadata("llada-moe.expert_used_count", gguf.ValueTypeUint32, uint32(2)),
					metadata("llada-moe.expert_feed_forward_length", gguf.ValueTypeUint32, uint32(6)),
				)
			}
			spec, err := ReadSpec(&gguf.File{Metadata: metadataItems})
			if err != nil {
				t.Fatal(err)
			}
			if !spec.NonCausalAttention || (architecture == "llada") != usesNormalRoPE(architecture) {
				t.Fatalf("unexpected %s spec: %+v", architecture, spec)
			}
			if architecture == "llada-moe" && (spec.ExpertCount != 4 || spec.ExpertFeedForward != 6) {
				t.Fatalf("unexpected LLaDA-MoE experts: %+v", spec)
			}
		})
	}
}

func TestReadLagunaSpecPreservesPerLayerHeadsAndHybridRoPE(t *testing.T) {
	file := &gguf.File{Metadata: []gguf.Metadata{
		metadata("general.architecture", gguf.ValueTypeString, "laguna"),
		metadata("laguna.block_count", gguf.ValueTypeUint32, uint32(2)),
		metadata("laguna.context_length", gguf.ValueTypeUint32, uint32(8192)),
		metadata("laguna.embedding_length", gguf.ValueTypeUint32, uint32(8)),
		metadata("laguna.feed_forward_length", gguf.ValueTypeUint32, uint32(16)),
		{Key: "laguna.attention.head_count", Value: gguf.Value{Type: gguf.ValueTypeArray, ArrayType: gguf.ValueTypeUint32, Data: []uint32{2, 4}}},
		metadata("laguna.attention.head_count_kv", gguf.ValueTypeUint32, uint32(1)),
		metadata("laguna.attention.key_length", gguf.ValueTypeUint32, uint32(4)),
		metadata("laguna.attention.value_length", gguf.ValueTypeUint32, uint32(4)),
		metadata("laguna.attention.layer_norm_rms_epsilon", gguf.ValueTypeFloat32, float32(1e-6)),
		metadata("laguna.leading_dense_block_count", gguf.ValueTypeUint32, uint32(1)),
		metadata("laguna.expert_count", gguf.ValueTypeUint32, uint32(4)),
		metadata("laguna.expert_used_count", gguf.ValueTypeUint32, uint32(2)),
		metadata("laguna.expert_feed_forward_length", gguf.ValueTypeUint32, uint32(12)),
		metadata("laguna.expert_shared_feed_forward_length", gguf.ValueTypeUint32, uint32(10)),
		metadata("laguna.expert_weights_scale", gguf.ValueTypeFloat32, float32(1.25)),
		metadata("laguna.expert_weights_norm", gguf.ValueTypeBool, true),
		metadata("laguna.rope.freq_base", gguf.ValueTypeFloat32, float32(500000)),
		metadata("laguna.rope.dimension_count", gguf.ValueTypeUint32, uint32(4)),
		metadata("laguna.rope.scaling.type", gguf.ValueTypeString, "yarn"),
		metadata("laguna.rope.scaling.factor", gguf.ValueTypeFloat32, float32(4)),
		metadata("laguna.rope.scaling.original_context_length", gguf.ValueTypeUint32, uint32(2048)),
		metadata("laguna.attention.sliding_window", gguf.ValueTypeUint32, uint32(1024)),
		metadata("laguna.attention.sliding_window_pattern", gguf.ValueTypeUint32, uint32(2)),
		metadata("laguna.rope.freq_base_swa", gguf.ValueTypeFloat32, float32(10000)),
		metadata("laguna.rope.dimension_count_swa", gguf.ValueTypeUint32, uint32(4)),
	}}
	spec, err := ReadSpec(file)
	if err != nil {
		t.Fatal(err)
	}
	if spec.LayerHeadCount(0) != 2 || spec.LayerHeadCount(1) != 4 ||
		spec.LayerKVHeadCount(1) != 1 || spec.IsSlidingLayer(0) || !spec.IsSlidingLayer(1) ||
		spec.RopeScalingType != "yarn" || spec.SharedExpertFF != 10 ||
		!spec.ExpertWeightsNorm || spec.ExpertGatingFunc != 2 ||
		math.Abs(float64(spec.YaRNAttentionFactor-1/(1+0.1*float32(math.Log(4))))) > 1e-6 {
		t.Fatalf("unexpected Laguna spec: %+v", spec)
	}
}

func TestReadOpenELMSpecPreservesPerLayerWidths(t *testing.T) {
	array := func(key string, values []uint32) gguf.Metadata {
		return gguf.Metadata{Key: key, Value: gguf.Value{
			Type: gguf.ValueTypeArray, ArrayType: gguf.ValueTypeUint32, Data: values,
		}}
	}
	file := &gguf.File{Metadata: []gguf.Metadata{
		metadata("general.architecture", gguf.ValueTypeString, "openelm"),
		metadata("openelm.block_count", gguf.ValueTypeUint32, uint32(2)),
		metadata("openelm.context_length", gguf.ValueTypeUint32, uint32(4096)),
		metadata("openelm.embedding_length", gguf.ValueTypeUint32, uint32(8)),
		array("openelm.feed_forward_length", []uint32{12, 16}),
		array("openelm.attention.head_count", []uint32{2, 4}),
		array("openelm.attention.head_count_kv", []uint32{1, 2}),
		metadata("openelm.attention.key_length", gguf.ValueTypeUint32, uint32(4)),
		metadata("openelm.attention.value_length", gguf.ValueTypeUint32, uint32(4)),
		metadata("openelm.rope.freq_base", gguf.ValueTypeFloat32, float32(10000)),
		metadata("openelm.attention.layer_norm_rms_epsilon", gguf.ValueTypeFloat32, float32(1e-6)),
	}}
	spec, err := ReadSpec(file)
	if err != nil {
		t.Fatal(err)
	}
	if spec.Architecture != "openelm" || spec.LayerHeadCount(1) != 4 ||
		spec.LayerKVHeadCount(1) != 2 || spec.LayerFeedForwardLength(0) != 12 ||
		spec.LayerFeedForwardLength(1) != 16 {
		t.Fatalf("unexpected OpenELM spec: %+v", spec)
	}
}

func TestReadDeciSpecPreservesSparseLayerSchedule(t *testing.T) {
	array := func(key string, values []uint32) gguf.Metadata {
		return gguf.Metadata{Key: key, Value: gguf.Value{
			Type: gguf.ValueTypeArray, ArrayType: gguf.ValueTypeUint32, Data: values,
		}}
	}
	file := &gguf.File{Metadata: []gguf.Metadata{
		metadata("general.architecture", gguf.ValueTypeString, "deci"),
		metadata("deci.block_count", gguf.ValueTypeUint32, uint32(4)),
		metadata("deci.context_length", gguf.ValueTypeUint32, uint32(4096)),
		metadata("deci.embedding_length", gguf.ValueTypeUint32, uint32(8)),
		array("deci.feed_forward_length", []uint32{12, 12, 12, 0}),
		array("deci.attention.head_count", []uint32{2, 2, 0, 0}),
		array("deci.attention.head_count_kv", []uint32{1, 0, 0, 0}),
		metadata("deci.attention.key_length", gguf.ValueTypeUint32, uint32(4)),
		metadata("deci.attention.value_length", gguf.ValueTypeUint32, uint32(4)),
		metadata("deci.rope.freq_base", gguf.ValueTypeFloat32, float32(500000)),
		metadata("deci.rope.dimension_count", gguf.ValueTypeUint32, uint32(4)),
		metadata("deci.attention.layer_norm_rms_epsilon", gguf.ValueTypeFloat32, float32(1e-5)),
	}}
	spec, err := ReadSpec(file)
	if err != nil {
		t.Fatal(err)
	}
	if spec.Architecture != "deci" || spec.HeadCount != 2 || spec.HeadCountKV != 1 ||
		spec.FeedForwardLength != 12 || spec.LayerHeadCount(2) != 0 ||
		spec.LayerKVHeadCount(1) != 0 || spec.LayerFeedForwardLength(3) != 0 ||
		spec.RopeDimensionCount != 4 || !usesNormalRoPE(spec.Architecture) {
		t.Fatalf("unexpected Deci spec: %+v", spec)
	}
}

func TestReadGrokSpec(t *testing.T) {
	file := &gguf.File{Metadata: []gguf.Metadata{
		metadata("general.architecture", gguf.ValueTypeString, "grok"),
		metadata("grok.block_count", gguf.ValueTypeUint32, uint32(2)),
		metadata("grok.context_length", gguf.ValueTypeUint32, uint32(8192)),
		metadata("grok.embedding_length", gguf.ValueTypeUint32, uint32(8)),
		metadata("grok.feed_forward_length", gguf.ValueTypeUint32, uint32(12)),
		metadata("grok.attention.head_count", gguf.ValueTypeUint32, uint32(2)),
		metadata("grok.attention.head_count_kv", gguf.ValueTypeUint32, uint32(1)),
		metadata("grok.attention.key_length", gguf.ValueTypeUint32, uint32(4)),
		metadata("grok.attention.value_length", gguf.ValueTypeUint32, uint32(4)),
		metadata("grok.attention.layer_norm_rms_epsilon", gguf.ValueTypeFloat32, float32(1e-5)),
		metadata("grok.rope.freq_base", gguf.ValueTypeFloat32, float32(10000)),
		metadata("grok.rope.dimension_count", gguf.ValueTypeUint32, uint32(4)),
		metadata("grok.rope.scaling.type", gguf.ValueTypeString, "yarn"),
		metadata("grok.rope.scaling.factor", gguf.ValueTypeFloat32, float32(4)),
		metadata("grok.rope.scaling.original_context_length", gguf.ValueTypeUint32, uint32(2048)),
		metadata("grok.rope.scaling.yarn_ext_factor", gguf.ValueTypeFloat32, float32(1)),
		metadata("grok.rope.scaling.yarn_attn_factor", gguf.ValueTypeFloat32, float32(1.25)),
		metadata("grok.rope.scaling.yarn_beta_fast", gguf.ValueTypeFloat32, float32(8)),
		metadata("grok.rope.scaling.yarn_beta_slow", gguf.ValueTypeFloat32, float32(1)),
		metadata("grok.expert_count", gguf.ValueTypeUint32, uint32(4)),
		metadata("grok.expert_used_count", gguf.ValueTypeUint32, uint32(2)),
		metadata("grok.expert_feed_forward_length", gguf.ValueTypeUint32, uint32(6)),
		metadata("grok.expert_weights_scale", gguf.ValueTypeFloat32, float32(1.5)),
		metadata("grok.embedding_scale", gguf.ValueTypeFloat32, float32(2)),
		metadata("grok.logit_scale", gguf.ValueTypeFloat32, float32(0.5)),
		metadata("grok.attention.output_scale", gguf.ValueTypeFloat32, float32(0.25)),
		metadata("grok.attn_logit_softcapping", gguf.ValueTypeFloat32, float32(30)),
	}}
	spec, err := ReadSpec(file)
	if err != nil {
		t.Fatal(err)
	}
	if spec.Architecture != "grok" || spec.ExpertFeedForward != 6 ||
		!spec.ExpertWeightsNorm || spec.ExpertWeightsScale != 1.5 ||
		spec.InputEmbeddingScale() != 2 || spec.OutputLogitMultiplier() != 0.5 ||
		spec.AttentionScale != 0.25 || spec.AttentionSoftcap != 30 ||
		spec.RopeScalingType != "yarn" || spec.YaRNAttentionFactor != 1.25 ||
		spec.YaRNBetaFast != 8 {
		t.Fatalf("unexpected Grok spec: %+v", spec)
	}
}

func TestReadGrokSpecUsesLegacyDefaults(t *testing.T) {
	file := &gguf.File{Metadata: []gguf.Metadata{
		metadata("general.architecture", gguf.ValueTypeString, "grok"),
		metadata("grok.block_count", gguf.ValueTypeUint32, uint32(2)),
		metadata("grok.context_length", gguf.ValueTypeUint32, uint32(8192)),
		metadata("grok.embedding_length", gguf.ValueTypeUint32, uint32(8)),
		metadata("grok.feed_forward_length", gguf.ValueTypeUint32, uint32(12)),
		metadata("grok.attention.head_count", gguf.ValueTypeUint32, uint32(2)),
		metadata("grok.attention.head_count_kv", gguf.ValueTypeUint32, uint32(1)),
		metadata("grok.attention.key_length", gguf.ValueTypeUint32, uint32(4)),
		metadata("grok.attention.value_length", gguf.ValueTypeUint32, uint32(4)),
		metadata("grok.attention.layer_norm_rms_epsilon", gguf.ValueTypeFloat32, float32(1e-5)),
		metadata("grok.rope.freq_base", gguf.ValueTypeFloat32, float32(10000)),
		metadata("grok.expert_count", gguf.ValueTypeUint32, uint32(4)),
		metadata("grok.expert_used_count", gguf.ValueTypeUint32, uint32(2)),
	}}
	spec, err := ReadSpec(file)
	if err != nil {
		t.Fatal(err)
	}
	if spec.ExpertFeedForward != 12 || !spec.ExpertWeightsNorm ||
		spec.RopeDimensionCount != 4 || spec.AttentionSoftcap != 30 ||
		spec.AttentionScale != float32(0.08838834764831845) ||
		spec.InputEmbeddingScale() != float32(78.38367176906169) ||
		spec.OutputLogitMultiplier() != float32(0.5773502691896257) {
		t.Fatalf("unexpected Grok legacy defaults: %+v", spec)
	}
}

func TestReadMellumSpec(t *testing.T) {
	file := &gguf.File{Metadata: []gguf.Metadata{
		metadata("general.architecture", gguf.ValueTypeString, "mellum"),
		metadata("mellum.block_count", gguf.ValueTypeUint32, uint32(4)),
		metadata("mellum.context_length", gguf.ValueTypeUint32, uint32(8192)),
		metadata("mellum.embedding_length", gguf.ValueTypeUint32, uint32(8)),
		metadata("mellum.feed_forward_length", gguf.ValueTypeUint32, uint32(12)),
		metadata("mellum.attention.head_count", gguf.ValueTypeUint32, uint32(2)),
		metadata("mellum.attention.head_count_kv", gguf.ValueTypeUint32, uint32(1)),
		metadata("mellum.attention.layer_norm_rms_epsilon", gguf.ValueTypeFloat32, float32(1e-6)),
		metadata("mellum.rope.freq_base", gguf.ValueTypeFloat32, float32(10000)),
		metadata("mellum.rope.scaling.type", gguf.ValueTypeString, "yarn"),
		metadata("mellum.rope.scaling.factor", gguf.ValueTypeFloat32, float32(4)),
		metadata("mellum.rope.scaling.original_context_length", gguf.ValueTypeUint32, uint32(2048)),
		metadata("mellum.attention.sliding_window", gguf.ValueTypeUint32, uint32(128)),
		metadata("mellum.attention.sliding_window_pattern", gguf.ValueTypeUint32, uint32(4)),
		metadata("mellum.rope.freq_base_swa", gguf.ValueTypeFloat32, float32(20000)),
		metadata("mellum.expert_count", gguf.ValueTypeUint32, uint32(4)),
		metadata("mellum.expert_used_count", gguf.ValueTypeUint32, uint32(2)),
		metadata("mellum.expert_feed_forward_length", gguf.ValueTypeUint32, uint32(6)),
	}}
	spec, err := ReadSpec(file)
	if err != nil {
		t.Fatal(err)
	}
	if spec.ExpertFeedForward != 6 || !spec.ExpertWeightsNorm ||
		spec.RopeDimensionCount != 4 || spec.RopeScalingType != "yarn" ||
		spec.RopeFrequencySWA != 20000 || !spec.IsSlidingLayer(0) || spec.IsSlidingLayer(3) {
		t.Fatalf("unexpected Mellum spec: %+v", spec)
	}
}

func TestReadQwenSpec(t *testing.T) {
	file := &gguf.File{Metadata: []gguf.Metadata{
		metadata("general.architecture", gguf.ValueTypeString, "qwen"),
		metadata("qwen.block_count", gguf.ValueTypeUint32, uint32(2)),
		metadata("qwen.context_length", gguf.ValueTypeUint32, uint32(2048)),
		metadata("qwen.embedding_length", gguf.ValueTypeUint32, uint32(8)),
		metadata("qwen.feed_forward_length", gguf.ValueTypeUint32, uint32(24)),
		metadata("qwen.attention.head_count", gguf.ValueTypeUint32, uint32(2)),
		metadata("qwen.attention.layer_norm_rms_epsilon", gguf.ValueTypeFloat32, float32(1e-6)),
		metadata("qwen.rope.freq_base", gguf.ValueTypeFloat32, float32(10000)),
		metadata("qwen.rope.dimension_count", gguf.ValueTypeUint32, uint32(4)),
	}}
	spec, err := ReadSpec(file)
	if err != nil {
		t.Fatal(err)
	}
	if spec.FeedForwardLength != 12 || spec.HeadCountKV != 2 || spec.RopeDimensionCount != 4 {
		t.Fatalf("unexpected Qwen spec: %+v", spec)
	}
}

func TestReadChatGLMSpec(t *testing.T) {
	file := &gguf.File{Metadata: []gguf.Metadata{
		metadata("general.architecture", gguf.ValueTypeString, "chatglm"),
		metadata("chatglm.block_count", gguf.ValueTypeUint32, uint32(2)),
		metadata("chatglm.context_length", gguf.ValueTypeUint32, uint32(2048)),
		metadata("chatglm.embedding_length", gguf.ValueTypeUint32, uint32(8)),
		metadata("chatglm.feed_forward_length", gguf.ValueTypeUint32, uint32(12)),
		metadata("chatglm.attention.head_count", gguf.ValueTypeUint32, uint32(2)),
		metadata("chatglm.attention.head_count_kv", gguf.ValueTypeUint32, uint32(1)),
		metadata("chatglm.attention.layer_norm_rms_epsilon", gguf.ValueTypeFloat32, float32(1e-6)),
		metadata("chatglm.rope.freq_base", gguf.ValueTypeFloat32, float32(10000)),
	}}
	spec, err := ReadSpec(file)
	if err != nil {
		t.Fatal(err)
	}
	if spec.RopeDimensionCount != 4 || !usesNormalRoPE(spec.Architecture) || !usesFusedGateUp(spec.Architecture) {
		t.Fatalf("unexpected ChatGLM spec: %+v", spec)
	}
}

func TestReadHunyuanDenseSpec(t *testing.T) {
	file := &gguf.File{Metadata: []gguf.Metadata{
		metadata("general.architecture", gguf.ValueTypeString, "hunyuan-dense"),
		metadata("hunyuan-dense.block_count", gguf.ValueTypeUint32, uint32(2)),
		metadata("hunyuan-dense.context_length", gguf.ValueTypeUint32, uint32(4096)),
		metadata("hunyuan-dense.embedding_length", gguf.ValueTypeUint32, uint32(8)),
		metadata("hunyuan-dense.feed_forward_length", gguf.ValueTypeUint32, uint32(12)),
		metadata("hunyuan-dense.attention.head_count", gguf.ValueTypeUint32, uint32(2)),
		metadata("hunyuan-dense.attention.head_count_kv", gguf.ValueTypeUint32, uint32(1)),
		metadata("hunyuan-dense.attention.layer_norm_rms_epsilon", gguf.ValueTypeFloat32, float32(1e-6)),
		metadata("hunyuan-dense.rope.freq_base", gguf.ValueTypeFloat32, float32(10000)),
		metadata("hunyuan-dense.rope.scaling.alpha", gguf.ValueTypeFloat32, float32(2)),
	}}
	spec, err := ReadSpec(file)
	if err != nil {
		t.Fatal(err)
	}
	if spec.RopeDimensionCount != 4 || spec.RopeFrequencyBase != 40000 {
		t.Fatalf("unexpected Hunyuan-Dense spec: %+v", spec)
	}
	file.Metadata = append(file.Metadata, gguf.Metadata{
		Key: "hunyuan-dense.rope.dimension_sections",
		Value: gguf.Value{Type: gguf.ValueTypeArray, ArrayType: gguf.ValueTypeInt32,
			Data: []int32{1, 0, 0, 0}},
	})
	if _, err = ReadSpec(file); err == nil || !strings.Contains(err.Error(), "multidimensional") {
		t.Fatalf("Hunyuan-Dense multidimensional RoPE error = %v", err)
	}
	file.Metadata[len(file.Metadata)-1].Value.Data = []int32{0, 0, 0}
	if _, err = ReadSpec(file); err == nil || !strings.Contains(err.Error(), "need 4") {
		t.Fatalf("Hunyuan-Dense RoPE sections error = %v", err)
	}
}

func TestReadAFMoESpec(t *testing.T) {
	file := &gguf.File{Metadata: []gguf.Metadata{
		metadata("general.architecture", gguf.ValueTypeString, "afmoe"),
		metadata("afmoe.block_count", gguf.ValueTypeUint32, uint32(1)),
		metadata("afmoe.context_length", gguf.ValueTypeUint32, uint32(4096)),
		metadata("afmoe.embedding_length", gguf.ValueTypeUint32, uint32(8)),
		metadata("afmoe.feed_forward_length", gguf.ValueTypeUint32, uint32(16)),
		metadata("afmoe.attention.head_count", gguf.ValueTypeUint32, uint32(2)),
		metadata("afmoe.attention.head_count_kv", gguf.ValueTypeUint32, uint32(1)),
		metadata("afmoe.attention.key_length", gguf.ValueTypeUint32, uint32(4)),
		metadata("afmoe.attention.value_length", gguf.ValueTypeUint32, uint32(4)),
		metadata("afmoe.attention.layer_norm_rms_epsilon", gguf.ValueTypeFloat32, float32(1e-6)),
		metadata("afmoe.expert_count", gguf.ValueTypeUint32, uint32(4)),
		metadata("afmoe.expert_used_count", gguf.ValueTypeUint32, uint32(2)),
		metadata("afmoe.expert_feed_forward_length", gguf.ValueTypeUint32, uint32(6)),
		metadata("afmoe.expert_shared_count", gguf.ValueTypeUint32, uint32(2)),
		metadata("afmoe.expert_weights_scale", gguf.ValueTypeFloat32, float32(2.826)),
		metadata("afmoe.expert_weights_norm", gguf.ValueTypeBool, true),
		metadata("afmoe.rope.freq_base", gguf.ValueTypeFloat32, float32(10000)),
		metadata("afmoe.attention.sliding_window", gguf.ValueTypeUint32, uint32(64)),
	}}
	spec, err := ReadSpec(file)
	if err != nil {
		t.Fatal(err)
	}
	if spec.Architecture != "afmoe" || spec.ExpertGatingFunc != 2 ||
		spec.SharedExpertCount != 2 || spec.SharedExpertFF != 12 ||
		!spec.ExpertWeightsNorm || spec.NoRopeLayerStep != 4 ||
		!spec.IsSlidingLayer(0) || spec.InputEmbeddingScale() != float32(math.Sqrt(8)) {
		t.Fatalf("unexpected AFMoE spec: %+v", spec)
	}
}

func TestReadChameleonSandwichSpec(t *testing.T) {
	file := &gguf.File{Metadata: []gguf.Metadata{
		metadata("general.architecture", gguf.ValueTypeString, "chameleon"),
		metadata("chameleon.block_count", gguf.ValueTypeUint32, uint32(2)),
		metadata("chameleon.context_length", gguf.ValueTypeUint32, uint32(4096)),
		metadata("chameleon.embedding_length", gguf.ValueTypeUint32, uint32(8)),
		metadata("chameleon.feed_forward_length", gguf.ValueTypeUint32, uint32(16)),
		metadata("chameleon.attention.head_count", gguf.ValueTypeUint32, uint32(2)),
		metadata("chameleon.attention.head_count_kv", gguf.ValueTypeUint32, uint32(1)),
		metadata("chameleon.rope.freq_base", gguf.ValueTypeFloat32, float32(10000)),
		metadata("chameleon.attention.layer_norm_rms_epsilon", gguf.ValueTypeFloat32, float32(1e-5)),
		metadata("chameleon.swin_norm", gguf.ValueTypeBool, true),
	}}
	spec, err := ReadSpec(file)
	if err != nil {
		t.Fatal(err)
	}
	if spec.Architecture != "chameleon" || !spec.SandwichNorm ||
		spec.QKNormEpsilon != 1e-5 || spec.KeyLength != 4 {
		t.Fatalf("unexpected Chameleon spec: %+v", spec)
	}
}

func TestReadLFM2HybridSpec(t *testing.T) {
	file := &gguf.File{Metadata: []gguf.Metadata{
		metadata("general.architecture", gguf.ValueTypeString, "lfm2"),
		metadata("lfm2.block_count", gguf.ValueTypeUint32, uint32(3)),
		metadata("lfm2.context_length", gguf.ValueTypeUint32, uint32(4096)),
		metadata("lfm2.embedding_length", gguf.ValueTypeUint32, uint32(8)),
		metadata("lfm2.feed_forward_length", gguf.ValueTypeUint32, uint32(16)),
		metadata("lfm2.attention.head_count", gguf.ValueTypeUint32, uint32(2)),
		{Key: "lfm2.attention.head_count_kv", Value: gguf.Value{
			Type: gguf.ValueTypeArray, ArrayType: gguf.ValueTypeUint32,
			Data: []uint32{0, 1, 0},
		}},
		metadata("lfm2.attention.key_length", gguf.ValueTypeUint32, uint32(4)),
		metadata("lfm2.attention.value_length", gguf.ValueTypeUint32, uint32(4)),
		metadata("lfm2.rope.freq_base", gguf.ValueTypeFloat32, float32(10000)),
		metadata("lfm2.attention.layer_norm_rms_epsilon", gguf.ValueTypeFloat32, float32(1e-6)),
		metadata("lfm2.shortconv.l_cache", gguf.ValueTypeUint32, uint32(4)),
	}}
	spec, err := ReadSpec(file)
	if err != nil {
		t.Fatal(err)
	}
	if spec.Architecture != "lfm2" || spec.HeadCountKV != 1 ||
		spec.ShortConvCacheLength != 4 || !spec.IsRecurrentLayer(0) ||
		spec.IsRecurrentLayer(1) || !spec.IsRecurrentLayer(2) {
		t.Fatalf("unexpected LFM2 spec: %+v", spec)
	}
}

func TestReadLFM2MoEHybridSpec(t *testing.T) {
	file := &gguf.File{Metadata: []gguf.Metadata{
		metadata("general.architecture", gguf.ValueTypeString, "lfm2moe"),
		metadata("lfm2moe.block_count", gguf.ValueTypeUint32, uint32(3)),
		metadata("lfm2moe.context_length", gguf.ValueTypeUint32, uint32(4096)),
		metadata("lfm2moe.embedding_length", gguf.ValueTypeUint32, uint32(8)),
		metadata("lfm2moe.feed_forward_length", gguf.ValueTypeUint32, uint32(16)),
		metadata("lfm2moe.expert_feed_forward_length", gguf.ValueTypeUint32, uint32(6)),
		metadata("lfm2moe.expert_count", gguf.ValueTypeUint32, uint32(8)),
		metadata("lfm2moe.expert_used_count", gguf.ValueTypeUint32, uint32(2)),
		metadata("lfm2moe.expert_weights_scale", gguf.ValueTypeFloat32, float32(1.25)),
		metadata("lfm2moe.expert_gating_func", gguf.ValueTypeUint32, uint32(2)),
		metadata("lfm2moe.leading_dense_block_count", gguf.ValueTypeUint32, uint32(1)),
		metadata("lfm2moe.attention.head_count", gguf.ValueTypeUint32, uint32(2)),
		{Key: "lfm2moe.attention.head_count_kv", Value: gguf.Value{
			Type: gguf.ValueTypeArray, ArrayType: gguf.ValueTypeUint32,
			Data: []uint32{0, 1, 0},
		}},
		metadata("lfm2moe.attention.key_length", gguf.ValueTypeUint32, uint32(4)),
		metadata("lfm2moe.attention.value_length", gguf.ValueTypeUint32, uint32(4)),
		metadata("lfm2moe.rope.freq_base", gguf.ValueTypeFloat32, float32(10000)),
		metadata("lfm2moe.attention.layer_norm_rms_epsilon", gguf.ValueTypeFloat32, float32(1e-6)),
		metadata("lfm2moe.shortconv.l_cache", gguf.ValueTypeUint32, uint32(4)),
		metadata("lfm2moe.attention.sliding_window", gguf.ValueTypeUint32, uint32(128)),
	}}
	spec, err := ReadSpec(file)
	if err != nil {
		t.Fatal(err)
	}
	if spec.Architecture != "lfm2moe" || spec.HeadCountKV != 1 ||
		spec.ShortConvCacheLength != 4 || spec.LeadingDenseBlocks != 1 ||
		spec.ExpertCount != 8 || spec.ExpertUsedCount != 2 || spec.ExpertFeedForward != 6 ||
		spec.ExpertWeightsScale != 1.25 || spec.ExpertGatingFunc != 2 ||
		!spec.IsRecurrentLayer(0) || spec.IsRecurrentLayer(1) || !spec.IsRecurrentLayer(2) ||
		!spec.IsSlidingLayer(1) {
		t.Fatalf("unexpected LFM2-MoE spec: %+v", spec)
	}
}

func TestReadPLMMLASpec(t *testing.T) {
	file := &gguf.File{Metadata: []gguf.Metadata{
		metadata("general.architecture", gguf.ValueTypeString, "plm"),
		metadata("plm.block_count", gguf.ValueTypeUint32, uint32(2)),
		metadata("plm.context_length", gguf.ValueTypeUint32, uint32(2048)),
		metadata("plm.embedding_length", gguf.ValueTypeUint32, uint32(8)),
		metadata("plm.feed_forward_length", gguf.ValueTypeUint32, uint32(16)),
		metadata("plm.attention.head_count", gguf.ValueTypeUint32, uint32(2)),
		metadata("plm.attention.head_count_kv", gguf.ValueTypeUint32, uint32(2)),
		metadata("plm.attention.key_length", gguf.ValueTypeUint32, uint32(6)),
		metadata("plm.attention.value_length", gguf.ValueTypeUint32, uint32(4)),
		metadata("plm.attention.kv_lora_rank", gguf.ValueTypeUint32, uint32(3)),
		metadata("plm.rope.dimension_count", gguf.ValueTypeUint32, uint32(2)),
		metadata("plm.rope.freq_base", gguf.ValueTypeFloat32, float32(10000)),
		metadata("plm.attention.layer_norm_rms_epsilon", gguf.ValueTypeFloat32, float32(1e-6)),
	}}
	spec, err := ReadSpec(file)
	if err != nil {
		t.Fatal(err)
	}
	if spec.Architecture != "plm" || spec.KVLoRARank != 3 || spec.RopeDimensionCount != 2 {
		t.Fatalf("unexpected PLM spec: %+v", spec)
	}
}

func TestReadMiniCPM3Spec(t *testing.T) {
	file := &gguf.File{Metadata: []gguf.Metadata{
		metadata("general.architecture", gguf.ValueTypeString, "minicpm3"),
		metadata("minicpm3.block_count", gguf.ValueTypeUint32, uint32(2)),
		metadata("minicpm3.context_length", gguf.ValueTypeUint32, uint32(4096)),
		metadata("minicpm3.embedding_length", gguf.ValueTypeUint32, uint32(8)),
		metadata("minicpm3.feed_forward_length", gguf.ValueTypeUint32, uint32(12)),
		metadata("minicpm3.attention.head_count", gguf.ValueTypeUint32, uint32(2)),
		metadata("minicpm3.attention.head_count_kv", gguf.ValueTypeUint32, uint32(2)),
		metadata("minicpm3.attention.key_length", gguf.ValueTypeUint32, uint32(6)),
		metadata("minicpm3.attention.q_lora_rank", gguf.ValueTypeUint32, uint32(3)),
		metadata("minicpm3.attention.kv_lora_rank", gguf.ValueTypeUint32, uint32(3)),
		metadata("minicpm3.rope.dimension_count", gguf.ValueTypeUint32, uint32(2)),
		metadata("minicpm3.attention.layer_norm_rms_epsilon", gguf.ValueTypeFloat32, float32(1e-6)),
	}}
	spec, err := ReadSpec(file)
	if err != nil {
		t.Fatal(err)
	}
	if spec.QLoRARank != 3 || spec.KVLoRARank != 3 || spec.KeyLength != 6 || spec.ValueLength != 4 ||
		spec.RopeDimensionCount != 2 || spec.RopeFrequencyBase != 10000 ||
		spec.EmbeddingScale != 12 || spec.LogitScale != 32 ||
		math.Abs(float64(spec.ResidualScale-float32(1.4/math.Sqrt(2)))) > 1e-6 {
		t.Fatalf("unexpected MiniCPM3 spec: %+v", spec)
	}
}

func TestReadQwen2Spec(t *testing.T) {
	file := &gguf.File{Metadata: []gguf.Metadata{
		metadata("general.architecture", gguf.ValueTypeString, "qwen2"),
		metadata("qwen2.block_count", gguf.ValueTypeUint32, uint32(28)),
		metadata("qwen2.context_length", gguf.ValueTypeUint32, uint32(32768)),
		metadata("qwen2.embedding_length", gguf.ValueTypeUint32, uint32(3584)),
		metadata("qwen2.feed_forward_length", gguf.ValueTypeUint32, uint32(18944)),
		metadata("qwen2.attention.head_count", gguf.ValueTypeUint32, uint32(28)),
		metadata("qwen2.attention.head_count_kv", gguf.ValueTypeUint32, uint32(4)),
		metadata("qwen2.attention.key_length", gguf.ValueTypeUint32, uint32(128)),
		metadata("qwen2.attention.value_length", gguf.ValueTypeUint32, uint32(128)),
		metadata("qwen2.rope.freq_base", gguf.ValueTypeFloat32, float32(1_000_000)),
		metadata("qwen2.attention.layer_norm_rms_epsilon", gguf.ValueTypeFloat32, float32(1e-6)),
	}}
	spec, err := ReadSpec(file)
	if err != nil {
		t.Fatal(err)
	}
	if spec.Architecture != "qwen2" || spec.BlockCount != 28 || spec.KeyLength != 128 {
		t.Fatalf("unexpected Qwen 2 spec: %+v", spec)
	}
}

func TestReadGPT2AndStarCoderLearnedPositionSpecs(t *testing.T) {
	for _, architecture := range []string{"gpt2", "starcoder"} {
		t.Run(architecture, func(t *testing.T) {
			prefix := architecture + "."
			file := &gguf.File{Metadata: []gguf.Metadata{
				metadata("general.architecture", gguf.ValueTypeString, architecture),
				metadata(prefix+"block_count", gguf.ValueTypeUint32, uint32(2)),
				metadata(prefix+"context_length", gguf.ValueTypeUint32, uint32(128)),
				metadata(prefix+"embedding_length", gguf.ValueTypeUint32, uint32(16)),
				metadata(prefix+"feed_forward_length", gguf.ValueTypeUint32, uint32(64)),
				metadata(prefix+"attention.head_count", gguf.ValueTypeUint32, uint32(4)),
				metadata(prefix+"attention.layer_norm_epsilon", gguf.ValueTypeFloat32, float32(1e-5)),
				metadata(prefix+"vocab_size", gguf.ValueTypeUint32, uint32(32)),
			}}
			spec, err := ReadSpec(file)
			if err != nil {
				t.Fatal(err)
			}
			if spec.Architecture != architecture || spec.HeadCountKV != 4 ||
				spec.KeyLength != 4 || !spec.RopeDisabled || spec.UsesRoPE(0) ||
				!spec.UsesLayerNorm() {
				t.Fatalf("unexpected learned-position spec: %+v", spec)
			}
		})
	}
}

func TestReadBloomALiBiSpec(t *testing.T) {
	file := &gguf.File{Metadata: []gguf.Metadata{
		metadata("general.architecture", gguf.ValueTypeString, "bloom"),
		metadata("bloom.block_count", gguf.ValueTypeUint32, uint32(2)),
		metadata("bloom.context_length", gguf.ValueTypeUint32, uint32(128)),
		metadata("bloom.embedding_length", gguf.ValueTypeUint32, uint32(16)),
		metadata("bloom.feed_forward_length", gguf.ValueTypeUint32, uint32(64)),
		metadata("bloom.attention.head_count", gguf.ValueTypeUint32, uint32(4)),
		metadata("bloom.attention.layer_norm_epsilon", gguf.ValueTypeFloat32, float32(1e-5)),
		metadata("bloom.vocab_size", gguf.ValueTypeUint32, uint32(32)),
	}}
	spec, err := ReadSpec(file)
	if err != nil {
		t.Fatal(err)
	}
	if !spec.RopeDisabled || spec.MaxALiBiBias != 8 || spec.HeadCountKV != 4 ||
		!spec.UsesLayerNorm() {
		t.Fatalf("unexpected Bloom spec: %+v", spec)
	}
}

func TestReadMPTAndRefactALiBiSpecs(t *testing.T) {
	tests := []struct {
		architecture string
		layerNorm    bool
	}{
		{architecture: "mpt", layerNorm: true},
		{architecture: "refact", layerNorm: false},
	}
	for _, test := range tests {
		t.Run(test.architecture, func(t *testing.T) {
			prefix := test.architecture + "."
			metadataItems := []gguf.Metadata{
				metadata("general.architecture", gguf.ValueTypeString, test.architecture),
				metadata(prefix+"block_count", gguf.ValueTypeUint32, uint32(2)),
				metadata(prefix+"context_length", gguf.ValueTypeUint32, uint32(128)),
				metadata(prefix+"embedding_length", gguf.ValueTypeUint32, uint32(16)),
				metadata(prefix+"feed_forward_length", gguf.ValueTypeUint32, uint32(64)),
				metadata(prefix+"attention.head_count", gguf.ValueTypeUint32, uint32(4)),
				metadata(prefix+"attention.head_count_kv", gguf.ValueTypeUint32, uint32(2)),
				metadata(prefix+"attention.max_alibi_bias", gguf.ValueTypeFloat32, float32(8)),
				metadata(prefix+"vocab_size", gguf.ValueTypeUint32, uint32(32)),
			}
			if test.layerNorm {
				metadataItems = append(metadataItems, metadata(
					prefix+"attention.layer_norm_epsilon", gguf.ValueTypeFloat32, float32(1e-5),
				))
			} else {
				metadataItems = append(metadataItems, metadata(
					prefix+"attention.layer_norm_rms_epsilon", gguf.ValueTypeFloat32, float32(1e-5),
				))
			}
			spec, err := ReadSpec(&gguf.File{Metadata: metadataItems})
			if err != nil {
				t.Fatal(err)
			}
			if !spec.RopeDisabled || spec.MaxALiBiBias != 8 ||
				spec.HeadCountKV != 2 || spec.UsesLayerNorm() != test.layerNorm {
				t.Fatalf("unexpected %s spec: %+v", test.architecture, spec)
			}
		})
	}
}

func TestReadInternLM2EXAONEAndXVERSESpecs(t *testing.T) {
	for _, architecture := range []string{"internlm2", "exaone", "xverse"} {
		t.Run(architecture, func(t *testing.T) {
			prefix := architecture + "."
			file := &gguf.File{Metadata: []gguf.Metadata{
				metadata("general.architecture", gguf.ValueTypeString, architecture),
				metadata(prefix+"block_count", gguf.ValueTypeUint32, uint32(2)),
				metadata(prefix+"context_length", gguf.ValueTypeUint32, uint32(4096)),
				metadata(prefix+"embedding_length", gguf.ValueTypeUint32, uint32(128)),
				metadata(prefix+"feed_forward_length", gguf.ValueTypeUint32, uint32(256)),
				metadata(prefix+"attention.head_count", gguf.ValueTypeUint32, uint32(4)),
				metadata(prefix+"attention.head_count_kv", gguf.ValueTypeUint32, uint32(2)),
				metadata(prefix+"rope.freq_base", gguf.ValueTypeFloat32, float32(10000)),
				metadata(
					prefix+"attention.layer_norm_rms_epsilon",
					gguf.ValueTypeFloat32,
					float32(1e-6),
				),
			}}
			spec, err := ReadSpec(file)
			if err != nil {
				t.Fatal(err)
			}
			if spec.Architecture != architecture ||
				spec.KeyLength != 32 ||
				spec.ValueLength != 32 {
				t.Fatalf("unexpected %s spec: %+v", architecture, spec)
			}
		})
	}
}

func TestReadOLMo2SlidingSpec(t *testing.T) {
	file := &gguf.File{Metadata: []gguf.Metadata{
		metadata("general.architecture", gguf.ValueTypeString, "olmo2"),
		metadata("olmo2.block_count", gguf.ValueTypeUint32, uint32(32)),
		metadata("olmo2.context_length", gguf.ValueTypeUint32, uint32(4096)),
		metadata("olmo2.embedding_length", gguf.ValueTypeUint32, uint32(4096)),
		metadata("olmo2.feed_forward_length", gguf.ValueTypeUint32, uint32(11008)),
		metadata("olmo2.attention.head_count", gguf.ValueTypeUint32, uint32(32)),
		metadata("olmo2.attention.head_count_kv", gguf.ValueTypeUint32, uint32(8)),
		metadata("olmo2.rope.freq_base", gguf.ValueTypeFloat32, float32(10000)),
		metadata("olmo2.rope.freq_base_swa", gguf.ValueTypeFloat32, float32(20000)),
		metadata("olmo2.attention.sliding_window", gguf.ValueTypeUint32, uint32(1024)),
		metadata("olmo2.attention.layer_norm_rms_epsilon", gguf.ValueTypeFloat32, float32(1e-6)),
	}}
	spec, err := ReadSpec(file)
	if err != nil {
		t.Fatal(err)
	}
	if spec.Architecture != "olmo2" ||
		spec.SlidingWindow != 1024 ||
		spec.SlidingPattern != 4 ||
		spec.RopeFrequencySWA != 20000 ||
		!spec.IsSlidingLayer(0) ||
		!spec.IsSlidingLayer(2) ||
		spec.IsSlidingLayer(3) {
		t.Fatalf("unexpected OLMo2 spec: %+v", spec)
	}
}

func TestReadCohere2Spec(t *testing.T) {
	file := &gguf.File{Metadata: []gguf.Metadata{
		metadata("general.architecture", gguf.ValueTypeString, "cohere2"),
		metadata("cohere2.block_count", gguf.ValueTypeUint32, uint32(32)),
		metadata("cohere2.context_length", gguf.ValueTypeUint32, uint32(131072)),
		metadata("cohere2.embedding_length", gguf.ValueTypeUint32, uint32(4096)),
		metadata("cohere2.feed_forward_length", gguf.ValueTypeUint32, uint32(14336)),
		metadata("cohere2.vocab_size", gguf.ValueTypeUint32, uint32(256000)),
		metadata("cohere2.attention.head_count", gguf.ValueTypeUint32, uint32(32)),
		metadata("cohere2.attention.head_count_kv", gguf.ValueTypeUint32, uint32(8)),
		metadata("cohere2.rope.freq_base", gguf.ValueTypeFloat32, float32(10000)),
		metadata("cohere2.rope.freq_base_swa", gguf.ValueTypeFloat32, float32(20000)),
		metadata("cohere2.rope.dimension_count", gguf.ValueTypeUint32, uint32(32)),
		metadata("cohere2.attention.sliding_window", gguf.ValueTypeUint32, uint32(4096)),
		metadata("cohere2.attention.sliding_window_pattern", gguf.ValueTypeUint32, uint32(4)),
		metadata("cohere2.attention.layer_norm_epsilon", gguf.ValueTypeFloat32, float32(1e-5)),
		metadata("cohere2.logit_scale", gguf.ValueTypeFloat32, float32(0.125)),
	}}
	spec, err := ReadSpec(file)
	if err != nil {
		t.Fatal(err)
	}
	if spec.Architecture != "cohere2" ||
		spec.VocabularySize != 256000 ||
		spec.RopeDimensionCount != 32 ||
		spec.SlidingWindow != 4096 ||
		spec.SlidingPattern != 4 ||
		spec.NoRopeLayerStep != 4 ||
		!spec.IsSlidingLayer(2) || spec.IsSlidingLayer(3) ||
		!spec.UsesRoPE(2) || spec.UsesRoPE(3) ||
		!spec.UsesWeightOnlyLayerNorm() ||
		spec.OutputLogitMultiplier() != 0.125 {
		t.Fatalf("unexpected Cohere2 spec: %+v", spec)
	}
}

func TestReadCohere2MoESpec(t *testing.T) {
	file := &gguf.File{Metadata: []gguf.Metadata{
		metadata("general.architecture", gguf.ValueTypeString, "cohere2moe"),
		metadata("cohere2moe.block_count", gguf.ValueTypeUint32, uint32(5)),
		metadata("cohere2moe.context_length", gguf.ValueTypeUint32, uint32(4096)),
		metadata("cohere2moe.embedding_length", gguf.ValueTypeUint32, uint32(8)),
		metadata("cohere2moe.feed_forward_length", gguf.ValueTypeUint32, uint32(12)),
		metadata("cohere2moe.vocab_size", gguf.ValueTypeUint32, uint32(32)),
		metadata("cohere2moe.attention.head_count", gguf.ValueTypeUint32, uint32(2)),
		metadata("cohere2moe.attention.head_count_kv", gguf.ValueTypeUint32, uint32(1)),
		metadata("cohere2moe.rope.freq_base", gguf.ValueTypeFloat32, float32(10000)),
		metadata("cohere2moe.rope.freq_base_swa", gguf.ValueTypeFloat32, float32(20000)),
		metadata("cohere2moe.rope.dimension_count", gguf.ValueTypeUint32, uint32(4)),
		metadata("cohere2moe.attention.sliding_window", gguf.ValueTypeUint32, uint32(128)),
		metadata("cohere2moe.attention.layer_norm_rms_epsilon", gguf.ValueTypeFloat32, float32(1e-5)),
		metadata("cohere2moe.logit_scale", gguf.ValueTypeFloat32, float32(0.5)),
		metadata("cohere2moe.nextn_predict_layers", gguf.ValueTypeUint32, uint32(1)),
		metadata("cohere2moe.leading_dense_block_count", gguf.ValueTypeUint32, uint32(1)),
		metadata("cohere2moe.expert_count", gguf.ValueTypeUint32, uint32(4)),
		metadata("cohere2moe.expert_used_count", gguf.ValueTypeUint32, uint32(2)),
		metadata("cohere2moe.expert_feed_forward_length", gguf.ValueTypeUint32, uint32(6)),
		metadata("cohere2moe.expert_shared_count", gguf.ValueTypeUint32, uint32(1)),
		metadata("cohere2moe.expert_weights_norm", gguf.ValueTypeBool, true),
		metadata("cohere2moe.expert_weights_scale", gguf.ValueTypeFloat32, float32(1.25)),
		{
			Key: "cohere2moe.attention.sliding_window_pattern",
			Value: gguf.Value{Type: gguf.ValueTypeArray, ArrayType: gguf.ValueTypeBool,
				Data: []bool{false, true, false, true}},
		},
	}}
	spec, err := ReadSpec(file)
	if err != nil {
		t.Fatal(err)
	}
	if spec.Architecture != "cohere2moe" || spec.BlockCount != 4 || spec.RMSNormEpsilon != 1e-5 ||
		spec.UsesWeightOnlyLayerNorm() || spec.LeadingDenseBlocks != 1 ||
		spec.ExpertFeedForward != 6 || spec.SharedExpertFF != 6 ||
		spec.ExpertGatingFunc != 2 || !spec.ExpertWeightsNorm || spec.ExpertWeightsScale != 1.25 ||
		spec.IsSlidingLayer(0) || !spec.IsSlidingLayer(1) ||
		!spec.UsesRoPE(0) || !spec.UsesRoPE(1) || spec.UsesRoPE(2) ||
		spec.OutputLogitMultiplier() != 0.5 {
		t.Fatalf("unexpected Cohere2-MoE spec: %+v", spec)
	}
}

func TestReadHYV3SpecTrimsNextNLayers(t *testing.T) {
	file := &gguf.File{Metadata: []gguf.Metadata{
		metadata("general.architecture", gguf.ValueTypeString, "hy_v3"),
		metadata("hy_v3.block_count", gguf.ValueTypeUint32, uint32(5)),
		metadata("hy_v3.context_length", gguf.ValueTypeUint32, uint32(4096)),
		metadata("hy_v3.embedding_length", gguf.ValueTypeUint32, uint32(8)),
		metadata("hy_v3.feed_forward_length", gguf.ValueTypeUint32, uint32(12)),
		metadata("hy_v3.vocab_size", gguf.ValueTypeUint32, uint32(32)),
		metadata("hy_v3.attention.head_count", gguf.ValueTypeUint32, uint32(2)),
		metadata("hy_v3.attention.head_count_kv", gguf.ValueTypeUint32, uint32(1)),
		metadata("hy_v3.rope.freq_base", gguf.ValueTypeFloat32, float32(10000)),
		metadata("hy_v3.attention.layer_norm_rms_epsilon", gguf.ValueTypeFloat32, float32(1e-5)),
		metadata("hy_v3.nextn_predict_layers", gguf.ValueTypeUint32, uint32(1)),
		metadata("hy_v3.expert_count", gguf.ValueTypeUint32, uint32(4)),
		metadata("hy_v3.expert_used_count", gguf.ValueTypeUint32, uint32(2)),
		metadata("hy_v3.expert_feed_forward_length", gguf.ValueTypeUint32, uint32(6)),
		metadata("hy_v3.expert_weights_norm", gguf.ValueTypeBool, true),
		metadata("hy_v3.expert_weights_scale", gguf.ValueTypeFloat32, float32(1.25)),
	}}
	spec, err := ReadSpec(file)
	if err != nil {
		t.Fatal(err)
	}
	if spec.Architecture != "hy_v3" || spec.BlockCount != 4 || spec.RMSNormEpsilon != 1e-5 ||
		spec.ExpertFeedForward != 6 || spec.SharedExpertFF != 6 || spec.ExpertGatingFunc != 2 ||
		!spec.ExpertWeightsNorm || spec.ExpertWeightsScale != 1.25 ||
		spec.RopeDimensionCount != 4 || usesNormalRoPE(spec.Architecture) {
		t.Fatalf("unexpected HY-V3 spec: %+v", spec)
	}
}

func TestReadDeepSeek2OCRSpec(t *testing.T) {
	file := &gguf.File{Metadata: []gguf.Metadata{
		metadata("general.architecture", gguf.ValueTypeString, "deepseek2-ocr"),
		metadata("deepseek2-ocr.block_count", gguf.ValueTypeUint32, uint32(2)),
		metadata("deepseek2-ocr.context_length", gguf.ValueTypeUint32, uint32(4096)),
		metadata("deepseek2-ocr.embedding_length", gguf.ValueTypeUint32, uint32(8)),
		metadata("deepseek2-ocr.feed_forward_length", gguf.ValueTypeUint32, uint32(12)),
		metadata("deepseek2-ocr.vocab_size", gguf.ValueTypeUint32, uint32(32)),
		metadata("deepseek2-ocr.attention.head_count", gguf.ValueTypeUint32, uint32(2)),
		metadata("deepseek2-ocr.attention.head_count_kv", gguf.ValueTypeUint32, uint32(2)),
		metadata("deepseek2-ocr.attention.layer_norm_rms_epsilon", gguf.ValueTypeFloat32, float32(1e-6)),
		metadata("deepseek2-ocr.leading_dense_block_count", gguf.ValueTypeUint32, uint32(1)),
		metadata("deepseek2-ocr.expert_count", gguf.ValueTypeUint32, uint32(4)),
		metadata("deepseek2-ocr.expert_used_count", gguf.ValueTypeUint32, uint32(2)),
		metadata("deepseek2-ocr.expert_feed_forward_length", gguf.ValueTypeUint32, uint32(6)),
		metadata("deepseek2-ocr.expert_shared_count", gguf.ValueTypeUint32, uint32(2)),
	}}
	spec, err := ReadSpec(file)
	if err != nil {
		t.Fatal(err)
	}
	if spec.Architecture != "deepseek2-ocr" || spec.LeadingDenseBlocks != 1 ||
		spec.ExpertFeedForward != 6 || spec.SharedExpertFF != 12 ||
		spec.ExpertGatingFunc != 1 || spec.RopeDimensionCount != 4 ||
		spec.RopeFrequencyBase != 10000 || usesNormalRoPE(spec.Architecture) {
		t.Fatalf("unexpected DeepSeek2-OCR spec: %+v", spec)
	}
	spec.RopeScalingType = "linear"
	spec.RopeScalingFactor = 2
	if err := spec.validate(); err == nil {
		t.Fatal("expected DeepSeek2-OCR RoPE scaling rejection")
	}
}

func TestReadCommandRSpec(t *testing.T) {
	file := &gguf.File{Metadata: []gguf.Metadata{
		metadata("general.architecture", gguf.ValueTypeString, "command-r"),
		metadata("command-r.block_count", gguf.ValueTypeUint32, uint32(40)),
		metadata("command-r.context_length", gguf.ValueTypeUint32, uint32(131072)),
		metadata("command-r.embedding_length", gguf.ValueTypeUint32, uint32(8192)),
		metadata("command-r.feed_forward_length", gguf.ValueTypeUint32, uint32(22528)),
		metadata("command-r.attention.head_count", gguf.ValueTypeUint32, uint32(64)),
		metadata("command-r.attention.head_count_kv", gguf.ValueTypeUint32, uint32(8)),
		metadata("command-r.rope.freq_base", gguf.ValueTypeFloat32, float32(8000000)),
		metadata("command-r.attention.layer_norm_epsilon", gguf.ValueTypeFloat32, float32(1e-5)),
		metadata("command-r.logit_scale", gguf.ValueTypeFloat32, float32(0.0625)),
	}}
	spec, err := ReadSpec(file)
	if err != nil {
		t.Fatal(err)
	}
	if spec.Architecture != "command-r" || spec.KeyLength != 128 ||
		!spec.UsesWeightOnlyLayerNorm() || !spec.UsesRoPE(39) ||
		spec.OutputLogitMultiplier() != 0.0625 {
		t.Fatalf("unexpected Command R spec: %+v", spec)
	}
}

func TestReadPLaMoSpec(t *testing.T) {
	file := &gguf.File{Metadata: []gguf.Metadata{
		metadata("general.architecture", gguf.ValueTypeString, "plamo"),
		metadata("plamo.block_count", gguf.ValueTypeUint32, uint32(40)),
		metadata("plamo.context_length", gguf.ValueTypeUint32, uint32(4096)),
		metadata("plamo.embedding_length", gguf.ValueTypeUint32, uint32(5120)),
		metadata("plamo.feed_forward_length", gguf.ValueTypeUint32, uint32(13824)),
		metadata("plamo.attention.head_count", gguf.ValueTypeUint32, uint32(40)),
		metadata("plamo.attention.head_count_kv", gguf.ValueTypeUint32, uint32(5)),
		metadata("plamo.rope.freq_base", gguf.ValueTypeFloat32, float32(10000)),
		metadata("plamo.attention.layer_norm_rms_epsilon", gguf.ValueTypeFloat32, float32(1e-5)),
	}}
	spec, err := ReadSpec(file)
	if err != nil {
		t.Fatal(err)
	}
	if spec.Architecture != "plamo" || spec.KeyLength != 128 ||
		!usesParallelResidual(spec.Architecture) {
		t.Fatalf("unexpected PLaMo spec: %+v", spec)
	}
}

func TestReadPLaMo3Spec(t *testing.T) {
	file := &gguf.File{Metadata: []gguf.Metadata{
		metadata("general.architecture", gguf.ValueTypeString, "plamo3"),
		metadata("plamo3.block_count", gguf.ValueTypeUint32, uint32(2)),
		metadata("plamo3.context_length", gguf.ValueTypeUint32, uint32(4096)),
		metadata("plamo3.embedding_length", gguf.ValueTypeUint32, uint32(8)),
		{Key: "plamo3.feed_forward_length", Value: gguf.Value{
			Type: gguf.ValueTypeArray, ArrayType: gguf.ValueTypeUint32, Data: []uint32{12, 16},
		}},
		{Key: "plamo3.attention.head_count", Value: gguf.Value{
			Type: gguf.ValueTypeArray, ArrayType: gguf.ValueTypeUint32, Data: []uint32{2, 4},
		}},
		{Key: "plamo3.attention.head_count_kv", Value: gguf.Value{
			Type: gguf.ValueTypeArray, ArrayType: gguf.ValueTypeUint32, Data: []uint32{1, 2},
		}},
		metadata("plamo3.attention.key_length", gguf.ValueTypeUint32, uint32(2)),
		metadata("plamo3.attention.value_length", gguf.ValueTypeUint32, uint32(2)),
		metadata("plamo3.rope.freq_base", gguf.ValueTypeFloat32, float32(10000)),
		metadata("plamo3.rope.freq_base_swa", gguf.ValueTypeFloat32, float32(20000)),
		metadata("plamo3.attention.sliding_window", gguf.ValueTypeUint32, uint32(128)),
		metadata("plamo3.attention.sliding_window_pattern", gguf.ValueTypeUint32, uint32(2)),
		metadata("plamo3.attention.layer_norm_rms_epsilon", gguf.ValueTypeFloat32, float32(1e-6)),
	}}
	spec, err := ReadSpec(file)
	if err != nil {
		t.Fatal(err)
	}
	if spec.LayerHeadCount(0) != 2 || spec.LayerHeadCount(1) != 4 ||
		spec.LayerKVHeadCount(0) != 1 || spec.LayerKVHeadCount(1) != 2 ||
		spec.LayerFeedForwardLength(0) != 12 || spec.LayerFeedForwardLength(1) != 16 ||
		!spec.IsSlidingLayer(0) || spec.IsSlidingLayer(1) ||
		spec.RopeFrequencySWA != 20000 || usesNormalRoPE(spec.Architecture) ||
		!hasPostNorm(spec.Architecture) || !usesFusedGateUp(spec.Architecture) {
		t.Fatalf("unexpected PLaMo 3 spec: %+v", spec)
	}
}

func TestReadJaisSpec(t *testing.T) {
	file := &gguf.File{Metadata: []gguf.Metadata{
		metadata("general.architecture", gguf.ValueTypeString, "jais"),
		metadata("jais.block_count", gguf.ValueTypeUint32, uint32(2)),
		metadata("jais.context_length", gguf.ValueTypeUint32, uint32(4096)),
		metadata("jais.embedding_length", gguf.ValueTypeUint32, uint32(8)),
		metadata("jais.feed_forward_length", gguf.ValueTypeUint32, uint32(12)),
		metadata("jais.attention.head_count", gguf.ValueTypeUint32, uint32(2)),
		metadata("jais.attention.max_alibi_bias", gguf.ValueTypeFloat32, float32(8)),
		metadata("jais.attention.layer_norm_epsilon", gguf.ValueTypeFloat32, float32(1e-5)),
	}}
	spec, err := ReadSpec(file)
	if err != nil {
		t.Fatal(err)
	}
	if spec.Architecture != "jais" || spec.HeadCountKV != 2 || spec.KeyLength != 4 ||
		spec.AttentionScale != 0.25 || spec.MaxALiBiBias != 8 || spec.UsesRoPE(0) ||
		!spec.UsesLayerNorm() || !spec.RequiresLayerNormBias() {
		t.Fatalf("unexpected Jais spec: %+v", spec)
	}
}

func TestReadStableLMSpec(t *testing.T) {
	file := &gguf.File{Metadata: []gguf.Metadata{
		metadata("general.architecture", gguf.ValueTypeString, "stablelm"),
		metadata("stablelm.block_count", gguf.ValueTypeUint32, uint32(40)),
		metadata("stablelm.context_length", gguf.ValueTypeUint32, uint32(4096)),
		metadata("stablelm.embedding_length", gguf.ValueTypeUint32, uint32(5120)),
		metadata("stablelm.feed_forward_length", gguf.ValueTypeUint32, uint32(13824)),
		metadata("stablelm.attention.head_count", gguf.ValueTypeUint32, uint32(40)),
		metadata("stablelm.attention.head_count_kv", gguf.ValueTypeUint32, uint32(5)),
		metadata("stablelm.rope.freq_base", gguf.ValueTypeFloat32, float32(10000)),
		metadata("stablelm.rope.dimension_count", gguf.ValueTypeUint32, uint32(32)),
		metadata("stablelm.attention.layer_norm_epsilon", gguf.ValueTypeFloat32, float32(1e-5)),
	}}
	spec, err := ReadSpec(file)
	if err != nil {
		t.Fatal(err)
	}
	if spec.Architecture != "stablelm" || spec.KeyLength != 128 ||
		spec.RopeDimensionCount != 32 || !spec.UsesLayerNorm() {
		t.Fatalf("unexpected StableLM spec: %+v", spec)
	}
}

func TestReadPhi2Spec(t *testing.T) {
	file := &gguf.File{Metadata: []gguf.Metadata{
		metadata("general.architecture", gguf.ValueTypeString, "phi2"),
		metadata("phi2.block_count", gguf.ValueTypeUint32, uint32(32)),
		metadata("phi2.context_length", gguf.ValueTypeUint32, uint32(2048)),
		metadata("phi2.embedding_length", gguf.ValueTypeUint32, uint32(2560)),
		metadata("phi2.feed_forward_length", gguf.ValueTypeUint32, uint32(10240)),
		metadata("phi2.attention.head_count", gguf.ValueTypeUint32, uint32(32)),
		metadata("phi2.attention.head_count_kv", gguf.ValueTypeUint32, uint32(32)),
		metadata("phi2.rope.freq_base", gguf.ValueTypeFloat32, float32(10000)),
		metadata("phi2.rope.dimension_count", gguf.ValueTypeUint32, uint32(32)),
		metadata("phi2.attention.layer_norm_epsilon", gguf.ValueTypeFloat32, float32(1e-5)),
	}}
	spec, err := ReadSpec(file)
	if err != nil {
		t.Fatal(err)
	}
	if spec.Architecture != "phi2" || spec.KeyLength != 80 ||
		spec.RopeDimensionCount != 32 || !spec.UsesLayerNorm() ||
		!usesParallelResidual(spec.Architecture) {
		t.Fatalf("unexpected Phi-2 spec: %+v", spec)
	}
}

func TestReadPhi3LongRoPESpec(t *testing.T) {
	file := &gguf.File{Metadata: []gguf.Metadata{
		metadata("general.architecture", gguf.ValueTypeString, "phi3"),
		metadata("phi3.block_count", gguf.ValueTypeUint32, uint32(32)),
		metadata("phi3.context_length", gguf.ValueTypeUint32, uint32(131072)),
		metadata("phi3.embedding_length", gguf.ValueTypeUint32, uint32(3072)),
		metadata("phi3.feed_forward_length", gguf.ValueTypeUint32, uint32(8192)),
		metadata("phi3.attention.head_count", gguf.ValueTypeUint32, uint32(32)),
		metadata("phi3.attention.head_count_kv", gguf.ValueTypeUint32, uint32(32)),
		metadata("phi3.rope.freq_base", gguf.ValueTypeFloat32, float32(10000)),
		metadata("phi3.rope.dimension_count", gguf.ValueTypeUint32, uint32(96)),
		metadata("phi3.rope.scaling.type", gguf.ValueTypeString, "longrope"),
		metadata("phi3.rope.scaling.original_context_length", gguf.ValueTypeUint32, uint32(4096)),
		metadata("phi3.rope.scaling.attn_factor", gguf.ValueTypeFloat32, float32(1.19)),
		metadata("phi3.attention.layer_norm_rms_epsilon", gguf.ValueTypeFloat32, float32(1e-5)),
	}}
	spec, err := ReadSpec(file)
	if err != nil {
		t.Fatal(err)
	}
	if spec.Architecture != "phi3" || spec.KeyLength != 96 ||
		spec.RopeScalingType != "longrope" || spec.OriginalContextLength != 4096 ||
		spec.RopeAttentionFactor != 1.19 || spec.RMSNormEpsilon != 1e-5 {
		t.Fatalf("unexpected Phi-3 spec: %+v", spec)
	}
}

func TestReadApertusSpecWithLayerXIELU(t *testing.T) {
	floatArray := func(key string, values ...float32) gguf.Metadata {
		return gguf.Metadata{Key: key, Value: gguf.Value{
			Type: gguf.ValueTypeArray, ArrayType: gguf.ValueTypeFloat32, Data: values,
		}}
	}
	file := &gguf.File{Metadata: []gguf.Metadata{
		metadata("general.architecture", gguf.ValueTypeString, "apertus"),
		metadata("apertus.block_count", gguf.ValueTypeUint32, uint32(2)),
		metadata("apertus.context_length", gguf.ValueTypeUint32, uint32(32768)),
		metadata("apertus.embedding_length", gguf.ValueTypeUint32, uint32(8)),
		metadata("apertus.feed_forward_length", gguf.ValueTypeUint32, uint32(16)),
		metadata("apertus.attention.head_count", gguf.ValueTypeUint32, uint32(2)),
		metadata("apertus.attention.head_count_kv", gguf.ValueTypeUint32, uint32(1)),
		metadata("apertus.rope.freq_base", gguf.ValueTypeFloat32, float32(10000)),
		metadata("apertus.rope.scaling.type", gguf.ValueTypeString, "longrope"),
		metadata("apertus.rope.scaling.original_context_length", gguf.ValueTypeUint32, uint32(8192)),
		metadata("apertus.rope.scaling.attn_factor", gguf.ValueTypeFloat32, float32(1.2)),
		metadata("apertus.attention.layer_norm_rms_epsilon", gguf.ValueTypeFloat32, float32(1e-5)),
		metadata("apertus.xielu.alpha_n", gguf.ValueTypeFloat32, float32(0.8)),
		floatArray("apertus.xielu.alpha_p", 0.1, 0.2),
		floatArray("apertus.xielu.beta", 0.5, 0.6),
		metadata("apertus.xielu.eps", gguf.ValueTypeFloat32, float32(-1e-6)),
	}}
	spec, err := ReadSpec(file)
	if err != nil {
		t.Fatal(err)
	}
	if spec.Architecture != "apertus" || spec.RopeDimensionCount != 4 ||
		spec.OriginalContextLength != 8192 || spec.RopeAttentionFactor != 1.2 ||
		len(spec.XIELUAlphaN) != 2 || spec.XIELUAlphaN[1] != 0.8 ||
		spec.XIELUAlphaP[1] != 0.2 || spec.XIELUBeta[0] != 0.5 ||
		spec.XIELUEpsilon[1] != -1e-6 {
		t.Fatalf("unexpected Apertus spec: %+v", spec)
	}
}

func TestReadGPTNeoXSpec(t *testing.T) {
	file := &gguf.File{Metadata: []gguf.Metadata{
		metadata("general.architecture", gguf.ValueTypeString, "gptneox"),
		metadata("gptneox.block_count", gguf.ValueTypeUint32, uint32(24)),
		metadata("gptneox.context_length", gguf.ValueTypeUint32, uint32(2048)),
		metadata("gptneox.embedding_length", gguf.ValueTypeUint32, uint32(1024)),
		metadata("gptneox.feed_forward_length", gguf.ValueTypeUint32, uint32(4096)),
		metadata("gptneox.attention.head_count", gguf.ValueTypeUint32, uint32(16)),
		metadata("gptneox.rope.dimension_count", gguf.ValueTypeUint32, uint32(16)),
		metadata("gptneox.attention.layer_norm_epsilon", gguf.ValueTypeFloat32, float32(1e-5)),
		metadata("gptneox.use_parallel_residual", gguf.ValueTypeBool, true),
	}}
	spec, err := ReadSpec(file)
	if err != nil {
		t.Fatal(err)
	}
	if spec.Architecture != "gptneox" || spec.HeadCountKV != 16 ||
		spec.KeyLength != 64 || spec.RopeDimensionCount != 16 ||
		spec.RopeFrequencyBase != 10000 || !spec.ParallelResidual ||
		!spec.UsesLayerNorm() {
		t.Fatalf("unexpected GPT-NeoX spec: %+v", spec)
	}
}

func TestReadTextGLM4Spec(t *testing.T) {
	file := &gguf.File{Metadata: []gguf.Metadata{
		metadata("general.architecture", gguf.ValueTypeString, "glm4"),
		metadata("glm4.block_count", gguf.ValueTypeUint32, uint32(40)),
		metadata("glm4.context_length", gguf.ValueTypeUint32, uint32(131072)),
		metadata("glm4.embedding_length", gguf.ValueTypeUint32, uint32(4096)),
		metadata("glm4.feed_forward_length", gguf.ValueTypeUint32, uint32(13696)),
		metadata("glm4.attention.head_count", gguf.ValueTypeUint32, uint32(32)),
		metadata("glm4.attention.head_count_kv", gguf.ValueTypeUint32, uint32(2)),
		metadata("glm4.rope.freq_base", gguf.ValueTypeFloat32, float32(10000)),
		metadata("glm4.rope.dimension_count", gguf.ValueTypeUint32, uint32(64)),
		metadata("glm4.attention.layer_norm_rms_epsilon", gguf.ValueTypeFloat32, float32(1e-5)),
	}}
	spec, err := ReadSpec(file)
	if err != nil {
		t.Fatal(err)
	}
	if spec.Architecture != "glm4" || spec.KeyLength != 128 ||
		spec.RopeDimensionCount != 64 || !usesNormalRoPE(spec.Architecture) ||
		!hasPostNorm(spec.Architecture) || !usesFusedGateUp(spec.Architecture) {
		t.Fatalf("unexpected GLM4 spec: %+v", spec)
	}

	file.Metadata = append(file.Metadata, metadata("glm4.nextn_predict_layers", gguf.ValueTypeUint32, uint32(1)))
	if _, err := ReadSpec(file); err == nil || !strings.Contains(err.Error(), "NextN/MTP") {
		t.Fatalf("GLM4 NextN error = %v", err)
	}
	file.Metadata[len(file.Metadata)-1] = gguf.Metadata{
		Key: "glm4.rope.dimension_sections",
		Value: gguf.Value{
			Type: gguf.ValueTypeArray, ArrayType: gguf.ValueTypeInt32,
			Data: []int32{16, 24, 24, 0},
		},
	}
	if _, err := ReadSpec(file); err == nil || !strings.Contains(err.Error(), "multimodal RoPE") {
		t.Fatalf("GLM4 multimodal error = %v", err)
	}
}

func TestReadEXAONE4Spec(t *testing.T) {
	file := &gguf.File{Metadata: []gguf.Metadata{
		metadata("general.architecture", gguf.ValueTypeString, "exaone4"),
		metadata("exaone4.block_count", gguf.ValueTypeUint32, uint32(64)),
		metadata("exaone4.context_length", gguf.ValueTypeUint32, uint32(32768)),
		metadata("exaone4.embedding_length", gguf.ValueTypeUint32, uint32(8)),
		metadata("exaone4.feed_forward_length", gguf.ValueTypeUint32, uint32(16)),
		metadata("exaone4.attention.head_count", gguf.ValueTypeUint32, uint32(2)),
		metadata("exaone4.attention.head_count_kv", gguf.ValueTypeUint32, uint32(1)),
		metadata("exaone4.rope.freq_base", gguf.ValueTypeFloat32, float32(1_000_000)),
		metadata("exaone4.attention.layer_norm_rms_epsilon", gguf.ValueTypeFloat32, float32(1e-5)),
	}}
	spec, err := ReadSpec(file)
	if err != nil {
		t.Fatal(err)
	}
	if spec.Architecture != "exaone4" || spec.RopeDimensionCount != 4 ||
		spec.SlidingWindow != 4096 || spec.SlidingPattern != 4 ||
		spec.NoRopeLayerStep != 4 || !spec.IsSlidingLayer(0) ||
		spec.IsSlidingLayer(3) || !spec.UsesRoPE(0) || spec.UsesRoPE(3) ||
		!usesPostOnlyNorm(spec.Architecture) {
		t.Fatalf("unexpected EXAONE 4 spec: %+v", spec)
	}
	file.Metadata = append(file.Metadata, metadata("exaone4.nextn_predict_layers", gguf.ValueTypeUint32, uint32(1)))
	if _, err := ReadSpec(file); err == nil || !strings.Contains(err.Error(), "NextN/MTP") {
		t.Fatalf("EXAONE 4 NextN error = %v", err)
	}
}

func TestReadFalconSpec(t *testing.T) {
	file := &gguf.File{Metadata: []gguf.Metadata{
		metadata("general.architecture", gguf.ValueTypeString, "falcon"),
		metadata("falcon.block_count", gguf.ValueTypeUint32, uint32(60)),
		metadata("falcon.context_length", gguf.ValueTypeUint32, uint32(2048)),
		metadata("falcon.embedding_length", gguf.ValueTypeUint32, uint32(8192)),
		metadata("falcon.feed_forward_length", gguf.ValueTypeUint32, uint32(32768)),
		metadata("falcon.attention.head_count", gguf.ValueTypeUint32, uint32(128)),
		metadata("falcon.rope.dimension_count", gguf.ValueTypeUint32, uint32(64)),
		metadata("falcon.attention.layer_norm_epsilon", gguf.ValueTypeFloat32, float32(1e-5)),
	}}
	spec, err := ReadSpec(file)
	if err != nil {
		t.Fatal(err)
	}
	if spec.Architecture != "falcon" || spec.HeadCountKV != 128 ||
		spec.KeyLength != 64 || spec.RopeDimensionCount != 64 ||
		spec.RopeFrequencyBase != 10000 || !spec.UsesLayerNorm() ||
		!usesParallelResidual(spec.Architecture) {
		t.Fatalf("unexpected Falcon spec: %+v", spec)
	}
}

func TestReadBitNetSpec(t *testing.T) {
	file := &gguf.File{Metadata: []gguf.Metadata{
		metadata("general.architecture", gguf.ValueTypeString, "bitnet"),
		metadata("bitnet.block_count", gguf.ValueTypeUint32, uint32(26)),
		metadata("bitnet.context_length", gguf.ValueTypeUint32, uint32(4096)),
		metadata("bitnet.embedding_length", gguf.ValueTypeUint32, uint32(2560)),
		metadata("bitnet.feed_forward_length", gguf.ValueTypeUint32, uint32(6912)),
		metadata("bitnet.attention.head_count", gguf.ValueTypeUint32, uint32(32)),
		metadata("bitnet.attention.head_count_kv", gguf.ValueTypeUint32, uint32(8)),
		metadata("bitnet.rope.freq_base", gguf.ValueTypeFloat32, float32(10000)),
		metadata("bitnet.attention.layer_norm_rms_epsilon", gguf.ValueTypeFloat32, float32(1e-5)),
	}}
	spec, err := ReadSpec(file)
	if err != nil {
		t.Fatal(err)
	}
	if spec.Architecture != "bitnet" || spec.KeyLength != 80 ||
		spec.HeadCountKV != 8 || spec.RMSNormEpsilon != 1e-5 {
		t.Fatalf("unexpected BitNet spec: %+v", spec)
	}
}

func TestReadSmolLM3Spec(t *testing.T) {
	file := &gguf.File{Metadata: []gguf.Metadata{
		metadata("general.architecture", gguf.ValueTypeString, "smollm3"),
		metadata("smollm3.block_count", gguf.ValueTypeUint32, uint32(36)),
		metadata("smollm3.context_length", gguf.ValueTypeUint32, uint32(65536)),
		metadata("smollm3.embedding_length", gguf.ValueTypeUint32, uint32(2048)),
		metadata("smollm3.feed_forward_length", gguf.ValueTypeUint32, uint32(8192)),
		metadata("smollm3.attention.head_count", gguf.ValueTypeUint32, uint32(16)),
		metadata("smollm3.attention.head_count_kv", gguf.ValueTypeUint32, uint32(4)),
		metadata("smollm3.rope.freq_base", gguf.ValueTypeFloat32, float32(1_000_000)),
		metadata("smollm3.attention.scale", gguf.ValueTypeFloat32, float32(0.125)),
		metadata("smollm3.attention.layer_norm_rms_epsilon", gguf.ValueTypeFloat32, float32(1e-6)),
	}}
	spec, err := ReadSpec(file)
	if err != nil {
		t.Fatal(err)
	}
	if spec.Architecture != "smollm3" ||
		spec.NoRopeLayerStep != 4 ||
		spec.AttentionScale != 0.125 ||
		!spec.UsesRoPE(2) ||
		spec.UsesRoPE(3) {
		t.Fatalf("unexpected SmolLM3 spec: %+v", spec)
	}
}

func TestReadMiniCPMSpecScales(t *testing.T) {
	file := &gguf.File{Metadata: []gguf.Metadata{
		metadata("general.architecture", gguf.ValueTypeString, "minicpm"),
		metadata("minicpm.block_count", gguf.ValueTypeUint32, uint32(40)),
		metadata("minicpm.context_length", gguf.ValueTypeUint32, uint32(4096)),
		metadata("minicpm.embedding_length", gguf.ValueTypeUint32, uint32(2304)),
		metadata("minicpm.feed_forward_length", gguf.ValueTypeUint32, uint32(5760)),
		metadata("minicpm.attention.head_count", gguf.ValueTypeUint32, uint32(36)),
		metadata("minicpm.attention.head_count_kv", gguf.ValueTypeUint32, uint32(36)),
		metadata("minicpm.rope.freq_base", gguf.ValueTypeFloat32, float32(10000)),
		metadata("minicpm.attention.layer_norm_rms_epsilon", gguf.ValueTypeFloat32, float32(1e-5)),
		metadata("minicpm.embedding_scale", gguf.ValueTypeFloat32, float32(10)),
		metadata("minicpm.residual_scale", gguf.ValueTypeFloat32, float32(0.2)),
		metadata("minicpm.logit_scale", gguf.ValueTypeFloat32, float32(0.5)),
	}}
	spec, err := ReadSpec(file)
	if err != nil {
		t.Fatal(err)
	}
	if spec.Architecture != "minicpm" ||
		spec.EmbeddingScale != 10 ||
		spec.ResidualScale != 0.2 ||
		spec.LogitScale != 0.5 ||
		spec.InputEmbeddingScale() != 10 ||
		spec.OutputLogitMultiplier() != 2 {
		t.Fatalf("unexpected MiniCPM spec: %+v", spec)
	}
}

func TestReadMiniCPMSpecUsesBackwardCompatibleScaleDefaults(t *testing.T) {
	file := &gguf.File{Metadata: []gguf.Metadata{
		metadata("general.architecture", gguf.ValueTypeString, "minicpm"),
		metadata("minicpm.block_count", gguf.ValueTypeUint32, uint32(40)),
		metadata("minicpm.context_length", gguf.ValueTypeUint32, uint32(4096)),
		metadata("minicpm.embedding_length", gguf.ValueTypeUint32, uint32(2304)),
		metadata("minicpm.feed_forward_length", gguf.ValueTypeUint32, uint32(5760)),
		metadata("minicpm.attention.head_count", gguf.ValueTypeUint32, uint32(36)),
		metadata("minicpm.attention.head_count_kv", gguf.ValueTypeUint32, uint32(36)),
		metadata("minicpm.rope.freq_base", gguf.ValueTypeFloat32, float32(10000)),
		metadata("minicpm.attention.layer_norm_rms_epsilon", gguf.ValueTypeFloat32, float32(1e-5)),
	}}
	spec, err := ReadSpec(file)
	if err != nil {
		t.Fatal(err)
	}
	if spec.EmbeddingScale != 12 ||
		math.Abs(float64(spec.ResidualScale)-1.4/math.Sqrt(40)) > 1e-7 ||
		math.Abs(float64(spec.LogitScale)-256.0/2304.0) > 1e-7 {
		t.Fatalf("unexpected MiniCPM default scales: %+v", spec)
	}
}

func TestReadGraniteDenseSpec(t *testing.T) {
	file := &gguf.File{Metadata: append(graniteMetadata(),
		metadata("granite.embedding_scale", gguf.ValueTypeFloat32, float32(2)),
		metadata("granite.residual_scale", gguf.ValueTypeFloat32, float32(0.5)),
		metadata("granite.attention.scale", gguf.ValueTypeFloat32, float32(0.25)),
		metadata("granite.rope.scaling.finetuned", gguf.ValueTypeBool, false),
	)}
	spec, err := ReadSpec(file)
	if err != nil {
		t.Fatal(err)
	}
	if spec.Architecture != "granite" ||
		spec.EmbeddingScale != 2 ||
		spec.ResidualScale != 0.5 ||
		spec.AttentionScale != 0.25 ||
		spec.LogitScale != 8 ||
		spec.UsesRoPE(0) ||
		spec.OutputLogitMultiplier() != 0.125 {
		t.Fatalf("unexpected Granite spec: %+v", spec)
	}
}

func TestReadGraniteLongRoPESpec(t *testing.T) {
	file := &gguf.File{Metadata: append(graniteMetadata(),
		metadata("granite.rope.scaling.type", gguf.ValueTypeString, "longrope"),
		metadata("granite.rope.dimension_count", gguf.ValueTypeUint32, uint32(128)),
		metadata("granite.rope.scaling.original_context_length", gguf.ValueTypeUint32, uint32(2048)),
	)}
	spec, err := ReadSpec(file)
	if err != nil {
		t.Fatal(err)
	}
	if spec.RopeScalingType != "longrope" || spec.RopeDimensionCount != 128 ||
		spec.OriginalContextLength != 2048 || !supportsLongRoPE(spec.Architecture) {
		t.Fatalf("unexpected Granite LongRoPE spec: %+v", spec)
	}
}

func TestReadGraniteMoESpec(t *testing.T) {
	file := &gguf.File{Metadata: []gguf.Metadata{
		metadata("general.architecture", gguf.ValueTypeString, "granitemoe"),
		metadata("granitemoe.block_count", gguf.ValueTypeUint32, uint32(2)),
		metadata("granitemoe.context_length", gguf.ValueTypeUint32, uint32(4096)),
		metadata("granitemoe.embedding_length", gguf.ValueTypeUint32, uint32(8)),
		metadata("granitemoe.feed_forward_length", gguf.ValueTypeUint32, uint32(6)),
		metadata("granitemoe.expert_count", gguf.ValueTypeUint32, uint32(8)),
		metadata("granitemoe.expert_used_count", gguf.ValueTypeUint32, uint32(2)),
		metadata("granitemoe.expert_shared_feed_forward_length", gguf.ValueTypeUint32, uint32(5)),
		metadata("granitemoe.attention.head_count", gguf.ValueTypeUint32, uint32(2)),
		metadata("granitemoe.attention.head_count_kv", gguf.ValueTypeUint32, uint32(1)),
		metadata("granitemoe.attention.key_length", gguf.ValueTypeUint32, uint32(4)),
		metadata("granitemoe.attention.value_length", gguf.ValueTypeUint32, uint32(4)),
		metadata("granitemoe.rope.freq_base", gguf.ValueTypeFloat32, float32(10000)),
		metadata("granitemoe.attention.layer_norm_rms_epsilon", gguf.ValueTypeFloat32, float32(1e-6)),
		metadata("granitemoe.logit_scale", gguf.ValueTypeFloat32, float32(8)),
		metadata("granitemoe.embedding_scale", gguf.ValueTypeFloat32, float32(2)),
		metadata("granitemoe.residual_scale", gguf.ValueTypeFloat32, float32(0.5)),
	}}
	spec, err := ReadSpec(file)
	if err != nil {
		t.Fatal(err)
	}
	if spec.Architecture != "granitemoe" || spec.ExpertCount != 8 ||
		spec.ExpertUsedCount != 2 || spec.ExpertFeedForward != 6 ||
		spec.SharedExpertFF != 5 || !spec.ExpertWeightsNorm ||
		spec.ExpertWeightsScale != 1 || spec.EmbeddingScale != 2 ||
		spec.ResidualScale != 0.5 || spec.OutputLogitMultiplier() != 0.125 ||
		!usesNormalRoPE(spec.Architecture) {
		t.Fatalf("unexpected GraniteMoE spec: %+v", spec)
	}
}

func TestReadSmallThinkerSpec(t *testing.T) {
	file := &gguf.File{Metadata: []gguf.Metadata{
		metadata("general.architecture", gguf.ValueTypeString, "smallthinker"),
		metadata("smallthinker.block_count", gguf.ValueTypeUint32, uint32(8)),
		metadata("smallthinker.context_length", gguf.ValueTypeUint32, uint32(4096)),
		metadata("smallthinker.embedding_length", gguf.ValueTypeUint32, uint32(8)),
		metadata("smallthinker.feed_forward_length", gguf.ValueTypeUint32, uint32(6)),
		metadata("smallthinker.expert_count", gguf.ValueTypeUint32, uint32(8)),
		metadata("smallthinker.expert_used_count", gguf.ValueTypeUint32, uint32(2)),
		metadata("smallthinker.expert_gating_func", gguf.ValueTypeUint32, uint32(2)),
		metadata("smallthinker.attention.head_count", gguf.ValueTypeUint32, uint32(2)),
		metadata("smallthinker.attention.head_count_kv", gguf.ValueTypeUint32, uint32(1)),
		metadata("smallthinker.attention.key_length", gguf.ValueTypeUint32, uint32(4)),
		metadata("smallthinker.attention.value_length", gguf.ValueTypeUint32, uint32(4)),
		metadata("smallthinker.attention.sliding_window", gguf.ValueTypeUint32, uint32(1024)),
		metadata("smallthinker.attention.sliding_window_pattern", gguf.ValueTypeUint32, uint32(4)),
		metadata("smallthinker.rope.freq_base", gguf.ValueTypeFloat32, float32(10000)),
		metadata("smallthinker.rope.freq_base_swa", gguf.ValueTypeFloat32, float32(20000)),
		metadata("smallthinker.attention.layer_norm_rms_epsilon", gguf.ValueTypeFloat32, float32(1e-6)),
	}}
	spec, err := ReadSpec(file)
	if err != nil {
		t.Fatal(err)
	}
	if spec.ExpertFeedForward != 6 || !spec.ExpertWeightsNorm || spec.ExpertGatingFunc != 2 ||
		spec.SlidingWindow != 4096 || spec.SlidingPattern != 4 || spec.NoRopeLayerStep != 4 ||
		spec.RopeFrequencySWA != 20000 || spec.IsSlidingLayer(0) || !spec.IsSlidingLayer(1) ||
		spec.IsSlidingLayer(4) || spec.UsesRoPE(0) || !spec.UsesRoPE(1) || spec.UsesRoPE(4) {
		t.Fatalf("unexpected SmallThinker spec: %+v", spec)
	}
}

func TestReadSmallThinkerWithoutSlidingUsesRoPEEverywhere(t *testing.T) {
	file := &gguf.File{Metadata: []gguf.Metadata{
		metadata("general.architecture", gguf.ValueTypeString, "smallthinker"),
		metadata("smallthinker.block_count", gguf.ValueTypeUint32, uint32(2)),
		metadata("smallthinker.context_length", gguf.ValueTypeUint32, uint32(4096)),
		metadata("smallthinker.embedding_length", gguf.ValueTypeUint32, uint32(8)),
		metadata("smallthinker.feed_forward_length", gguf.ValueTypeUint32, uint32(6)),
		metadata("smallthinker.expert_count", gguf.ValueTypeUint32, uint32(8)),
		metadata("smallthinker.expert_used_count", gguf.ValueTypeUint32, uint32(2)),
		metadata("smallthinker.expert_gating_func", gguf.ValueTypeUint32, uint32(1)),
		metadata("smallthinker.attention.head_count", gguf.ValueTypeUint32, uint32(2)),
		metadata("smallthinker.attention.head_count_kv", gguf.ValueTypeUint32, uint32(1)),
		metadata("smallthinker.attention.key_length", gguf.ValueTypeUint32, uint32(4)),
		metadata("smallthinker.attention.value_length", gguf.ValueTypeUint32, uint32(4)),
		metadata("smallthinker.rope.freq_base", gguf.ValueTypeFloat32, float32(10000)),
		metadata("smallthinker.attention.layer_norm_rms_epsilon", gguf.ValueTypeFloat32, float32(1e-6)),
	}}
	spec, err := ReadSpec(file)
	if err != nil {
		t.Fatal(err)
	}
	if spec.SlidingWindow != 0 || spec.NoRopeLayerStep != 2 ||
		!spec.UsesRoPE(0) || !spec.UsesRoPE(1) {
		t.Fatalf("unexpected non-sliding SmallThinker spec: %+v", spec)
	}
}

func TestReadDOTS1Spec(t *testing.T) {
	file := &gguf.File{Metadata: []gguf.Metadata{
		metadata("general.architecture", gguf.ValueTypeString, "dots1"),
		metadata("dots1.block_count", gguf.ValueTypeUint32, uint32(4)),
		metadata("dots1.context_length", gguf.ValueTypeUint32, uint32(4096)),
		metadata("dots1.embedding_length", gguf.ValueTypeUint32, uint32(8)),
		metadata("dots1.feed_forward_length", gguf.ValueTypeUint32, uint32(12)),
		metadata("dots1.expert_count", gguf.ValueTypeUint32, uint32(8)),
		metadata("dots1.expert_used_count", gguf.ValueTypeUint32, uint32(2)),
		metadata("dots1.expert_feed_forward_length", gguf.ValueTypeUint32, uint32(6)),
		metadata("dots1.expert_shared_count", gguf.ValueTypeUint32, uint32(2)),
		metadata("dots1.expert_weights_scale", gguf.ValueTypeFloat32, float32(1.25)),
		metadata("dots1.expert_weights_norm", gguf.ValueTypeBool, true),
		metadata("dots1.expert_gating_func", gguf.ValueTypeUint32, uint32(2)),
		metadata("dots1.leading_dense_block_count", gguf.ValueTypeUint32, uint32(1)),
		metadata("dots1.attention.head_count", gguf.ValueTypeUint32, uint32(2)),
		metadata("dots1.attention.head_count_kv", gguf.ValueTypeUint32, uint32(2)),
		metadata("dots1.attention.key_length", gguf.ValueTypeUint32, uint32(4)),
		metadata("dots1.attention.value_length", gguf.ValueTypeUint32, uint32(4)),
		metadata("dots1.rope.freq_base", gguf.ValueTypeFloat32, float32(10000)),
		metadata("dots1.attention.layer_norm_rms_epsilon", gguf.ValueTypeFloat32, float32(1e-6)),
	}}
	spec, err := ReadSpec(file)
	if err != nil {
		t.Fatal(err)
	}
	if spec.ExpertFeedForward != 6 || spec.SharedExpertCount != 2 ||
		spec.SharedExpertFF != 12 || spec.ExpertWeightsScale != 1.25 ||
		!spec.ExpertWeightsNorm || spec.ExpertGatingFunc != 2 || spec.LeadingDenseBlocks != 1 {
		t.Fatalf("unexpected DOTS1 spec: %+v", spec)
	}
}

func TestReadMiniMaxM2Spec(t *testing.T) {
	file := &gguf.File{Metadata: []gguf.Metadata{
		metadata("general.architecture", gguf.ValueTypeString, "minimax-m2"),
		metadata("minimax-m2.block_count", gguf.ValueTypeUint32, uint32(2)),
		metadata("minimax-m2.context_length", gguf.ValueTypeUint32, uint32(4096)),
		metadata("minimax-m2.embedding_length", gguf.ValueTypeUint32, uint32(8)),
		metadata("minimax-m2.feed_forward_length", gguf.ValueTypeUint32, uint32(6)),
		metadata("minimax-m2.expert_count", gguf.ValueTypeUint32, uint32(4)),
		metadata("minimax-m2.expert_used_count", gguf.ValueTypeUint32, uint32(2)),
		metadata("minimax-m2.expert_feed_forward_length", gguf.ValueTypeUint32, uint32(6)),
		metadata("minimax-m2.expert_gating_func", gguf.ValueTypeUint32, uint32(2)),
		metadata("minimax-m2.attention.head_count", gguf.ValueTypeUint32, uint32(2)),
		metadata("minimax-m2.attention.head_count_kv", gguf.ValueTypeUint32, uint32(1)),
		metadata("minimax-m2.attention.key_length", gguf.ValueTypeUint32, uint32(4)),
		metadata("minimax-m2.attention.value_length", gguf.ValueTypeUint32, uint32(4)),
		metadata("minimax-m2.rope.dimension_count", gguf.ValueTypeUint32, uint32(2)),
		metadata("minimax-m2.rope.freq_base", gguf.ValueTypeFloat32, float32(10000)),
		metadata("minimax-m2.attention.layer_norm_rms_epsilon", gguf.ValueTypeFloat32, float32(1e-6)),
	}}
	spec, err := ReadSpec(file)
	if err != nil {
		t.Fatal(err)
	}
	if spec.ExpertFeedForward != 6 || !spec.ExpertWeightsNorm ||
		spec.ExpertGatingFunc != 2 || spec.RopeDimensionCount != 2 {
		t.Fatalf("unexpected MiniMax-M2 spec: %+v", spec)
	}
}

func TestReadGraniteRejectsNonDenseVariants(t *testing.T) {
	for _, extra := range []gguf.Metadata{
		metadata("granite.expert_count", gguf.ValueTypeUint32, uint32(8)),
		{
			Key: "granite.deepstack_mapping",
			Value: gguf.Value{
				Type:      gguf.ValueTypeArray,
				ArrayType: gguf.ValueTypeInt32,
				Data:      []int32{0, -1},
			},
		},
	} {
		file := &gguf.File{Metadata: append(graniteMetadata(), extra)}
		if _, err := ReadSpec(file); err == nil {
			t.Fatalf("Granite variant metadata %q was accepted", extra.Key)
		}
	}
}

func TestReadMaincoderSpec(t *testing.T) {
	file := &gguf.File{Metadata: []gguf.Metadata{
		metadata("general.architecture", gguf.ValueTypeString, "maincoder"),
		metadata("maincoder.block_count", gguf.ValueTypeUint32, uint32(32)),
		metadata("maincoder.context_length", gguf.ValueTypeUint32, uint32(32768)),
		metadata("maincoder.embedding_length", gguf.ValueTypeUint32, uint32(2048)),
		metadata("maincoder.feed_forward_length", gguf.ValueTypeUint32, uint32(8192)),
		metadata("maincoder.attention.head_count", gguf.ValueTypeUint32, uint32(16)),
		metadata("maincoder.attention.head_count_kv", gguf.ValueTypeUint32, uint32(4)),
		metadata("maincoder.rope.freq_base", gguf.ValueTypeFloat32, float32(1_000_000)),
		metadata("maincoder.attention.layer_norm_rms_epsilon", gguf.ValueTypeFloat32, float32(1e-6)),
	}}
	spec, err := ReadSpec(file)
	if err != nil {
		t.Fatal(err)
	}
	if spec.Architecture != "maincoder" ||
		spec.KeyLength != 128 ||
		spec.ValueLength != 128 ||
		!spec.UsesRoPE(0) {
		t.Fatalf("unexpected Maincoder spec: %+v", spec)
	}
}

func TestReadDenseMistral3Spec(t *testing.T) {
	file := &gguf.File{Metadata: mistral3Metadata()}
	spec, err := ReadSpec(file)
	if err != nil {
		t.Fatal(err)
	}
	if spec.Architecture != "mistral3" ||
		spec.KeyLength != 128 ||
		spec.ValueLength != 128 ||
		!spec.UsesRoPE(0) {
		t.Fatalf("unexpected Mistral 3 spec: %+v", spec)
	}
}

func TestReadOrionSpec(t *testing.T) {
	file := &gguf.File{Metadata: []gguf.Metadata{
		metadata("general.architecture", gguf.ValueTypeString, "orion"),
		metadata("orion.block_count", gguf.ValueTypeUint32, uint32(40)),
		metadata("orion.context_length", gguf.ValueTypeUint32, uint32(4096)),
		metadata("orion.embedding_length", gguf.ValueTypeUint32, uint32(5120)),
		metadata("orion.feed_forward_length", gguf.ValueTypeUint32, uint32(13696)),
		metadata("orion.attention.head_count", gguf.ValueTypeUint32, uint32(40)),
		metadata("orion.attention.head_count_kv", gguf.ValueTypeUint32, uint32(40)),
		metadata("orion.rope.freq_base", gguf.ValueTypeFloat32, float32(10000)),
		metadata("orion.attention.layer_norm_epsilon", gguf.ValueTypeFloat32, float32(1e-5)),
	}}
	spec, err := ReadSpec(file)
	if err != nil {
		t.Fatal(err)
	}
	if spec.Architecture != "orion" ||
		!spec.UsesLayerNorm() ||
		spec.LayerNormEpsilon != 1e-5 ||
		spec.RMSNormEpsilon != 0 ||
		spec.KeyLength != 128 ||
		!spec.UsesRoPE(0) {
		t.Fatalf("unexpected Orion spec: %+v", spec)
	}
}

func TestReadStarCoder2Spec(t *testing.T) {
	file := &gguf.File{Metadata: []gguf.Metadata{
		metadata("general.architecture", gguf.ValueTypeString, "starcoder2"),
		metadata("starcoder2.block_count", gguf.ValueTypeUint32, uint32(30)),
		metadata("starcoder2.context_length", gguf.ValueTypeUint32, uint32(16384)),
		metadata("starcoder2.embedding_length", gguf.ValueTypeUint32, uint32(3072)),
		metadata("starcoder2.feed_forward_length", gguf.ValueTypeUint32, uint32(12288)),
		metadata("starcoder2.attention.head_count", gguf.ValueTypeUint32, uint32(24)),
		metadata("starcoder2.attention.head_count_kv", gguf.ValueTypeUint32, uint32(2)),
		metadata("starcoder2.rope.freq_base", gguf.ValueTypeFloat32, float32(100000)),
		metadata("starcoder2.attention.layer_norm_epsilon", gguf.ValueTypeFloat32, float32(1e-5)),
	}}
	spec, err := ReadSpec(file)
	if err != nil {
		t.Fatal(err)
	}
	if spec.Architecture != "starcoder2" ||
		!spec.UsesLayerNorm() ||
		spec.LayerNormEpsilon != 1e-5 ||
		spec.KeyLength != 128 {
		t.Fatalf("unexpected StarCoder2 spec: %+v", spec)
	}
}

func TestReadCodeShellSpec(t *testing.T) {
	file := &gguf.File{Metadata: []gguf.Metadata{
		metadata("general.architecture", gguf.ValueTypeString, "codeshell"),
		metadata("codeshell.block_count", gguf.ValueTypeUint32, uint32(42)),
		metadata("codeshell.context_length", gguf.ValueTypeUint32, uint32(8192)),
		metadata("codeshell.embedding_length", gguf.ValueTypeUint32, uint32(4096)),
		metadata("codeshell.feed_forward_length", gguf.ValueTypeUint32, uint32(16384)),
		metadata("codeshell.attention.head_count", gguf.ValueTypeUint32, uint32(32)),
		metadata("codeshell.attention.head_count_kv", gguf.ValueTypeUint32, uint32(8)),
		metadata("codeshell.rope.freq_base", gguf.ValueTypeFloat32, float32(10000)),
		metadata("codeshell.attention.layer_norm_epsilon", gguf.ValueTypeFloat32, float32(1e-5)),
	}}
	spec, err := ReadSpec(file)
	if err != nil {
		t.Fatal(err)
	}
	if spec.Architecture != "codeshell" ||
		!spec.UsesLayerNorm() ||
		spec.LayerNormEpsilon != 1e-5 ||
		spec.KeyLength != 128 {
		t.Fatalf("unexpected CodeShell spec: %+v", spec)
	}
}

func TestReadBaichuan7BSpec(t *testing.T) {
	metadataValues := []gguf.Metadata{
		metadata("general.architecture", gguf.ValueTypeString, "baichuan"),
		metadata("baichuan.block_count", gguf.ValueTypeUint32, uint32(32)),
		metadata("baichuan.context_length", gguf.ValueTypeUint32, uint32(4096)),
		metadata("baichuan.embedding_length", gguf.ValueTypeUint32, uint32(4096)),
		metadata("baichuan.feed_forward_length", gguf.ValueTypeUint32, uint32(11008)),
		metadata("baichuan.attention.head_count", gguf.ValueTypeUint32, uint32(32)),
		metadata("baichuan.attention.head_count_kv", gguf.ValueTypeUint32, uint32(32)),
		metadata("baichuan.rope.freq_base", gguf.ValueTypeFloat32, float32(10000)),
		metadata("baichuan.attention.layer_norm_rms_epsilon", gguf.ValueTypeFloat32, float32(1e-6)),
	}
	spec, err := ReadSpec(&gguf.File{Metadata: metadataValues})
	if err != nil {
		t.Fatal(err)
	}
	if spec.Architecture != "baichuan" ||
		spec.BlockCount != 32 ||
		spec.KeyLength != 128 ||
		!spec.UsesRoPE(0) {
		t.Fatalf("unexpected Baichuan spec: %+v", spec)
	}

	metadataValues[1] = metadata("baichuan.block_count", gguf.ValueTypeUint32, uint32(40))
	if _, err := ReadSpec(&gguf.File{Metadata: metadataValues}); err == nil ||
		!strings.Contains(err.Error(), "32-layer Baichuan") {
		t.Fatalf("Baichuan 13B error = %v, want explicit unsupported variant", err)
	}
}

func TestReadArceeSpec(t *testing.T) {
	file := &gguf.File{Metadata: []gguf.Metadata{
		metadata("general.architecture", gguf.ValueTypeString, "arcee"),
		metadata("arcee.block_count", gguf.ValueTypeUint32, uint32(36)),
		metadata("arcee.context_length", gguf.ValueTypeUint32, uint32(32768)),
		metadata("arcee.embedding_length", gguf.ValueTypeUint32, uint32(3072)),
		metadata("arcee.feed_forward_length", gguf.ValueTypeUint32, uint32(8192)),
		metadata("arcee.attention.head_count", gguf.ValueTypeUint32, uint32(24)),
		metadata("arcee.attention.head_count_kv", gguf.ValueTypeUint32, uint32(8)),
		metadata("arcee.rope.freq_base", gguf.ValueTypeFloat32, float32(10000)),
		metadata("arcee.attention.layer_norm_rms_epsilon", gguf.ValueTypeFloat32, float32(1e-5)),
		metadata("arcee.attention.scale", gguf.ValueTypeFloat32, float32(0.125)),
	}}
	spec, err := ReadSpec(file)
	if err != nil {
		t.Fatal(err)
	}
	if spec.Architecture != "arcee" ||
		spec.AttentionScale != 0.125 ||
		spec.KeyLength != 128 {
		t.Fatalf("unexpected Arcee spec: %+v", spec)
	}
}

func TestReadNemotronSpec(t *testing.T) {
	file := &gguf.File{Metadata: []gguf.Metadata{
		metadata("general.architecture", gguf.ValueTypeString, "nemotron"),
		metadata("nemotron.block_count", gguf.ValueTypeUint32, uint32(32)),
		metadata("nemotron.context_length", gguf.ValueTypeUint32, uint32(4096)),
		metadata("nemotron.embedding_length", gguf.ValueTypeUint32, uint32(4096)),
		metadata("nemotron.feed_forward_length", gguf.ValueTypeUint32, uint32(11008)),
		metadata("nemotron.attention.head_count", gguf.ValueTypeUint32, uint32(32)),
		metadata("nemotron.attention.head_count_kv", gguf.ValueTypeUint32, uint32(8)),
		metadata("nemotron.rope.freq_base", gguf.ValueTypeFloat32, float32(10000)),
		metadata("nemotron.attention.layer_norm_epsilon", gguf.ValueTypeFloat32, float32(1e-5)),
	}}
	spec, err := ReadSpec(file)
	if err != nil {
		t.Fatal(err)
	}
	if spec.Architecture != "nemotron" ||
		!spec.UsesLayerNorm() ||
		spec.LayerNormEpsilon != 1e-5 ||
		spec.KeyLength != 128 {
		t.Fatalf("unexpected Nemotron spec: %+v", spec)
	}
}

func TestReadJais2Spec(t *testing.T) {
	file := &gguf.File{Metadata: []gguf.Metadata{
		metadata("general.architecture", gguf.ValueTypeString, "jais2"),
		metadata("jais2.block_count", gguf.ValueTypeUint32, uint32(32)),
		metadata("jais2.context_length", gguf.ValueTypeUint32, uint32(32768)),
		metadata("jais2.embedding_length", gguf.ValueTypeUint32, uint32(4096)),
		metadata("jais2.feed_forward_length", gguf.ValueTypeUint32, uint32(14336)),
		metadata("jais2.attention.head_count", gguf.ValueTypeUint32, uint32(32)),
		metadata("jais2.attention.head_count_kv", gguf.ValueTypeUint32, uint32(32)),
		metadata("jais2.rope.freq_base", gguf.ValueTypeFloat32, float32(10000)),
		metadata("jais2.attention.layer_norm_epsilon", gguf.ValueTypeFloat32, float32(1e-5)),
	}}
	spec, err := ReadSpec(file)
	if err != nil {
		t.Fatal(err)
	}
	if spec.Architecture != "jais2" || !spec.UsesLayerNorm() || spec.KeyLength != 128 {
		t.Fatalf("unexpected Jais2 spec: %+v", spec)
	}
}

func TestReadOLMoSpec(t *testing.T) {
	metadataValues := []gguf.Metadata{
		metadata("general.architecture", gguf.ValueTypeString, "olmo"),
		metadata("olmo.block_count", gguf.ValueTypeUint32, uint32(32)),
		metadata("olmo.context_length", gguf.ValueTypeUint32, uint32(4096)),
		metadata("olmo.embedding_length", gguf.ValueTypeUint32, uint32(4096)),
		metadata("olmo.feed_forward_length", gguf.ValueTypeUint32, uint32(11008)),
		metadata("olmo.attention.head_count", gguf.ValueTypeUint32, uint32(32)),
		metadata("olmo.attention.head_count_kv", gguf.ValueTypeUint32, uint32(8)),
		metadata("olmo.rope.freq_base", gguf.ValueTypeFloat32, float32(10000)),
		metadata("olmo.attention.layer_norm_epsilon", gguf.ValueTypeFloat32, float32(1e-5)),
	}
	spec, err := ReadSpec(&gguf.File{Metadata: metadataValues})
	if err != nil {
		t.Fatal(err)
	}
	if spec.Architecture != "olmo" || !spec.UsesUnweightedLayerNorm() || spec.KeyLength != 128 {
		t.Fatalf("unexpected OLMo spec: %+v", spec)
	}
	metadataValues = append(metadataValues,
		metadata("olmo.attention.clamp_kqv", gguf.ValueTypeFloat32, float32(8)),
	)
	if _, err := ReadSpec(&gguf.File{Metadata: metadataValues}); err == nil ||
		!strings.Contains(err.Error(), "clamping") {
		t.Fatalf("OLMo clamp error = %v, want explicit rejection", err)
	}
}

func TestReadSeedOSSSpec(t *testing.T) {
	file := &gguf.File{Metadata: []gguf.Metadata{
		metadata("general.architecture", gguf.ValueTypeString, "seed_oss"),
		metadata("seed_oss.block_count", gguf.ValueTypeUint32, uint32(64)),
		metadata("seed_oss.context_length", gguf.ValueTypeUint32, uint32(32768)),
		metadata("seed_oss.embedding_length", gguf.ValueTypeUint32, uint32(5120)),
		metadata("seed_oss.feed_forward_length", gguf.ValueTypeUint32, uint32(13824)),
		metadata("seed_oss.attention.head_count", gguf.ValueTypeUint32, uint32(40)),
		metadata("seed_oss.attention.head_count_kv", gguf.ValueTypeUint32, uint32(8)),
		metadata("seed_oss.rope.freq_base", gguf.ValueTypeFloat32, float32(10000)),
		metadata("seed_oss.attention.layer_norm_rms_epsilon", gguf.ValueTypeFloat32, float32(1e-5)),
		metadata("seed_oss.attention.scale", gguf.ValueTypeFloat32, float32(0.125)),
	}}
	spec, err := ReadSpec(file)
	if err != nil {
		t.Fatal(err)
	}
	if spec.Architecture != "seed_oss" || spec.AttentionScale != 0.125 || spec.KeyLength != 128 {
		t.Fatalf("unexpected Seed-OSS spec: %+v", spec)
	}
}

func TestReadMistral3RejectsUnsupportedVariants(t *testing.T) {
	for _, extra := range []gguf.Metadata{
		metadata("mistral3.expert_count", gguf.ValueTypeUint32, uint32(8)),
		metadata("mistral3.attention.temperature_scale", gguf.ValueTypeFloat32, float32(0.1)),
	} {
		file := &gguf.File{Metadata: append(mistral3Metadata(), extra)}
		if _, err := ReadSpec(file); err == nil {
			t.Fatalf("Mistral 3 variant metadata %q was accepted", extra.Key)
		}
	}
}

func TestReadSpecDerivesHeadLength(t *testing.T) {
	file := &gguf.File{Metadata: []gguf.Metadata{
		metadata("general.architecture", gguf.ValueTypeString, "llama"),
		metadata("llama.block_count", gguf.ValueTypeUint32, uint32(2)),
		metadata("llama.context_length", gguf.ValueTypeUint32, uint32(2048)),
		metadata("llama.embedding_length", gguf.ValueTypeUint32, uint32(128)),
		metadata("llama.feed_forward_length", gguf.ValueTypeUint32, uint32(256)),
		metadata("llama.attention.head_count", gguf.ValueTypeUint32, uint32(4)),
		metadata("llama.attention.head_count_kv", gguf.ValueTypeUint32, uint32(2)),
		metadata("llama.rope.freq_base", gguf.ValueTypeFloat32, float32(10000)),
		metadata("llama.attention.layer_norm_rms_epsilon", gguf.ValueTypeFloat32, float32(1e-5)),
	}}
	spec, err := ReadSpec(file)
	if err != nil {
		t.Fatal(err)
	}
	if spec.KeyLength != 32 || spec.ValueLength != 32 {
		t.Fatalf("derived lengths = %d/%d, want 32/32", spec.KeyLength, spec.ValueLength)
	}
}

func TestReadSpecAcceptsLinearRoPEScaling(t *testing.T) {
	file := &gguf.File{Metadata: []gguf.Metadata{
		metadata("general.architecture", gguf.ValueTypeString, "llama"),
		metadata("llama.block_count", gguf.ValueTypeUint32, uint32(2)),
		metadata("llama.context_length", gguf.ValueTypeUint32, uint32(8192)),
		metadata("llama.embedding_length", gguf.ValueTypeUint32, uint32(128)),
		metadata("llama.feed_forward_length", gguf.ValueTypeUint32, uint32(256)),
		metadata("llama.attention.head_count", gguf.ValueTypeUint32, uint32(4)),
		metadata("llama.attention.head_count_kv", gguf.ValueTypeUint32, uint32(2)),
		metadata("llama.rope.freq_base", gguf.ValueTypeFloat32, float32(10000)),
		metadata("llama.rope.scaling.type", gguf.ValueTypeString, "linear"),
		metadata("llama.rope.scaling.factor", gguf.ValueTypeFloat32, float32(4)),
		metadata("llama.final_logit_softcapping", gguf.ValueTypeFloat32, float32(30)),
		metadata("llama.attention.layer_norm_rms_epsilon", gguf.ValueTypeFloat32, float32(1e-5)),
	}}
	spec, err := ReadSpec(file)
	if err != nil {
		t.Fatal(err)
	}
	if spec.RopeScalingType != "linear" ||
		spec.RopeScalingFactor != 4 ||
		spec.FinalLogitSoftcap != 30 {
		t.Fatalf(
			"scaling = %q/%v softcap=%v, want linear/4/30",
			spec.RopeScalingType,
			spec.RopeScalingFactor,
			spec.FinalLogitSoftcap,
		)
	}
}

func TestReadGemma2SpecDefaults(t *testing.T) {
	file := &gguf.File{Metadata: []gguf.Metadata{
		metadata("general.architecture", gguf.ValueTypeString, "gemma2"),
		metadata("gemma2.block_count", gguf.ValueTypeUint32, uint32(26)),
		metadata("gemma2.context_length", gguf.ValueTypeUint32, uint32(8192)),
		metadata("gemma2.embedding_length", gguf.ValueTypeUint32, uint32(2304)),
		metadata("gemma2.feed_forward_length", gguf.ValueTypeUint32, uint32(9216)),
		metadata("gemma2.attention.head_count", gguf.ValueTypeUint32, uint32(8)),
		metadata("gemma2.attention.head_count_kv", gguf.ValueTypeUint32, uint32(4)),
		metadata("gemma2.attention.key_length", gguf.ValueTypeUint32, uint32(256)),
		metadata("gemma2.attention.value_length", gguf.ValueTypeUint32, uint32(256)),
		metadata("gemma2.rope.freq_base", gguf.ValueTypeFloat32, float32(10000)),
		metadata("gemma2.attention.layer_norm_rms_epsilon", gguf.ValueTypeFloat32, float32(1e-6)),
		metadata("gemma2.attn_logit_softcapping", gguf.ValueTypeFloat32, float32(50)),
		metadata("gemma2.final_logit_softcapping", gguf.ValueTypeFloat32, float32(30)),
	}}
	spec, err := ReadSpec(file)
	if err != nil {
		t.Fatal(err)
	}
	if spec.Architecture != "gemma2" ||
		spec.RopeFrequencySWA != 10000 ||
		spec.SlidingWindow != 4096 ||
		spec.SlidingPattern != 2 ||
		spec.AttentionSoftcap != 50 ||
		spec.FinalLogitSoftcap != 30 ||
		!spec.IsSlidingLayer(0) ||
		spec.IsSlidingLayer(1) {
		t.Fatalf("unexpected Gemma 2 spec: %+v", spec)
	}
}

func TestReadGemmaSpec(t *testing.T) {
	file := &gguf.File{Metadata: []gguf.Metadata{
		metadata("general.architecture", gguf.ValueTypeString, "gemma"),
		metadata("gemma.block_count", gguf.ValueTypeUint32, uint32(18)),
		metadata("gemma.context_length", gguf.ValueTypeUint32, uint32(8192)),
		metadata("gemma.embedding_length", gguf.ValueTypeUint32, uint32(2048)),
		metadata("gemma.feed_forward_length", gguf.ValueTypeUint32, uint32(16384)),
		metadata("gemma.attention.head_count", gguf.ValueTypeUint32, uint32(8)),
		metadata("gemma.attention.head_count_kv", gguf.ValueTypeUint32, uint32(1)),
		metadata("gemma.attention.key_length", gguf.ValueTypeUint32, uint32(256)),
		metadata("gemma.attention.value_length", gguf.ValueTypeUint32, uint32(256)),
		metadata("gemma.rope.freq_base", gguf.ValueTypeFloat32, float32(10000)),
		metadata("gemma.attention.layer_norm_rms_epsilon", gguf.ValueTypeFloat32, float32(1e-6)),
	}}
	spec, err := ReadSpec(file)
	if err != nil {
		t.Fatal(err)
	}
	if spec.Architecture != "gemma" ||
		spec.BlockCount != 18 ||
		spec.IsSlidingLayer(0) {
		t.Fatalf("unexpected Gemma spec: %+v", spec)
	}
}

func TestReadQwen35Spec(t *testing.T) {
	file := &gguf.File{Metadata: []gguf.Metadata{
		metadata("general.architecture", gguf.ValueTypeString, "qwen35"),
		metadata("qwen35.block_count", gguf.ValueTypeUint32, uint32(32)),
		metadata("qwen35.context_length", gguf.ValueTypeUint32, uint32(262144)),
		metadata("qwen35.embedding_length", gguf.ValueTypeUint32, uint32(4096)),
		metadata("qwen35.feed_forward_length", gguf.ValueTypeUint32, uint32(12288)),
		metadata("qwen35.attention.head_count", gguf.ValueTypeUint32, uint32(16)),
		metadata("qwen35.attention.head_count_kv", gguf.ValueTypeUint32, uint32(4)),
		metadata("qwen35.attention.key_length", gguf.ValueTypeUint32, uint32(256)),
		metadata("qwen35.attention.value_length", gguf.ValueTypeUint32, uint32(256)),
		metadata("qwen35.rope.freq_base", gguf.ValueTypeFloat32, float32(10_000_000)),
		metadata("qwen35.rope.dimension_count", gguf.ValueTypeUint32, uint32(64)),
		{
			Key: "qwen35.rope.dimension_sections",
			Value: gguf.Value{
				Type:      gguf.ValueTypeArray,
				ArrayType: gguf.ValueTypeInt32,
				Data:      []int32{11, 11, 10, 0},
			},
		},
		metadata("qwen35.attention.layer_norm_rms_epsilon", gguf.ValueTypeFloat32, float32(1e-6)),
		metadata("qwen35.ssm.conv_kernel", gguf.ValueTypeUint32, uint32(4)),
		metadata("qwen35.ssm.inner_size", gguf.ValueTypeUint32, uint32(4096)),
		metadata("qwen35.ssm.state_size", gguf.ValueTypeUint32, uint32(128)),
		metadata("qwen35.ssm.time_step_rank", gguf.ValueTypeUint32, uint32(32)),
		metadata("qwen35.ssm.group_count", gguf.ValueTypeUint32, uint32(16)),
	}}
	spec, err := ReadSpec(file)
	if err != nil {
		t.Fatal(err)
	}
	if spec.Architecture != "qwen35" || spec.SSMStateSize != 128 ||
		spec.FullAttentionInterval != 4 ||
		spec.RopeSections != [4]int32{11, 11, 10, 0} ||
		!spec.IsRecurrentLayer(0) || spec.IsRecurrentLayer(3) {
		t.Fatalf("unexpected Qwen3.5 spec: %+v", spec)
	}
}

func TestReadQwen35ExplicitRecurrentLayers(t *testing.T) {
	file := &gguf.File{Metadata: []gguf.Metadata{
		metadata("general.architecture", gguf.ValueTypeString, "qwen35"),
		metadata("qwen35.block_count", gguf.ValueTypeUint32, uint32(2)),
		metadata("qwen35.context_length", gguf.ValueTypeUint32, uint32(1024)),
		metadata("qwen35.embedding_length", gguf.ValueTypeUint32, uint32(8)),
		metadata("qwen35.feed_forward_length", gguf.ValueTypeUint32, uint32(16)),
		metadata("qwen35.attention.head_count", gguf.ValueTypeUint32, uint32(2)),
		metadata("qwen35.attention.head_count_kv", gguf.ValueTypeUint32, uint32(1)),
		metadata("qwen35.attention.key_length", gguf.ValueTypeUint32, uint32(4)),
		metadata("qwen35.attention.value_length", gguf.ValueTypeUint32, uint32(4)),
		metadata("qwen35.rope.freq_base", gguf.ValueTypeFloat32, float32(10000)),
		metadata("qwen35.rope.dimension_count", gguf.ValueTypeUint32, uint32(4)),
		metadata("qwen35.attention.layer_norm_rms_epsilon", gguf.ValueTypeFloat32, float32(1e-6)),
		metadata("qwen35.ssm.conv_kernel", gguf.ValueTypeUint32, uint32(3)),
		metadata("qwen35.ssm.inner_size", gguf.ValueTypeUint32, uint32(4)),
		metadata("qwen35.ssm.state_size", gguf.ValueTypeUint32, uint32(2)),
		metadata("qwen35.ssm.time_step_rank", gguf.ValueTypeUint32, uint32(2)),
		metadata("qwen35.ssm.group_count", gguf.ValueTypeUint32, uint32(1)),
		{
			Key: "qwen35.rope.dimension_sections",
			Value: gguf.Value{
				Type: gguf.ValueTypeArray, ArrayType: gguf.ValueTypeInt32,
				Data: []int32{1, 1, 0, 0},
			},
		},
		{
			Key: "qwen35.attention.recurrent_layers",
			Value: gguf.Value{
				Type: gguf.ValueTypeArray, ArrayType: gguf.ValueTypeBool,
				Data: []bool{false, true},
			},
		},
	}}
	spec, err := ReadSpec(file)
	if err != nil {
		t.Fatal(err)
	}
	if spec.IsRecurrentLayer(0) || !spec.IsRecurrentLayer(1) {
		t.Fatalf("explicit recurrent layers were not preserved: %v", spec.RecurrentLayers)
	}
}

func TestReadT5EncoderSpec(t *testing.T) {
	file := &gguf.File{Metadata: []gguf.Metadata{
		metadata("general.architecture", gguf.ValueTypeString, "t5encoder"),
		metadata("t5encoder.block_count", gguf.ValueTypeUint32, uint32(24)),
		metadata("t5encoder.context_length", gguf.ValueTypeUint32, uint32(512)),
		metadata("t5encoder.embedding_length", gguf.ValueTypeUint32, uint32(4096)),
		metadata("t5encoder.feed_forward_length", gguf.ValueTypeUint32, uint32(10240)),
		metadata("t5encoder.attention.head_count", gguf.ValueTypeUint32, uint32(64)),
		metadata("t5encoder.attention.key_length", gguf.ValueTypeUint32, uint32(64)),
		metadata("t5encoder.attention.value_length", gguf.ValueTypeUint32, uint32(64)),
		metadata("t5encoder.attention.layer_norm_rms_epsilon", gguf.ValueTypeFloat32, float32(1e-6)),
		metadata("t5encoder.attention.relative_buckets_count", gguf.ValueTypeUint32, uint32(32)),
	}}
	spec, err := ReadSpec(file)
	if err != nil {
		t.Fatal(err)
	}
	if spec.Architecture != "t5encoder" || spec.HeadCountKV != 64 ||
		spec.RelativeBuckets != 32 || spec.RopeFrequencyBase != 0 {
		t.Fatalf("unexpected T5 encoder spec: %+v", spec)
	}
}

func TestReadErnie45MoESpec(t *testing.T) {
	file := &gguf.File{Metadata: []gguf.Metadata{
		metadata("general.architecture", gguf.ValueTypeString, "ernie4_5-moe"),
		metadata("ernie4_5-moe.block_count", gguf.ValueTypeUint32, uint32(4)),
		metadata("ernie4_5-moe.context_length", gguf.ValueTypeUint32, uint32(4096)),
		metadata("ernie4_5-moe.embedding_length", gguf.ValueTypeUint32, uint32(8)),
		metadata("ernie4_5-moe.feed_forward_length", gguf.ValueTypeUint32, uint32(12)),
		metadata("ernie4_5-moe.expert_count", gguf.ValueTypeUint32, uint32(4)),
		metadata("ernie4_5-moe.expert_used_count", gguf.ValueTypeUint32, uint32(2)),
		metadata("ernie4_5-moe.expert_feed_forward_length", gguf.ValueTypeUint32, uint32(6)),
		metadata("ernie4_5-moe.expert_shared_feed_forward_length", gguf.ValueTypeUint32, uint32(5)),
		metadata("ernie4_5-moe.expert_weights_scale", gguf.ValueTypeFloat32, float32(1.25)),
		metadata("ernie4_5-moe.interleave_moe_layer_step", gguf.ValueTypeUint32, uint32(2)),
		metadata("ernie4_5-moe.leading_dense_block_count", gguf.ValueTypeUint32, uint32(1)),
		metadata("ernie4_5-moe.attention.head_count", gguf.ValueTypeUint32, uint32(2)),
		metadata("ernie4_5-moe.attention.head_count_kv", gguf.ValueTypeUint32, uint32(1)),
		metadata("ernie4_5-moe.attention.key_length", gguf.ValueTypeUint32, uint32(4)),
		metadata("ernie4_5-moe.attention.value_length", gguf.ValueTypeUint32, uint32(4)),
		metadata("ernie4_5-moe.rope.freq_base", gguf.ValueTypeFloat32, float32(10000)),
		metadata("ernie4_5-moe.attention.layer_norm_rms_epsilon", gguf.ValueTypeFloat32, float32(1e-6)),
	}}
	spec, err := ReadSpec(file)
	if err != nil {
		t.Fatal(err)
	}
	if spec.MoELayerStep != 2 || spec.LeadingDenseBlocks != 1 ||
		spec.ExpertFeedForward != 6 || spec.SharedExpertFF != 5 ||
		spec.ExpertWeightsScale != 1.25 || !spec.ExpertWeightsNorm ||
		spec.IsInterleavedMoELayer(0) || !spec.IsInterleavedMoELayer(1) ||
		spec.IsInterleavedMoELayer(2) || !spec.IsInterleavedMoELayer(3) ||
		!usesNormalRoPE(spec.Architecture) || !spec.UsesRoPE(0) {
		t.Fatalf("unexpected ERNIE 4.5 MoE spec: %+v", spec)
	}
}

func TestReadPaddleOCRSpec(t *testing.T) {
	file := &gguf.File{Metadata: []gguf.Metadata{
		metadata("general.architecture", gguf.ValueTypeString, "paddleocr"),
		metadata("paddleocr.block_count", gguf.ValueTypeUint32, uint32(2)),
		metadata("paddleocr.context_length", gguf.ValueTypeUint32, uint32(4096)),
		metadata("paddleocr.embedding_length", gguf.ValueTypeUint32, uint32(8)),
		metadata("paddleocr.feed_forward_length", gguf.ValueTypeUint32, uint32(12)),
		metadata("paddleocr.attention.head_count", gguf.ValueTypeUint32, uint32(2)),
		metadata("paddleocr.attention.head_count_kv", gguf.ValueTypeUint32, uint32(1)),
		metadata("paddleocr.attention.key_length", gguf.ValueTypeUint32, uint32(4)),
		metadata("paddleocr.attention.value_length", gguf.ValueTypeUint32, uint32(4)),
		metadata("paddleocr.rope.freq_base", gguf.ValueTypeFloat32, float32(10000)),
		metadata("paddleocr.attention.layer_norm_rms_epsilon", gguf.ValueTypeFloat32, float32(1e-6)),
		{Key: "paddleocr.rope.dimension_sections", Value: gguf.Value{
			Type: gguf.ValueTypeArray, ArrayType: gguf.ValueTypeInt32, Data: []int32{1, 1, 0, 0},
		}},
	}}
	spec, err := ReadSpec(file)
	if err != nil {
		t.Fatal(err)
	}
	if spec.RopeDimensionCount != 4 || spec.RopeSections != [4]int32{1, 1, 0, 0} ||
		usesNormalRoPE(spec.Architecture) || !spec.UsesRoPE(1) {
		t.Fatalf("unexpected PaddleOCR spec: %+v", spec)
	}
}

func TestReadSpecRejectsUnsupportedArchitecture(t *testing.T) {
	file := &gguf.File{Metadata: []gguf.Metadata{
		metadata("general.architecture", gguf.ValueTypeString, "mamba"),
	}}
	_, err := ReadSpec(file)
	var unsupported *UnsupportedArchitectureError
	if !errors.As(err, &unsupported) {
		t.Fatalf("error = %v, want UnsupportedArchitectureError", err)
	}
}

func TestReadSpecRejectsUnsupportedRoPEScaling(t *testing.T) {
	file := &gguf.File{Metadata: []gguf.Metadata{
		metadata("general.architecture", gguf.ValueTypeString, "llama"),
		metadata("llama.block_count", gguf.ValueTypeUint32, uint32(2)),
		metadata("llama.context_length", gguf.ValueTypeUint32, uint32(2048)),
		metadata("llama.embedding_length", gguf.ValueTypeUint32, uint32(128)),
		metadata("llama.feed_forward_length", gguf.ValueTypeUint32, uint32(256)),
		metadata("llama.attention.head_count", gguf.ValueTypeUint32, uint32(4)),
		metadata("llama.attention.head_count_kv", gguf.ValueTypeUint32, uint32(2)),
		metadata("llama.rope.freq_base", gguf.ValueTypeFloat32, float32(10000)),
		metadata("llama.rope.scaling.type", gguf.ValueTypeString, "llama3"),
		metadata("llama.attention.layer_norm_rms_epsilon", gguf.ValueTypeFloat32, float32(1e-5)),
	}}
	_, err := ReadSpec(file)
	if err == nil || !strings.Contains(err.Error(), "unsupported RoPE scaling") {
		t.Fatalf("error = %v, want unsupported RoPE scaling", err)
	}
}

func metadata(key string, valueType gguf.ValueType, data any) gguf.Metadata {
	return gguf.Metadata{Key: key, Value: gguf.Value{Type: valueType, Data: data}}
}

func graniteMetadata() []gguf.Metadata {
	return []gguf.Metadata{
		metadata("general.architecture", gguf.ValueTypeString, "granite"),
		metadata("granite.block_count", gguf.ValueTypeUint32, uint32(32)),
		metadata("granite.context_length", gguf.ValueTypeUint32, uint32(4096)),
		metadata("granite.embedding_length", gguf.ValueTypeUint32, uint32(4096)),
		metadata("granite.feed_forward_length", gguf.ValueTypeUint32, uint32(11008)),
		metadata("granite.attention.head_count", gguf.ValueTypeUint32, uint32(32)),
		metadata("granite.attention.head_count_kv", gguf.ValueTypeUint32, uint32(8)),
		metadata("granite.rope.freq_base", gguf.ValueTypeFloat32, float32(10000)),
		metadata("granite.attention.layer_norm_rms_epsilon", gguf.ValueTypeFloat32, float32(1e-5)),
		metadata("granite.logit_scale", gguf.ValueTypeFloat32, float32(8)),
	}
}

func mistral3Metadata() []gguf.Metadata {
	return []gguf.Metadata{
		metadata("general.architecture", gguf.ValueTypeString, "mistral3"),
		metadata("mistral3.block_count", gguf.ValueTypeUint32, uint32(34)),
		metadata("mistral3.context_length", gguf.ValueTypeUint32, uint32(32768)),
		metadata("mistral3.embedding_length", gguf.ValueTypeUint32, uint32(4096)),
		metadata("mistral3.feed_forward_length", gguf.ValueTypeUint32, uint32(14336)),
		metadata("mistral3.attention.head_count", gguf.ValueTypeUint32, uint32(32)),
		metadata("mistral3.attention.head_count_kv", gguf.ValueTypeUint32, uint32(8)),
		metadata("mistral3.rope.freq_base", gguf.ValueTypeFloat32, float32(1_000_000)),
		metadata("mistral3.attention.layer_norm_rms_epsilon", gguf.ValueTypeFloat32, float32(1e-5)),
	}
}
