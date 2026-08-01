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

func TestReadLlama4Spec(t *testing.T) {
	file := &gguf.File{Metadata: []gguf.Metadata{
		metadata("general.architecture", gguf.ValueTypeString, "llama4"),
		metadata("llama4.block_count", gguf.ValueTypeUint32, uint32(4)),
		metadata("llama4.context_length", gguf.ValueTypeUint32, uint32(131072)),
		metadata("llama4.embedding_length", gguf.ValueTypeUint32, uint32(8)),
		metadata("llama4.feed_forward_length", gguf.ValueTypeUint32, uint32(16)),
		metadata("llama4.attention.head_count", gguf.ValueTypeUint32, uint32(2)),
		metadata("llama4.attention.head_count_kv", gguf.ValueTypeUint32, uint32(1)),
		metadata("llama4.rope.freq_base", gguf.ValueTypeFloat32, float32(500000)),
		metadata("llama4.rope.dimension_count", gguf.ValueTypeUint32, uint32(4)),
		metadata("llama4.attention.layer_norm_rms_epsilon", gguf.ValueTypeFloat32, float32(1e-5)),
		metadata("llama4.attention.sliding_window_pattern", gguf.ValueTypeUint32, uint32(4)),
		metadata("llama4.expert_count", gguf.ValueTypeUint32, uint32(16)),
		metadata("llama4.expert_used_count", gguf.ValueTypeUint32, uint32(2)),
		metadata("llama4.expert_feed_forward_length", gguf.ValueTypeUint32, uint32(6)),
		metadata("llama4.interleave_moe_layer_step", gguf.ValueTypeUint32, uint32(2)),
	}}
	spec, err := ReadSpec(file)
	if err != nil {
		t.Fatal(err)
	}
	if spec.Architecture != "llama4" || spec.SlidingWindow != 8192 || spec.SlidingPattern != 4 ||
		spec.AttentionTempFloor != 8192 || spec.AttentionTempScale != 0.1 || spec.AttentionTempOffset != 1 ||
		spec.ExpertCount != 16 || spec.ExpertUsedCount != 2 || spec.ExpertFeedForward != 6 ||
		spec.SharedExpertFF != 6 || !spec.IsSlidingLayer(2) || spec.IsSlidingLayer(3) ||
		!spec.UsesRoPE(2) || spec.UsesRoPE(3) || !spec.IsInterleavedMoELayer(1) {
		t.Fatalf("unexpected Llama 4 spec: %+v", spec)
	}
}

func TestReadGPTOSSSpec(t *testing.T) {
	file := &gguf.File{Metadata: []gguf.Metadata{
		metadata("general.architecture", gguf.ValueTypeString, "gpt-oss"),
		metadata("gpt-oss.block_count", gguf.ValueTypeUint32, uint32(24)),
		metadata("gpt-oss.context_length", gguf.ValueTypeUint32, uint32(131072)),
		metadata("gpt-oss.embedding_length", gguf.ValueTypeUint32, uint32(2880)),
		metadata("gpt-oss.feed_forward_length", gguf.ValueTypeUint32, uint32(2880)),
		metadata("gpt-oss.attention.head_count", gguf.ValueTypeUint32, uint32(64)),
		metadata("gpt-oss.attention.head_count_kv", gguf.ValueTypeUint32, uint32(8)),
		metadata("gpt-oss.attention.key_length", gguf.ValueTypeUint32, uint32(64)),
		metadata("gpt-oss.attention.value_length", gguf.ValueTypeUint32, uint32(64)),
		metadata("gpt-oss.rope.freq_base", gguf.ValueTypeFloat32, float32(150000)),
		metadata("gpt-oss.rope.freq_base_swa", gguf.ValueTypeFloat32, float32(10000)),
		metadata("gpt-oss.attention.layer_norm_rms_epsilon", gguf.ValueTypeFloat32, float32(1e-5)),
		metadata("gpt-oss.attention.sliding_window", gguf.ValueTypeUint32, uint32(128)),
		metadata("gpt-oss.expert_count", gguf.ValueTypeUint32, uint32(32)),
		metadata("gpt-oss.expert_used_count", gguf.ValueTypeUint32, uint32(4)),
		metadata("gpt-oss.expert_feed_forward_length", gguf.ValueTypeUint32, uint32(2880)),
	}}
	spec, err := ReadSpec(file)
	if err != nil {
		t.Fatal(err)
	}
	if spec.KeyLength != 64 || spec.RopeDimensionCount != 64 || spec.SlidingWindow != 128 ||
		spec.SlidingPattern != 2 || !spec.IsSlidingLayer(0) || spec.IsSlidingLayer(1) ||
		spec.RopeFrequencySWA != 10000 || spec.ExpertGatingFunc != 3 || spec.ExpertWeightsNorm {
		t.Fatalf("unexpected GPT-OSS spec: %+v", spec)
	}
}

func TestReadGroveMoESpec(t *testing.T) {
	file := &gguf.File{Metadata: []gguf.Metadata{
		metadata("general.architecture", gguf.ValueTypeString, "grovemoe"),
		metadata("grovemoe.block_count", gguf.ValueTypeUint32, uint32(2)),
		metadata("grovemoe.context_length", gguf.ValueTypeUint32, uint32(4096)),
		metadata("grovemoe.embedding_length", gguf.ValueTypeUint32, uint32(8)),
		metadata("grovemoe.feed_forward_length", gguf.ValueTypeUint32, uint32(24)),
		metadata("grovemoe.expert_feed_forward_length", gguf.ValueTypeUint32, uint32(12)),
		metadata("grovemoe.expert_chunk_feed_forward_length", gguf.ValueTypeUint32, uint32(4)),
		metadata("grovemoe.expert_count", gguf.ValueTypeUint32, uint32(8)),
		metadata("grovemoe.expert_used_count", gguf.ValueTypeUint32, uint32(2)),
		metadata("grovemoe.expert_group_scale", gguf.ValueTypeFloat32, float32(0.5)),
		metadata("grovemoe.experts_per_group", gguf.ValueTypeUint32, uint32(4)),
		metadata("grovemoe.attention.head_count", gguf.ValueTypeUint32, uint32(2)),
		metadata("grovemoe.attention.head_count_kv", gguf.ValueTypeUint32, uint32(1)),
		metadata("grovemoe.attention.key_length", gguf.ValueTypeUint32, uint32(4)),
		metadata("grovemoe.attention.value_length", gguf.ValueTypeUint32, uint32(4)),
		metadata("grovemoe.rope.freq_base", gguf.ValueTypeFloat32, float32(10000)),
		metadata("grovemoe.attention.layer_norm_rms_epsilon", gguf.ValueTypeFloat32, float32(1e-6)),
	}}
	spec, err := ReadSpec(file)
	if err != nil {
		t.Fatal(err)
	}
	if spec.Architecture != "grovemoe" || spec.ExpertCount != 8 ||
		spec.ExpertChunkFeedForward != 4 || spec.ExpertsPerGroup != 4 ||
		spec.ExpertGroupScale != 0.5 {
		t.Fatalf("unexpected GroveMoE spec: %+v", spec)
	}
}

func TestReadGLM4MoESpec(t *testing.T) {
	file := &gguf.File{Metadata: []gguf.Metadata{
		metadata("general.architecture", gguf.ValueTypeString, "glm4moe"),
		metadata("glm4moe.block_count", gguf.ValueTypeUint32, uint32(3)),
		metadata("glm4moe.nextn_predict_layers", gguf.ValueTypeUint32, uint32(1)),
		metadata("glm4moe.context_length", gguf.ValueTypeUint32, uint32(4096)),
		metadata("glm4moe.embedding_length", gguf.ValueTypeUint32, uint32(8)),
		metadata("glm4moe.feed_forward_length", gguf.ValueTypeUint32, uint32(12)),
		metadata("glm4moe.expert_feed_forward_length", gguf.ValueTypeUint32, uint32(6)),
		metadata("glm4moe.expert_count", gguf.ValueTypeUint32, uint32(4)),
		metadata("glm4moe.expert_used_count", gguf.ValueTypeUint32, uint32(2)),
		metadata("glm4moe.expert_shared_count", gguf.ValueTypeUint32, uint32(2)),
		metadata("glm4moe.leading_dense_block_count", gguf.ValueTypeUint32, uint32(1)),
		metadata("glm4moe.expert_weights_scale", gguf.ValueTypeFloat32, float32(1.5)),
		metadata("glm4moe.expert_weights_norm", gguf.ValueTypeBool, true),
		metadata("glm4moe.attention.head_count", gguf.ValueTypeUint32, uint32(2)),
		metadata("glm4moe.attention.head_count_kv", gguf.ValueTypeUint32, uint32(1)),
		metadata("glm4moe.attention.key_length", gguf.ValueTypeUint32, uint32(4)),
		metadata("glm4moe.attention.value_length", gguf.ValueTypeUint32, uint32(4)),
		metadata("glm4moe.rope.dimension_count", gguf.ValueTypeUint32, uint32(4)),
		{Key: "glm4moe.rope.dimension_sections", Value: gguf.Value{
			Type: gguf.ValueTypeArray, ArrayType: gguf.ValueTypeInt32, Data: []int32{1, 1, 0, 0},
		}},
		metadata("glm4moe.rope.freq_base", gguf.ValueTypeFloat32, float32(10000)),
		metadata("glm4moe.attention.layer_norm_rms_epsilon", gguf.ValueTypeFloat32, float32(1e-6)),
	}}
	spec, err := ReadSpec(file)
	if err != nil {
		t.Fatal(err)
	}
	if spec.Architecture != "glm4moe" || spec.BlockCount != 2 ||
		spec.LeadingDenseBlocks != 1 || spec.SharedExpertFF != 12 ||
		spec.ExpertGatingFunc != 2 || !spec.ExpertWeightsNorm || spec.RopeSections[1] != 1 {
		t.Fatalf("unexpected GLM4-MoE spec: %+v", spec)
	}
}

func TestReadMiMo2SpecTrimsMTPArrays(t *testing.T) {
	file := &gguf.File{Metadata: []gguf.Metadata{
		metadata("general.architecture", gguf.ValueTypeString, "mimo2"),
		metadata("mimo2.block_count", gguf.ValueTypeUint32, uint32(4)),
		metadata("mimo2.nextn_predict_layers", gguf.ValueTypeUint32, uint32(1)),
		metadata("mimo2.context_length", gguf.ValueTypeUint32, uint32(4096)),
		metadata("mimo2.embedding_length", gguf.ValueTypeUint32, uint32(8)),
		metadata("mimo2.feed_forward_length", gguf.ValueTypeUint32, uint32(12)),
		metadata("mimo2.expert_feed_forward_length", gguf.ValueTypeUint32, uint32(6)),
		metadata("mimo2.expert_count", gguf.ValueTypeUint32, uint32(4)),
		metadata("mimo2.expert_used_count", gguf.ValueTypeUint32, uint32(2)),
		metadata("mimo2.attention.head_count", gguf.ValueTypeUint32, uint32(2)),
		{Key: "mimo2.attention.head_count_kv", Value: gguf.Value{
			Type: gguf.ValueTypeArray, ArrayType: gguf.ValueTypeInt32, Data: []int32{1, 2, 1, 1},
		}},
		metadata("mimo2.attention.key_length", gguf.ValueTypeUint32, uint32(4)),
		metadata("mimo2.attention.value_length", gguf.ValueTypeUint32, uint32(3)),
		metadata("mimo2.rope.dimension_count", gguf.ValueTypeUint32, uint32(4)),
		metadata("mimo2.rope.freq_base", gguf.ValueTypeFloat32, float32(10000)),
		metadata("mimo2.rope.freq_base_swa", gguf.ValueTypeFloat32, float32(20000)),
		metadata("mimo2.attention.sliding_window", gguf.ValueTypeUint32, uint32(128)),
		{Key: "mimo2.attention.sliding_window_pattern", Value: gguf.Value{
			Type: gguf.ValueTypeArray, ArrayType: gguf.ValueTypeInt32, Data: []int32{1, 0, 1, 1},
		}},
		metadata("mimo2.attention.value_scale", gguf.ValueTypeFloat32, float32(0.5)),
		metadata("mimo2.attention.layer_norm_rms_epsilon", gguf.ValueTypeFloat32, float32(1e-5)),
	}}
	spec, err := ReadSpec(file)
	if err != nil {
		t.Fatal(err)
	}
	if spec.Architecture != "mimo2" || spec.BlockCount != 3 ||
		len(spec.LayerKVHeadCounts) != 3 || spec.LayerKVHeadCount(1) != 2 ||
		len(spec.SlidingLayers) != 3 || !spec.IsSlidingLayer(0) || spec.IsSlidingLayer(1) ||
		!spec.IsSlidingLayer(2) || spec.AttentionValueScale != 0.5 ||
		spec.RopeDimensionCount != 4 || spec.RopeFrequencySWA != 20000 ||
		spec.ExpertGatingFunc != 2 || !spec.ExpertWeightsNorm {
		t.Fatalf("unexpected MiMo2 spec: %+v", spec)
	}
}

func TestReadStep35SpecTrimsMTPArrays(t *testing.T) {
	file := &gguf.File{Metadata: []gguf.Metadata{
		metadata("general.architecture", gguf.ValueTypeString, "step35"),
		metadata("step35.block_count", gguf.ValueTypeUint32, uint32(3)),
		metadata("step35.nextn_predict_layers", gguf.ValueTypeUint32, uint32(1)),
		metadata("step35.context_length", gguf.ValueTypeUint32, uint32(4096)),
		metadata("step35.embedding_length", gguf.ValueTypeUint32, uint32(8)),
		metadata("step35.feed_forward_length", gguf.ValueTypeUint32, uint32(12)),
		metadata("step35.expert_feed_forward_length", gguf.ValueTypeUint32, uint32(6)),
		metadata("step35.expert_shared_feed_forward_length", gguf.ValueTypeUint32, uint32(8)),
		metadata("step35.expert_count", gguf.ValueTypeUint32, uint32(4)),
		metadata("step35.expert_used_count", gguf.ValueTypeUint32, uint32(2)),
		metadata("step35.expert_weights_scale", gguf.ValueTypeFloat32, float32(1.25)),
		metadata("step35.expert_weights_norm", gguf.ValueTypeBool, true),
		metadata("step35.leading_dense_block_count", gguf.ValueTypeUint32, uint32(1)),
		metadata("step35.moe_every_n_layers", gguf.ValueTypeUint32, uint32(1)),
		{Key: "step35.attention.head_count", Value: gguf.Value{
			Type: gguf.ValueTypeArray, ArrayType: gguf.ValueTypeInt32, Data: []int32{2, 4, 2},
		}},
		{Key: "step35.attention.head_count_kv", Value: gguf.Value{
			Type: gguf.ValueTypeArray, ArrayType: gguf.ValueTypeInt32, Data: []int32{1, 2, 1},
		}},
		metadata("step35.attention.key_length", gguf.ValueTypeUint32, uint32(4)),
		metadata("step35.attention.value_length", gguf.ValueTypeUint32, uint32(4)),
		metadata("step35.rope.freq_base", gguf.ValueTypeFloat32, float32(10000)),
		metadata("step35.rope.freq_base_swa", gguf.ValueTypeFloat32, float32(20000)),
		metadata("step35.attention.sliding_window", gguf.ValueTypeUint32, uint32(128)),
		{Key: "step35.attention.sliding_window_pattern", Value: gguf.Value{
			Type: gguf.ValueTypeArray, ArrayType: gguf.ValueTypeBool, Data: []bool{false, true, false},
		}},
		{Key: "step35.swiglu_clamp_exp", Value: gguf.Value{
			Type: gguf.ValueTypeArray, ArrayType: gguf.ValueTypeFloat32, Data: []float32{2, 3, 0},
		}},
		{Key: "step35.swiglu_clamp_shexp", Value: gguf.Value{
			Type: gguf.ValueTypeArray, ArrayType: gguf.ValueTypeFloat32, Data: []float32{4, 5, 0},
		}},
		metadata("step35.attention.layer_norm_rms_epsilon", gguf.ValueTypeFloat32, float32(1e-5)),
	}}
	spec, err := ReadSpec(file)
	if err != nil {
		t.Fatal(err)
	}
	if spec.BlockCount != 2 || len(spec.LayerHeadCounts) != 2 || len(spec.LayerKVHeadCounts) != 2 ||
		spec.LayerHeadCount(1) != 4 || spec.LayerKVHeadCount(1) != 2 ||
		spec.LayerRopeDimensionCount(0) != 2 || spec.LayerRopeDimensionCount(1) != 4 ||
		spec.IsSlidingLayer(0) || !spec.IsSlidingLayer(1) ||
		spec.LayerExpertSwiGLUClamp(1) != 3 || spec.LayerSharedSwiGLUClampLimit(1) != 5 ||
		spec.ExpertGatingFunc != 2 || !spec.ExpertWeightsNorm || spec.SharedExpertFF != 8 {
		t.Fatalf("unexpected Step3.5 spec: %+v", spec)
	}
}

func TestReadGemma4Spec(t *testing.T) {
	file := &gguf.File{Metadata: []gguf.Metadata{
		metadata("general.architecture", gguf.ValueTypeString, "gemma4"),
		metadata("gemma4.block_count", gguf.ValueTypeUint32, uint32(4)),
		metadata("gemma4.context_length", gguf.ValueTypeUint32, uint32(4096)),
		metadata("gemma4.embedding_length", gguf.ValueTypeUint32, uint32(8)),
		{Key: "gemma4.feed_forward_length", Value: gguf.Value{
			Type: gguf.ValueTypeArray, ArrayType: gguf.ValueTypeInt32, Data: []int32{6, 6, 12, 12},
		}},
		metadata("gemma4.attention.head_count", gguf.ValueTypeUint32, uint32(2)),
		{Key: "gemma4.attention.head_count_kv", Value: gguf.Value{
			Type: gguf.ValueTypeArray, ArrayType: gguf.ValueTypeInt32, Data: []int32{1, 1, 1, 1},
		}},
		metadata("gemma4.attention.key_length", gguf.ValueTypeUint32, uint32(4)),
		metadata("gemma4.attention.value_length", gguf.ValueTypeUint32, uint32(4)),
		metadata("gemma4.attention.key_length_swa", gguf.ValueTypeUint32, uint32(2)),
		metadata("gemma4.attention.value_length_swa", gguf.ValueTypeUint32, uint32(2)),
		metadata("gemma4.rope.dimension_count", gguf.ValueTypeUint32, uint32(4)),
		metadata("gemma4.rope.dimension_count_swa", gguf.ValueTypeUint32, uint32(2)),
		metadata("gemma4.rope.freq_base", gguf.ValueTypeFloat32, float32(1_000_000)),
		metadata("gemma4.rope.freq_base_swa", gguf.ValueTypeFloat32, float32(10_000)),
		metadata("gemma4.attention.sliding_window", gguf.ValueTypeUint32, uint32(512)),
		{Key: "gemma4.attention.sliding_window_pattern", Value: gguf.Value{
			Type: gguf.ValueTypeArray, ArrayType: gguf.ValueTypeBool, Data: []bool{true, false, true, false},
		}},
		metadata("gemma4.attention.shared_kv_layers", gguf.ValueTypeUint32, uint32(2)),
		metadata("gemma4.embedding_length_per_layer_input", gguf.ValueTypeUint32, uint32(3)),
		metadata("gemma4.attention.layer_norm_rms_epsilon", gguf.ValueTypeFloat32, float32(1e-6)),
		metadata("gemma4.expert_count", gguf.ValueTypeUint32, uint32(4)),
		metadata("gemma4.expert_used_count", gguf.ValueTypeUint32, uint32(2)),
		metadata("gemma4.expert_feed_forward_length", gguf.ValueTypeUint32, uint32(3)),
	}}
	spec, err := ReadSpec(file)
	if err != nil {
		t.Fatal(err)
	}
	if spec.Architecture != "gemma4" || spec.LayerFeedForwardLength(3) != 12 ||
		spec.LayerKeyLength(0) != 2 || spec.LayerKeyLength(1) != 4 ||
		spec.LayerRopeDimensionCount(0) != 2 || spec.LayerRopeDimensionCount(1) != 4 ||
		spec.LayerHasKV(2) || spec.LayerSharedKVSource(2) != 0 || spec.LayerSharedKVSource(3) != 1 ||
		spec.EmbeddingPerLayer != 3 || spec.ExpertCount != 4 || spec.AttentionScale != 1 {
		t.Fatalf("unexpected Gemma 4 spec: %+v", spec)
	}
}

func TestReadGemma3nSpec(t *testing.T) {
	file := &gguf.File{Metadata: []gguf.Metadata{
		metadata("general.architecture", gguf.ValueTypeString, "gemma3n"),
		metadata("gemma3n.block_count", gguf.ValueTypeUint32, uint32(30)),
		metadata("gemma3n.context_length", gguf.ValueTypeUint32, uint32(32768)),
		metadata("gemma3n.embedding_length", gguf.ValueTypeUint32, uint32(2048)),
		metadata("gemma3n.feed_forward_length", gguf.ValueTypeUint32, uint32(8192)),
		metadata("gemma3n.attention.head_count", gguf.ValueTypeUint32, uint32(8)),
		metadata("gemma3n.attention.head_count_kv", gguf.ValueTypeUint32, uint32(4)),
		metadata("gemma3n.attention.key_length", gguf.ValueTypeUint32, uint32(256)),
		metadata("gemma3n.attention.value_length", gguf.ValueTypeUint32, uint32(256)),
		metadata("gemma3n.rope.freq_base", gguf.ValueTypeFloat32, float32(1_000_000)),
		metadata("gemma3n.rope.freq_base_swa", gguf.ValueTypeFloat32, float32(10_000)),
		metadata("gemma3n.attention.sliding_window", gguf.ValueTypeUint32, uint32(1024)),
		metadata("gemma3n.attention.sliding_window_pattern", gguf.ValueTypeUint32, uint32(5)),
		metadata("gemma3n.attention.layer_norm_rms_epsilon", gguf.ValueTypeFloat32, float32(1e-6)),
		metadata("gemma3n.final_logit_softcapping", gguf.ValueTypeFloat32, float32(30)),
		metadata("gemma3n.vocab_size", gguf.ValueTypeUint32, uint32(262144)),
	}}
	spec, err := ReadSpec(file)
	if err != nil {
		t.Fatal(err)
	}
	if spec.Architecture != "gemma3n" || spec.AltUpCount != 4 || spec.LaurelRank != 64 ||
		spec.EmbeddingPerLayer != 256 || spec.KVFromStart != 20 || spec.SharedKVLayers != 10 ||
		!spec.IsSlidingLayer(18) || spec.IsSlidingLayer(19) || spec.LayerHasKV(20) ||
		spec.LayerSharedKVSource(20) != 18 || spec.LayerSharedKVSource(24) != 19 ||
		spec.InputEmbeddingScale() != float32(math.Sqrt(2048)) {
		t.Fatalf("unexpected Gemma 3n spec: %+v", spec)
	}
}

func TestReadGemma4AssistantSpec(t *testing.T) {
	file := &gguf.File{Metadata: []gguf.Metadata{
		metadata("general.architecture", gguf.ValueTypeString, "gemma4-assistant"),
		metadata("gemma4-assistant.block_count", gguf.ValueTypeUint32, uint32(2)),
		metadata("gemma4-assistant.context_length", gguf.ValueTypeUint32, uint32(4096)),
		metadata("gemma4-assistant.embedding_length", gguf.ValueTypeUint32, uint32(8)),
		metadata("gemma4-assistant.embedding_length_out", gguf.ValueTypeUint32, uint32(12)),
		metadata("gemma4-assistant.feed_forward_length", gguf.ValueTypeUint32, uint32(16)),
		metadata("gemma4-assistant.attention.head_count", gguf.ValueTypeUint32, uint32(2)),
		metadata("gemma4-assistant.attention.head_count_kv", gguf.ValueTypeUint32, uint32(1)),
		metadata("gemma4-assistant.attention.key_length", gguf.ValueTypeUint32, uint32(4)),
		metadata("gemma4-assistant.attention.value_length", gguf.ValueTypeUint32, uint32(4)),
		metadata("gemma4-assistant.attention.key_length_swa", gguf.ValueTypeUint32, uint32(2)),
		metadata("gemma4-assistant.attention.value_length_swa", gguf.ValueTypeUint32, uint32(2)),
		metadata("gemma4-assistant.rope.freq_base", gguf.ValueTypeFloat32, float32(1_000_000)),
		metadata("gemma4-assistant.rope.freq_base_swa", gguf.ValueTypeFloat32, float32(10_000)),
		metadata("gemma4-assistant.attention.sliding_window", gguf.ValueTypeUint32, uint32(512)),
		{Key: "gemma4-assistant.attention.sliding_window_pattern", Value: gguf.Value{
			Type: gguf.ValueTypeArray, ArrayType: gguf.ValueTypeBool, Data: []bool{true, false},
		}},
		metadata("gemma4-assistant.attention.layer_norm_rms_epsilon", gguf.ValueTypeFloat32, float32(1e-6)),
		metadata("gemma4-assistant.nextn_predict_layers", gguf.ValueTypeUint32, uint32(2)),
		metadata("gemma4-assistant.vocab_size", gguf.ValueTypeUint32, uint32(32)),
	}}
	spec, err := ReadSpec(file)
	if err != nil {
		t.Fatal(err)
	}
	if spec.Architecture != "gemma4-assistant" || spec.TargetHiddenSize != 12 ||
		spec.LayerKeyLength(0) != 2 || spec.LayerKeyLength(1) != 4 ||
		spec.LayerRopeDimensionCount(0) != 2 || spec.LayerRopeDimensionCount(1) != 4 ||
		!spec.IsSlidingLayer(0) || spec.IsSlidingLayer(1) || spec.AttentionScale != 1 {
		t.Fatalf("unexpected Gemma 4 assistant spec: %+v", spec)
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

func TestReadJinaBERTV3SpecUsesBidirectionalNeoXRoPE(t *testing.T) {
	file := &gguf.File{Metadata: []gguf.Metadata{
		metadata("general.architecture", gguf.ValueTypeString, "jina-bert-v3"),
		metadata("jina-bert-v3.block_count", gguf.ValueTypeUint32, uint32(24)),
		metadata("jina-bert-v3.context_length", gguf.ValueTypeUint32, uint32(8192)),
		metadata("jina-bert-v3.embedding_length", gguf.ValueTypeUint32, uint32(8)),
		metadata("jina-bert-v3.feed_forward_length", gguf.ValueTypeUint32, uint32(16)),
		metadata("jina-bert-v3.vocab_size", gguf.ValueTypeUint32, uint32(32)),
		metadata("jina-bert-v3.attention.head_count", gguf.ValueTypeUint32, uint32(2)),
		metadata("jina-bert-v3.attention.layer_norm_epsilon", gguf.ValueTypeFloat32, float32(1e-5)),
		metadata("tokenizer.ggml.token_type_count", gguf.ValueTypeUint32, uint32(2)),
	}}
	spec, err := ReadSpec(file)
	if err != nil {
		t.Fatal(err)
	}
	if spec.Architecture != "jina-bert-v3" || !spec.NonCausalAttention || !spec.IsEncoderOnly() ||
		spec.HeadCountKV != 2 || spec.KeyLength != 4 || spec.ValueLength != 4 ||
		spec.RopeDimensionCount != 4 || spec.RopeFrequencyBase != 10000 ||
		spec.TokenTypeCount != 2 || !spec.UsesLayerNorm() || usesNormalRoPE(spec.Architecture) ||
		!usesPostOnlyNorm(spec.Architecture) || !usesGELU(spec.Architecture) {
		t.Fatalf("unexpected JinaBERT v3 spec: %+v", spec)
	}
}

func TestReadNomicBERTMoESpecUsesInterleavedExperts(t *testing.T) {
	file := &gguf.File{Metadata: []gguf.Metadata{
		metadata("general.architecture", gguf.ValueTypeString, "nomic-bert-moe"),
		metadata("nomic-bert-moe.block_count", gguf.ValueTypeUint32, uint32(12)),
		metadata("nomic-bert-moe.context_length", gguf.ValueTypeUint32, uint32(8192)),
		metadata("nomic-bert-moe.embedding_length", gguf.ValueTypeUint32, uint32(8)),
		metadata("nomic-bert-moe.feed_forward_length", gguf.ValueTypeUint32, uint32(16)),
		metadata("nomic-bert-moe.vocab_size", gguf.ValueTypeUint32, uint32(32)),
		metadata("nomic-bert-moe.attention.head_count", gguf.ValueTypeUint32, uint32(2)),
		metadata("nomic-bert-moe.attention.layer_norm_epsilon", gguf.ValueTypeFloat32, float32(1e-5)),
		metadata("nomic-bert-moe.expert_count", gguf.ValueTypeUint32, uint32(4)),
		metadata("nomic-bert-moe.expert_used_count", gguf.ValueTypeUint32, uint32(2)),
		metadata("nomic-bert-moe.moe_every_n_layers", gguf.ValueTypeUint32, uint32(2)),
		metadata("tokenizer.ggml.token_type_count", gguf.ValueTypeUint32, uint32(2)),
	}}
	spec, err := ReadSpec(file)
	if err != nil {
		t.Fatal(err)
	}
	if !spec.NonCausalAttention || !spec.IsEncoderOnly() || spec.ExpertCount != 4 ||
		spec.ExpertUsedCount != 2 || spec.ExpertFeedForward != 16 || spec.MoELayerStep != 2 ||
		spec.IsInterleavedMoELayer(0) || !spec.IsInterleavedMoELayer(1) ||
		spec.IsInterleavedMoELayer(2) || !spec.IsInterleavedMoELayer(3) ||
		spec.RopeDimensionCount != 4 || spec.RopeFrequencyBase != 10000 {
		t.Fatalf("unexpected NomicBERT-MoE spec: %+v", spec)
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

func TestReadCogVLMSpecUsesFusedNormalRoPEDecoder(t *testing.T) {
	file := &gguf.File{Metadata: []gguf.Metadata{
		metadata("general.architecture", gguf.ValueTypeString, "cogvlm"),
		metadata("cogvlm.block_count", gguf.ValueTypeUint32, uint32(2)),
		metadata("cogvlm.context_length", gguf.ValueTypeUint32, uint32(2048)),
		metadata("cogvlm.embedding_length", gguf.ValueTypeUint32, uint32(8)),
		metadata("cogvlm.feed_forward_length", gguf.ValueTypeUint32, uint32(12)),
		metadata("cogvlm.attention.head_count", gguf.ValueTypeUint32, uint32(2)),
		metadata("cogvlm.attention.head_count_kv", gguf.ValueTypeUint32, uint32(2)),
		metadata("cogvlm.attention.key_length", gguf.ValueTypeUint32, uint32(4)),
		metadata("cogvlm.attention.value_length", gguf.ValueTypeUint32, uint32(4)),
		metadata("cogvlm.attention.layer_norm_rms_epsilon", gguf.ValueTypeFloat32, float32(1e-6)),
		metadata("cogvlm.rope.freq_base", gguf.ValueTypeFloat32, float32(10000)),
		metadata("cogvlm.vocab_size", gguf.ValueTypeUint32, uint32(32)),
	}}
	spec, err := ReadSpec(file)
	if err != nil {
		t.Fatal(err)
	}
	if spec.Architecture != "cogvlm" || spec.RopeDimensionCount != 4 ||
		!usesNormalRoPE(spec.Architecture) {
		t.Fatalf("unexpected CogVLM spec: %+v", spec)
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

func TestReadHunyuanVLSpecUsesOptionalMRoPEAndXDRoPE(t *testing.T) {
	file := &gguf.File{Metadata: []gguf.Metadata{
		metadata("general.architecture", gguf.ValueTypeString, "hunyuan_vl"),
		metadata("hunyuan_vl.block_count", gguf.ValueTypeUint32, uint32(2)),
		metadata("hunyuan_vl.context_length", gguf.ValueTypeUint32, uint32(4096)),
		metadata("hunyuan_vl.embedding_length", gguf.ValueTypeUint32, uint32(8)),
		metadata("hunyuan_vl.feed_forward_length", gguf.ValueTypeUint32, uint32(12)),
		metadata("hunyuan_vl.attention.head_count", gguf.ValueTypeUint32, uint32(2)),
		metadata("hunyuan_vl.attention.head_count_kv", gguf.ValueTypeUint32, uint32(1)),
		metadata("hunyuan_vl.attention.key_length", gguf.ValueTypeUint32, uint32(4)),
		metadata("hunyuan_vl.attention.value_length", gguf.ValueTypeUint32, uint32(4)),
		metadata("hunyuan_vl.attention.layer_norm_rms_epsilon", gguf.ValueTypeFloat32, float32(1e-6)),
		metadata("hunyuan_vl.rope.freq_base", gguf.ValueTypeFloat32, float32(10000)),
		metadata("hunyuan_vl.rope.scaling.alpha", gguf.ValueTypeFloat32, float32(2)),
		metadata("hunyuan_vl.vocab_size", gguf.ValueTypeUint32, uint32(32)),
		{Key: "hunyuan_vl.rope.dimension_sections", Value: gguf.Value{
			Type: gguf.ValueTypeArray, ArrayType: gguf.ValueTypeInt32, Data: []int32{1, 1, 0, 0},
		}},
	}}
	spec, err := ReadSpec(file)
	if err != nil {
		t.Fatal(err)
	}
	if spec.Architecture != "hunyuan_vl" || spec.RopeDimensionCount != 4 ||
		spec.RopeFrequencyBase != 40000 || spec.RopeSections != [4]int32{1, 1, 0, 0} ||
		!usesNormalRoPE(spec.Architecture) {
		t.Fatalf("unexpected Hunyuan-VL spec: %+v", spec)
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

func TestReadDeepSeek2Spec(t *testing.T) {
	file := &gguf.File{Metadata: []gguf.Metadata{
		metadata("general.architecture", gguf.ValueTypeString, "deepseek2"),
		metadata("deepseek2.block_count", gguf.ValueTypeUint32, uint32(4)),
		metadata("deepseek2.context_length", gguf.ValueTypeUint32, uint32(16384)),
		metadata("deepseek2.embedding_length", gguf.ValueTypeUint32, uint32(8)),
		metadata("deepseek2.feed_forward_length", gguf.ValueTypeUint32, uint32(12)),
		metadata("deepseek2.vocab_size", gguf.ValueTypeUint32, uint32(32)),
		metadata("deepseek2.attention.head_count", gguf.ValueTypeUint32, uint32(2)),
		metadata("deepseek2.attention.head_count_kv", gguf.ValueTypeUint32, uint32(1)),
		metadata("deepseek2.attention.key_length_mla", gguf.ValueTypeUint32, uint32(6)),
		metadata("deepseek2.attention.value_length_mla", gguf.ValueTypeUint32, uint32(4)),
		metadata("deepseek2.attention.q_lora_rank", gguf.ValueTypeUint32, uint32(3)),
		metadata("deepseek2.attention.kv_lora_rank", gguf.ValueTypeUint32, uint32(3)),
		metadata("deepseek2.rope.dimension_count", gguf.ValueTypeUint32, uint32(2)),
		metadata("deepseek2.rope.freq_base", gguf.ValueTypeFloat32, float32(10000)),
		metadata("deepseek2.rope.scaling.type", gguf.ValueTypeString, "yarn"),
		metadata("deepseek2.rope.scaling.factor", gguf.ValueTypeFloat32, float32(4)),
		metadata("deepseek2.rope.scaling.original_context_length", gguf.ValueTypeUint32, uint32(4096)),
		metadata("deepseek2.rope.scaling.yarn_log_multiplier", gguf.ValueTypeFloat32, float32(0.1)),
		metadata("deepseek2.attention.layer_norm_rms_epsilon", gguf.ValueTypeFloat32, float32(1e-6)),
		metadata("deepseek2.attention.temperature_scale", gguf.ValueTypeFloat32, float32(0.1)),
		metadata("deepseek2.attention.temperature_length", gguf.ValueTypeUint32, uint32(8192)),
		metadata("deepseek2.leading_dense_block_count", gguf.ValueTypeUint32, uint32(1)),
		metadata("deepseek2.expert_count", gguf.ValueTypeUint32, uint32(4)),
		metadata("deepseek2.expert_used_count", gguf.ValueTypeUint32, uint32(2)),
		metadata("deepseek2.expert_feed_forward_length", gguf.ValueTypeUint32, uint32(6)),
		metadata("deepseek2.expert_shared_count", gguf.ValueTypeUint32, uint32(1)),
	}}
	spec, err := ReadSpec(file)
	if err != nil {
		t.Fatal(err)
	}
	if spec.QLoRARank != 3 || spec.KVLoRARank != 3 || spec.KeyLength != 6 || spec.ValueLength != 4 ||
		spec.RopeDimensionCount != 2 || spec.RopeScalingType != "yarn" ||
		spec.RopeYaRNLogMultiplier != 1 || spec.LeadingDenseBlocks != 1 ||
		spec.SharedExpertFF != 6 || spec.ExpertGatingFunc != 1 || spec.AttentionTempFloor != 8192 {
		t.Fatalf("unexpected DeepSeek2 spec: %+v", spec)
	}
	wantAttentionFactor := float32(1 / (1 + 0.1*math.Log(4)))
	if math.Abs(float64(spec.YaRNAttentionFactor-wantAttentionFactor)) > 1e-6 {
		t.Fatalf("DeepSeek2 YaRN attention factor = %v, want %v", spec.YaRNAttentionFactor, wantAttentionFactor)
	}
}

func TestReadGLMDSASpec(t *testing.T) {
	prefix := "glm-dsa."
	file := &gguf.File{Metadata: []gguf.Metadata{
		metadata("general.architecture", gguf.ValueTypeString, "glm-dsa"),
		metadata(prefix+"block_count", gguf.ValueTypeUint32, uint32(6)),
		metadata(prefix+"context_length", gguf.ValueTypeUint32, uint32(1048576)),
		metadata(prefix+"embedding_length", gguf.ValueTypeUint32, uint32(8)),
		metadata(prefix+"feed_forward_length", gguf.ValueTypeUint32, uint32(12)),
		metadata(prefix+"vocab_size", gguf.ValueTypeUint32, uint32(32)),
		metadata(prefix+"attention.head_count", gguf.ValueTypeUint32, uint32(2)),
		metadata(prefix+"attention.head_count_kv", gguf.ValueTypeUint32, uint32(1)),
		metadata(prefix+"attention.key_length_mla", gguf.ValueTypeUint32, uint32(6)),
		metadata(prefix+"attention.value_length_mla", gguf.ValueTypeUint32, uint32(4)),
		metadata(prefix+"attention.q_lora_rank", gguf.ValueTypeUint32, uint32(3)),
		metadata(prefix+"attention.kv_lora_rank", gguf.ValueTypeUint32, uint32(3)),
		metadata(prefix+"rope.dimension_count", gguf.ValueTypeUint32, uint32(2)),
		metadata(prefix+"rope.freq_base", gguf.ValueTypeFloat32, float32(10000)),
		metadata(prefix+"rope.scaling.type", gguf.ValueTypeString, "yarn"),
		metadata(prefix+"rope.scaling.factor", gguf.ValueTypeFloat32, float32(8)),
		metadata(prefix+"rope.scaling.original_context_length", gguf.ValueTypeUint32, uint32(131072)),
		metadata(prefix+"rope.scaling.yarn_log_multiplier", gguf.ValueTypeFloat32, float32(0.1)),
		metadata(prefix+"attention.layer_norm_rms_epsilon", gguf.ValueTypeFloat32, float32(1e-6)),
		metadata(prefix+"leading_dense_block_count", gguf.ValueTypeUint32, uint32(1)),
		metadata(prefix+"expert_count", gguf.ValueTypeUint32, uint32(4)),
		metadata(prefix+"expert_used_count", gguf.ValueTypeUint32, uint32(2)),
		metadata(prefix+"expert_feed_forward_length", gguf.ValueTypeUint32, uint32(6)),
		metadata(prefix+"expert_shared_count", gguf.ValueTypeUint32, uint32(1)),
		metadata(prefix+"attention.indexer.head_count", gguf.ValueTypeUint32, uint32(2)),
		metadata(prefix+"attention.indexer.key_length", gguf.ValueTypeUint32, uint32(8)),
		metadata(prefix+"attention.indexer.top_k", gguf.ValueTypeUint32, uint32(4)),
		{Key: prefix + "rope.dimension_sections", Value: gguf.Value{Type: gguf.ValueTypeArray, ArrayType: gguf.ValueTypeInt32, Data: []int32{1, 0, 0, 0}}},
		{Key: prefix + "attention.indexer.types", Value: gguf.Value{Type: gguf.ValueTypeArray, ArrayType: gguf.ValueTypeUint32, Data: []uint32{1, 1, 1, 0, 0, 0}}},
	}}
	spec, err := ReadSpec(file)
	if err != nil {
		t.Fatal(err)
	}
	if spec.Architecture != "glm-dsa" || spec.IndexerHeadCount != 2 || spec.IndexerKeyLength != 8 ||
		spec.IndexerTopK != 4 || spec.ExpertGatingFunc != 2 || spec.RopeScalingType != "yarn" ||
		spec.RopeYaRNLogMultiplier != 1 || !spec.LayerHasFullIndexer(2) ||
		spec.LayerHasFullIndexer(3) || spec.RopeSections != [4]int32{1, 0, 0, 0} {
		t.Fatalf("unexpected GLM-DSA spec: %+v", spec)
	}
	withoutSections := &gguf.File{}
	for _, item := range file.Metadata {
		if item.Key != prefix+"rope.dimension_sections" {
			withoutSections.Metadata = append(withoutSections.Metadata, item)
		}
	}
	if _, err := ReadSpec(withoutSections); err != nil {
		t.Fatalf("optional GLM-DSA RoPE sections: %v", err)
	}
}

func TestReadDeepSeek32Spec(t *testing.T) {
	prefix := "deepseek32."
	file := &gguf.File{Metadata: []gguf.Metadata{
		metadata("general.architecture", gguf.ValueTypeString, "deepseek32"),
		metadata(prefix+"block_count", gguf.ValueTypeUint32, uint32(63)),
		metadata(prefix+"nextn_predict_layers", gguf.ValueTypeUint32, uint32(1)),
		metadata(prefix+"context_length", gguf.ValueTypeUint32, uint32(163840)),
		metadata(prefix+"embedding_length", gguf.ValueTypeUint32, uint32(8)),
		metadata(prefix+"feed_forward_length", gguf.ValueTypeUint32, uint32(12)),
		metadata(prefix+"vocab_size", gguf.ValueTypeUint32, uint32(32)),
		metadata(prefix+"attention.head_count", gguf.ValueTypeUint32, uint32(2)),
		metadata(prefix+"attention.head_count_kv", gguf.ValueTypeUint32, uint32(1)),
		metadata(prefix+"attention.key_length_mla", gguf.ValueTypeUint32, uint32(6)),
		metadata(prefix+"attention.value_length_mla", gguf.ValueTypeUint32, uint32(4)),
		metadata(prefix+"attention.q_lora_rank", gguf.ValueTypeUint32, uint32(3)),
		metadata(prefix+"attention.kv_lora_rank", gguf.ValueTypeUint32, uint32(3)),
		metadata(prefix+"rope.dimension_count", gguf.ValueTypeUint32, uint32(2)),
		metadata(prefix+"rope.freq_base", gguf.ValueTypeFloat32, float32(10000)),
		metadata(prefix+"rope.scaling.type", gguf.ValueTypeString, "yarn"),
		metadata(prefix+"rope.scaling.factor", gguf.ValueTypeFloat32, float32(8)),
		metadata(prefix+"rope.scaling.original_context_length", gguf.ValueTypeUint32, uint32(4096)),
		metadata(prefix+"rope.scaling.yarn_log_multiplier", gguf.ValueTypeFloat32, float32(0.1)),
		metadata(prefix+"attention.layer_norm_rms_epsilon", gguf.ValueTypeFloat32, float32(1e-6)),
		metadata(prefix+"leading_dense_block_count", gguf.ValueTypeUint32, uint32(1)),
		metadata(prefix+"expert_count", gguf.ValueTypeUint32, uint32(4)),
		metadata(prefix+"expert_used_count", gguf.ValueTypeUint32, uint32(2)),
		metadata(prefix+"expert_feed_forward_length", gguf.ValueTypeUint32, uint32(6)),
		metadata(prefix+"expert_shared_count", gguf.ValueTypeUint32, uint32(1)),
		metadata(prefix+"expert_gating_func", gguf.ValueTypeUint32, uint32(2)),
		metadata(prefix+"attention.indexer.head_count", gguf.ValueTypeUint32, uint32(2)),
		metadata(prefix+"attention.indexer.key_length", gguf.ValueTypeUint32, uint32(8)),
		metadata(prefix+"attention.indexer.top_k", gguf.ValueTypeUint32, uint32(4)),
	}}
	spec, err := ReadSpec(file)
	if err != nil {
		t.Fatal(err)
	}
	if spec.Architecture != "deepseek32" || spec.BlockCount != 62 ||
		spec.IndexerHeadCount != 2 || spec.IndexerKeyLength != 8 || spec.IndexerTopK != 4 ||
		spec.ExpertGatingFunc != 2 || spec.LayerNormEpsilon != 1e-6 ||
		!spec.LayerHasFullIndexer(0) || !spec.LayerHasFullIndexer(61) {
		t.Fatalf("unexpected DeepSeek 3.2 spec: %+v", spec)
	}
}

func TestReadDeepSeek4Spec(t *testing.T) {
	prefix := "deepseek4."
	ratios := make([]uint32, 44)
	ratios[0], ratios[1], ratios[2] = 0, 4, 128
	file := &gguf.File{Metadata: []gguf.Metadata{
		metadata("general.architecture", gguf.ValueTypeString, "deepseek4"),
		metadata(prefix+"block_count", gguf.ValueTypeUint32, uint32(43)),
		metadata(prefix+"context_length", gguf.ValueTypeUint32, uint32(16384)),
		metadata(prefix+"embedding_length", gguf.ValueTypeUint32, uint32(8)),
		metadata(prefix+"feed_forward_length", gguf.ValueTypeUint32, uint32(12)),
		metadata(prefix+"vocab_size", gguf.ValueTypeUint32, uint32(32)),
		metadata(prefix+"attention.head_count", gguf.ValueTypeUint32, uint32(2)),
		metadata(prefix+"attention.head_count_kv", gguf.ValueTypeUint32, uint32(1)),
		metadata(prefix+"attention.key_length", gguf.ValueTypeUint32, uint32(4)),
		metadata(prefix+"attention.value_length", gguf.ValueTypeUint32, uint32(4)),
		metadata(prefix+"attention.q_lora_rank", gguf.ValueTypeUint32, uint32(3)),
		metadata(prefix+"attention.sliding_window", gguf.ValueTypeUint32, uint32(128)),
		metadata(prefix+"attention.indexer.head_count", gguf.ValueTypeUint32, uint32(2)),
		metadata(prefix+"attention.indexer.key_length", gguf.ValueTypeUint32, uint32(8)),
		metadata(prefix+"attention.indexer.top_k", gguf.ValueTypeUint32, uint32(4)),
		metadata(prefix+"attention.output_group_count", gguf.ValueTypeUint32, uint32(1)),
		metadata(prefix+"attention.output_lora_rank", gguf.ValueTypeUint32, uint32(3)),
		metadata(prefix+"attention.compress_rope_freq_base", gguf.ValueTypeFloat32, float32(10000)),
		metadata(prefix+"hyper_connection.count", gguf.ValueTypeUint32, uint32(4)),
		metadata(prefix+"hyper_connection.sinkhorn_iterations", gguf.ValueTypeUint32, uint32(2)),
		metadata(prefix+"hyper_connection.epsilon", gguf.ValueTypeFloat32, float32(1e-6)),
		metadata(prefix+"hash_layer_count", gguf.ValueTypeUint32, uint32(1)),
		metadata(prefix+"expert_count", gguf.ValueTypeUint32, uint32(4)),
		metadata(prefix+"expert_used_count", gguf.ValueTypeUint32, uint32(2)),
		metadata(prefix+"expert_feed_forward_length", gguf.ValueTypeUint32, uint32(8)),
		metadata(prefix+"expert_shared_count", gguf.ValueTypeUint32, uint32(1)),
		metadata(prefix+"expert_gating_func", gguf.ValueTypeUint32, uint32(4)),
		metadata(prefix+"expert_weights_scale", gguf.ValueTypeFloat32, float32(1.25)),
		metadata(prefix+"expert_weights_norm", gguf.ValueTypeBool, true),
		metadata(prefix+"swiglu_clamp_exp", gguf.ValueTypeFloat32, float32(7)),
		metadata(prefix+"swiglu_clamp_shexp", gguf.ValueTypeFloat32, float32(6)),
		metadata(prefix+"rope.dimension_count", gguf.ValueTypeUint32, uint32(2)),
		metadata(prefix+"rope.freq_base", gguf.ValueTypeFloat32, float32(10000)),
		metadata(prefix+"rope.scaling.type", gguf.ValueTypeString, "yarn"),
		metadata(prefix+"rope.scaling.factor", gguf.ValueTypeFloat32, float32(4)),
		metadata(prefix+"rope.scaling.original_context_length", gguf.ValueTypeUint32, uint32(4096)),
		metadata(prefix+"attention.layer_norm_rms_epsilon", gguf.ValueTypeFloat32, float32(1e-6)),
		{Key: prefix + "attention.compress_ratios", Value: gguf.Value{Type: gguf.ValueTypeArray, ArrayType: gguf.ValueTypeUint32, Data: ratios}},
	}}
	spec, err := ReadSpec(file)
	if err != nil {
		t.Fatal(err)
	}
	if spec.Architecture != "deepseek4" || spec.BlockCount != 43 || spec.QLoRARank != 3 ||
		len(spec.CompressRatios) != 43 || spec.CompressRatios[1] != 4 || spec.CompressRatios[2] != 128 || spec.HyperConnectionCount != 4 ||
		spec.AttentionOutputGroups != 1 || spec.AttentionOutputRank != 3 || spec.HashLayerCount != 1 ||
		spec.ExpertGatingFunc != 4 || spec.SharedExpertFF != 8 || spec.LayerExpertSwiGLUClamp(0) != 7 ||
		spec.LayerSharedSwiGLUClampLimit(0) != 6 || !usesNormalRoPE(spec.Architecture) {
		t.Fatalf("unexpected DeepSeek 4 spec: %+v", spec)
	}
}

func TestReadMistral4Spec(t *testing.T) {
	prefix := "mistral4."
	file := &gguf.File{Metadata: []gguf.Metadata{
		metadata("general.architecture", gguf.ValueTypeString, "mistral4"),
		metadata(prefix+"block_count", gguf.ValueTypeUint32, uint32(4)),
		metadata(prefix+"context_length", gguf.ValueTypeUint32, uint32(16384)),
		metadata(prefix+"embedding_length", gguf.ValueTypeUint32, uint32(8)),
		metadata(prefix+"feed_forward_length", gguf.ValueTypeUint32, uint32(12)),
		metadata(prefix+"vocab_size", gguf.ValueTypeUint32, uint32(32)),
		metadata(prefix+"attention.head_count", gguf.ValueTypeUint32, uint32(2)),
		metadata(prefix+"attention.head_count_kv", gguf.ValueTypeUint32, uint32(1)),
		metadata(prefix+"attention.key_length_mla", gguf.ValueTypeUint32, uint32(6)),
		metadata(prefix+"attention.value_length_mla", gguf.ValueTypeUint32, uint32(4)),
		metadata(prefix+"attention.q_lora_rank", gguf.ValueTypeUint32, uint32(3)),
		metadata(prefix+"attention.kv_lora_rank", gguf.ValueTypeUint32, uint32(3)),
		metadata(prefix+"rope.dimension_count", gguf.ValueTypeUint32, uint32(2)),
		metadata(prefix+"rope.freq_base", gguf.ValueTypeFloat32, float32(10000)),
		metadata(prefix+"attention.layer_norm_rms_epsilon", gguf.ValueTypeFloat32, float32(1e-6)),
		metadata(prefix+"leading_dense_block_count", gguf.ValueTypeUint32, uint32(1)),
		metadata(prefix+"expert_count", gguf.ValueTypeUint32, uint32(4)),
		metadata(prefix+"expert_used_count", gguf.ValueTypeUint32, uint32(2)),
		metadata(prefix+"expert_feed_forward_length", gguf.ValueTypeUint32, uint32(6)),
		metadata(prefix+"expert_shared_count", gguf.ValueTypeUint32, uint32(1)),
		metadata(prefix+"attention.temperature_scale", gguf.ValueTypeFloat32, float32(0.1)),
		metadata(prefix+"attention.temperature_length", gguf.ValueTypeUint32, uint32(8192)),
	}}
	spec, err := ReadSpec(file)
	if err != nil {
		t.Fatal(err)
	}
	if spec.Architecture != "mistral4" || spec.QLoRARank != 3 || spec.KVLoRARank != 3 ||
		spec.KeyLength != 6 || spec.ValueLength != 4 || spec.LeadingDenseBlocks != 1 ||
		spec.ExpertCount != 4 || spec.ExpertUsedCount != 2 || spec.SharedExpertFF != 6 ||
		spec.AttentionTempFloor != 8192 {
		t.Fatalf("unexpected Mistral 4 spec: %+v", spec)
	}
}

func TestReadDenseDeepSeek2LiteSpec(t *testing.T) {
	file := &gguf.File{Metadata: []gguf.Metadata{
		metadata("general.architecture", gguf.ValueTypeString, "deepseek2"),
		metadata("deepseek2.block_count", gguf.ValueTypeUint32, uint32(27)),
		metadata("deepseek2.context_length", gguf.ValueTypeUint32, uint32(4096)),
		metadata("deepseek2.embedding_length", gguf.ValueTypeUint32, uint32(8)),
		metadata("deepseek2.feed_forward_length", gguf.ValueTypeUint32, uint32(12)),
		metadata("deepseek2.vocab_size", gguf.ValueTypeUint32, uint32(32)),
		metadata("deepseek2.attention.head_count", gguf.ValueTypeUint32, uint32(2)),
		metadata("deepseek2.attention.head_count_kv", gguf.ValueTypeUint32, uint32(1)),
		metadata("deepseek2.attention.key_length_mla", gguf.ValueTypeUint32, uint32(6)),
		metadata("deepseek2.attention.value_length_mla", gguf.ValueTypeUint32, uint32(4)),
		metadata("deepseek2.attention.kv_lora_rank", gguf.ValueTypeUint32, uint32(3)),
		metadata("deepseek2.rope.dimension_count", gguf.ValueTypeUint32, uint32(2)),
		metadata("deepseek2.rope.freq_base", gguf.ValueTypeFloat32, float32(10000)),
		metadata("deepseek2.attention.layer_norm_rms_epsilon", gguf.ValueTypeFloat32, float32(1e-6)),
		metadata("deepseek2.leading_dense_block_count", gguf.ValueTypeUint32, uint32(27)),
		metadata("deepseek2.expert_feed_forward_length", gguf.ValueTypeUint32, uint32(12)),
		metadata("deepseek2.expert_shared_count", gguf.ValueTypeUint32, uint32(0)),
	}}
	spec, err := ReadSpec(file)
	if err != nil {
		t.Fatal(err)
	}
	if spec.QLoRARank != 0 || spec.ExpertCount != 0 || spec.ExpertUsedCount != 0 ||
		spec.LeadingDenseBlocks != spec.BlockCount || spec.HeadCountKV != 1 {
		t.Fatalf("unexpected dense DeepSeek2 spec: %+v", spec)
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
				if test.architecture == "mpt" {
					metadataItems = append(metadataItems, metadata(
						prefix+"attention.clamp_kqv", gguf.ValueTypeFloat32, float32(4),
					))
				}
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
			if test.architecture == "mpt" && spec.AttentionClamp != 4 {
				t.Fatalf("MPT clamp = %g, want 4", spec.AttentionClamp)
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

func TestReadBaichuanVariantsSpec(t *testing.T) {
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
	metadataValues = append(metadataValues[:7], metadataValues[8:]...)
	spec, err = ReadSpec(&gguf.File{Metadata: metadataValues})
	if err != nil {
		t.Fatal(err)
	}
	if spec.BlockCount != 40 || spec.UsesRoPE(0) || spec.MaxALiBiBias != 8 {
		t.Fatalf("unexpected Baichuan 13B spec: %+v", spec)
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
	clamped, err := ReadSpec(&gguf.File{Metadata: metadataValues})
	if err != nil {
		t.Fatal(err)
	}
	if clamped.AttentionClamp != 8 {
		t.Fatalf("OLMo clamp = %g, want 8", clamped.AttentionClamp)
	}
}

func TestReadMPTAndOLMoRejectInvalidClamp(t *testing.T) {
	for _, architecture := range []string{"mpt", "olmo"} {
		prefix := architecture + "."
		metadataItems := []gguf.Metadata{
			metadata("general.architecture", gguf.ValueTypeString, architecture),
			metadata(prefix+"block_count", gguf.ValueTypeUint32, uint32(2)),
			metadata(prefix+"context_length", gguf.ValueTypeUint32, uint32(128)),
			metadata(prefix+"embedding_length", gguf.ValueTypeUint32, uint32(16)),
			metadata(prefix+"feed_forward_length", gguf.ValueTypeUint32, uint32(64)),
			metadata(prefix+"attention.head_count", gguf.ValueTypeUint32, uint32(4)),
			metadata(prefix+"attention.head_count_kv", gguf.ValueTypeUint32, uint32(2)),
			metadata(prefix+"attention.layer_norm_epsilon", gguf.ValueTypeFloat32, float32(1e-5)),
			metadata(prefix+"attention.clamp_kqv", gguf.ValueTypeFloat32, float32(-1)),
			metadata(prefix+"vocab_size", gguf.ValueTypeUint32, uint32(32)),
		}
		if _, err := ReadSpec(&gguf.File{Metadata: metadataItems}); err == nil {
			t.Fatalf("negative %s clamp was accepted", architecture)
		}
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

func TestReadMistral3TemperatureAndMoESpec(t *testing.T) {
	file := &gguf.File{Metadata: append(mistral3Metadata(),
		metadata("mistral3.rope.scaling.original_context_length", gguf.ValueTypeUint32, uint32(8192)),
		metadata("mistral3.attention.temperature_scale", gguf.ValueTypeFloat32, float32(0.1)),
		metadata("mistral3.expert_count", gguf.ValueTypeUint32, uint32(8)),
		metadata("mistral3.expert_used_count", gguf.ValueTypeUint32, uint32(2)),
	)}
	spec, err := ReadSpec(file)
	if err != nil {
		t.Fatal(err)
	}
	if spec.AttentionTempScale != 0.1 || spec.AttentionTempFloor != 8192 ||
		spec.OriginalContextLength != 8192 || spec.ExpertCount != 8 ||
		spec.ExpertUsedCount != 2 || spec.ExpertFeedForward != spec.FeedForwardLength ||
		!spec.ExpertWeightsNorm || spec.ExpertWeightsScale != 1 {
		t.Fatalf("unexpected Mistral 3 variant spec: %+v", spec)
	}
}

func TestReadMistral3RejectsInvalidVariants(t *testing.T) {
	tests := [][]gguf.Metadata{
		{
			metadata("mistral3.attention.temperature_scale", gguf.ValueTypeFloat32, float32(-0.1)),
		},
		{
			metadata("mistral3.expert_count", gguf.ValueTypeUint32, uint32(8)),
		},
		{
			metadata("mistral3.expert_count", gguf.ValueTypeUint32, uint32(8)),
			metadata("mistral3.expert_used_count", gguf.ValueTypeUint32, uint32(9)),
		},
	}
	for index, extra := range tests {
		file := &gguf.File{Metadata: append(mistral3Metadata(), extra...)}
		if _, err := ReadSpec(file); err == nil {
			t.Fatalf("invalid Mistral 3 variant %d was accepted", index)
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

func TestReadLlamaEmbedSpecUsesBidirectionalLlamaExecution(t *testing.T) {
	file := &gguf.File{Metadata: []gguf.Metadata{
		metadata("general.architecture", gguf.ValueTypeString, "llama-embed"),
		metadata("llama-embed.block_count", gguf.ValueTypeUint32, uint32(2)),
		metadata("llama-embed.context_length", gguf.ValueTypeUint32, uint32(2048)),
		metadata("llama-embed.embedding_length", gguf.ValueTypeUint32, uint32(128)),
		metadata("llama-embed.feed_forward_length", gguf.ValueTypeUint32, uint32(256)),
		metadata("llama-embed.attention.head_count", gguf.ValueTypeUint32, uint32(4)),
		metadata("llama-embed.attention.head_count_kv", gguf.ValueTypeUint32, uint32(2)),
		metadata("llama-embed.rope.freq_base", gguf.ValueTypeFloat32, float32(10000)),
		metadata("llama-embed.attention.layer_norm_rms_epsilon", gguf.ValueTypeFloat32, float32(1e-5)),
		metadata("llama-embed.vocab_size", gguf.ValueTypeUint32, uint32(32)),
	}}
	spec, err := ReadSpec(file)
	if err != nil {
		t.Fatal(err)
	}
	if spec.Architecture != "llama-embed" || !spec.NonCausalAttention || !spec.IsEncoderOnly() ||
		!usesNormalRoPE(spec.Architecture) || spec.KeyLength != 32 || spec.ValueLength != 32 {
		t.Fatalf("unexpected Llama Embed spec: %+v", spec)
	}
}

func TestReadPanguEmbeddedSpecUsesCausalNeoXLongRoPE(t *testing.T) {
	file := &gguf.File{Metadata: []gguf.Metadata{
		metadata("general.architecture", gguf.ValueTypeString, "pangu-embedded"),
		metadata("pangu-embedded.block_count", gguf.ValueTypeUint32, uint32(26)),
		metadata("pangu-embedded.context_length", gguf.ValueTypeUint32, uint32(4096)),
		metadata("pangu-embedded.embedding_length", gguf.ValueTypeUint32, uint32(8)),
		metadata("pangu-embedded.feed_forward_length", gguf.ValueTypeUint32, uint32(16)),
		metadata("pangu-embedded.attention.head_count", gguf.ValueTypeUint32, uint32(2)),
		metadata("pangu-embedded.attention.head_count_kv", gguf.ValueTypeUint32, uint32(1)),
		metadata("pangu-embedded.rope.freq_base", gguf.ValueTypeFloat32, float32(10000)),
		metadata("pangu-embedded.rope.dimension_count", gguf.ValueTypeUint32, uint32(4)),
		metadata("pangu-embedded.rope.scaling.type", gguf.ValueTypeString, "longrope"),
		metadata("pangu-embedded.rope.scaling.original_context_length", gguf.ValueTypeUint32, uint32(2048)),
		metadata("pangu-embedded.rope.scaling.attn_factor", gguf.ValueTypeFloat32, float32(1.25)),
		metadata("pangu-embedded.attention.layer_norm_rms_epsilon", gguf.ValueTypeFloat32, float32(1e-6)),
		metadata("pangu-embedded.vocab_size", gguf.ValueTypeUint32, uint32(32)),
	}}
	spec, err := ReadSpec(file)
	if err != nil {
		t.Fatal(err)
	}
	if spec.Architecture != "pangu-embedded" || spec.NonCausalAttention || spec.IsEncoderOnly() ||
		usesNormalRoPE(spec.Architecture) || !supportsLongRoPE(spec.Architecture) ||
		spec.RopeDimensionCount != 4 || spec.OriginalContextLength != 2048 ||
		spec.RopeAttentionFactor != 1.25 {
		t.Fatalf("unexpected Pangu Embedded spec: %+v", spec)
	}
}

func TestReadModernBERTSpecUsesDenseFirstSymmetricWindows(t *testing.T) {
	file := &gguf.File{Metadata: []gguf.Metadata{
		metadata("general.architecture", gguf.ValueTypeString, "modern-bert"),
		metadata("modern-bert.block_count", gguf.ValueTypeUint32, uint32(22)),
		metadata("modern-bert.context_length", gguf.ValueTypeUint32, uint32(8192)),
		metadata("modern-bert.embedding_length", gguf.ValueTypeUint32, uint32(8)),
		metadata("modern-bert.feed_forward_length", gguf.ValueTypeUint32, uint32(16)),
		metadata("modern-bert.attention.head_count", gguf.ValueTypeUint32, uint32(2)),
		metadata("modern-bert.rope.freq_base", gguf.ValueTypeFloat32, float32(10000)),
		metadata("modern-bert.rope.freq_base_swa", gguf.ValueTypeFloat32, float32(50000)),
		metadata("modern-bert.attention.sliding_window", gguf.ValueTypeUint32, uint32(128)),
		metadata("modern-bert.attention.sliding_window_pattern", gguf.ValueTypeUint32, uint32(3)),
		metadata("modern-bert.attention.layer_norm_epsilon", gguf.ValueTypeFloat32, float32(1e-5)),
		metadata("modern-bert.hidden_act", gguf.ValueTypeString, "silu"),
		metadata("modern-bert.vocab_size", gguf.ValueTypeUint32, uint32(32)),
	}}
	spec, err := ReadSpec(file)
	if err != nil {
		t.Fatal(err)
	}
	if spec.Architecture != "modern-bert" || !spec.NonCausalAttention || !spec.IsEncoderOnly() ||
		!spec.UsesWeightOnlyLayerNorm() || spec.HeadCountKV != 2 || spec.RopeDimensionCount != 4 ||
		spec.HiddenActivation != "silu" || spec.IsSlidingLayer(0) || !spec.IsSlidingLayer(1) ||
		!spec.IsSlidingLayer(2) || spec.IsSlidingLayer(3) {
		t.Fatalf("unexpected ModernBERT spec: %+v", spec)
	}
}

func TestReadGemmaEmbeddingSpecUsesPeriodicSymmetricWindows(t *testing.T) {
	file := &gguf.File{Metadata: []gguf.Metadata{
		metadata("general.architecture", gguf.ValueTypeString, "gemma-embedding"),
		metadata("gemma-embedding.block_count", gguf.ValueTypeUint32, uint32(24)),
		metadata("gemma-embedding.context_length", gguf.ValueTypeUint32, uint32(2048)),
		metadata("gemma-embedding.embedding_length", gguf.ValueTypeUint32, uint32(8)),
		metadata("gemma-embedding.feed_forward_length", gguf.ValueTypeUint32, uint32(16)),
		metadata("gemma-embedding.attention.head_count", gguf.ValueTypeUint32, uint32(2)),
		metadata("gemma-embedding.attention.head_count_kv", gguf.ValueTypeUint32, uint32(1)),
		metadata("gemma-embedding.attention.sliding_window", gguf.ValueTypeUint32, uint32(128)),
		metadata("gemma-embedding.attention.layer_norm_rms_epsilon", gguf.ValueTypeFloat32, float32(1e-6)),
		metadata("gemma-embedding.dense_2_feat_in", gguf.ValueTypeUint32, uint32(8)),
		metadata("gemma-embedding.dense_2_feat_out", gguf.ValueTypeUint32, uint32(6)),
		metadata("gemma-embedding.dense_3_feat_in", gguf.ValueTypeUint32, uint32(6)),
		metadata("gemma-embedding.dense_3_feat_out", gguf.ValueTypeUint32, uint32(8)),
		metadata("gemma-embedding.vocab_size", gguf.ValueTypeUint32, uint32(32)),
	}}
	spec, err := ReadSpec(file)
	if err != nil {
		t.Fatal(err)
	}
	if spec.Architecture != "gemma-embedding" || !spec.NonCausalAttention || !spec.IsEncoderOnly() ||
		spec.HeadCountKV != 1 || spec.RopeFrequencyBase != 10000 || spec.RopeFrequencySWA != 10000 ||
		spec.RopeDimensionCount != 4 || spec.SlidingPattern != 6 || !spec.IsSlidingLayer(0) ||
		!spec.IsSlidingLayer(4) || spec.IsSlidingLayer(5) || spec.InputEmbeddingScale() != float32(math.Sqrt(8)) ||
		spec.Dense2FeatureOut != 6 || spec.Dense3FeatureIn != 6 {
		t.Fatalf("unexpected Gemma embedding spec: %+v", spec)
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
		metadata("qwen35.block_count", gguf.ValueTypeUint32, uint32(33)),
		metadata("qwen35.nextn_predict_layers", gguf.ValueTypeUint32, uint32(1)),
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
	if spec.Architecture != "qwen35" || spec.BlockCount != 32 || spec.NextNPredictLayers != 1 || spec.SSMStateSize != 128 ||
		spec.FullAttentionInterval != 4 ||
		spec.RopeSections != [4]int32{11, 11, 10, 0} ||
		!spec.IsRecurrentLayer(0) || spec.IsRecurrentLayer(3) {
		t.Fatalf("unexpected Qwen3.5 spec: %+v", spec)
	}
}

func TestReadKimiLinearSpec(t *testing.T) {
	file := &gguf.File{Metadata: []gguf.Metadata{
		metadata("general.architecture", gguf.ValueTypeString, "kimi-linear"),
		metadata("kimi-linear.block_count", gguf.ValueTypeUint32, uint32(2)),
		metadata("kimi-linear.context_length", gguf.ValueTypeUint32, uint32(4096)),
		metadata("kimi-linear.embedding_length", gguf.ValueTypeUint32, uint32(8)),
		metadata("kimi-linear.feed_forward_length", gguf.ValueTypeUint32, uint32(12)),
		metadata("kimi-linear.attention.head_count", gguf.ValueTypeUint32, uint32(2)),
		{Key: "kimi-linear.attention.head_count_kv", Value: gguf.Value{Type: gguf.ValueTypeArray, ArrayType: gguf.ValueTypeUint32, Data: []uint32{0, 1}}},
		metadata("kimi-linear.attention.key_length_mla", gguf.ValueTypeUint32, uint32(4)),
		metadata("kimi-linear.attention.value_length_mla", gguf.ValueTypeUint32, uint32(2)),
		metadata("kimi-linear.attention.kv_lora_rank", gguf.ValueTypeUint32, uint32(3)),
		metadata("kimi-linear.rope.dimension_count", gguf.ValueTypeUint32, uint32(2)),
		metadata("kimi-linear.attention.layer_norm_rms_epsilon", gguf.ValueTypeFloat32, float32(1e-6)),
		metadata("kimi-linear.ssm.conv_kernel", gguf.ValueTypeUint32, uint32(3)),
		metadata("kimi-linear.kda.head_dim", gguf.ValueTypeUint32, uint32(2)),
		metadata("kimi-linear.expert_count", gguf.ValueTypeUint32, uint32(4)),
		metadata("kimi-linear.expert_used_count", gguf.ValueTypeUint32, uint32(2)),
		metadata("kimi-linear.expert_feed_forward_length", gguf.ValueTypeUint32, uint32(6)),
		metadata("kimi-linear.expert_shared_count", gguf.ValueTypeUint32, uint32(1)),
		metadata("kimi-linear.leading_dense_block_count", gguf.ValueTypeUint32, uint32(1)),
		metadata("kimi-linear.expert_weights_scale", gguf.ValueTypeFloat32, float32(2.446)),
		metadata("kimi-linear.expert_gating_func", gguf.ValueTypeUint32, uint32(2)),
	}}
	spec, err := ReadSpec(file)
	if err != nil {
		t.Fatal(err)
	}
	if spec.Architecture != "kimi-linear" || !spec.RopeDisabled ||
		!spec.IsRecurrentLayer(0) || spec.IsRecurrentLayer(1) ||
		spec.KDAHeadDim != 2 || spec.SSMInnerSize != 4 ||
		spec.KVLoRARank != 3 || spec.SharedExpertFF != 6 ||
		!spec.ExpertWeightsNorm || spec.ExpertWeightsScale != 2.446 {
		t.Fatalf("unexpected Kimi Linear spec: %+v", spec)
	}
}

func TestReadRWKV6Qwen2Spec(t *testing.T) {
	prefix := "rwkv6qwen2."
	file := &gguf.File{Metadata: []gguf.Metadata{
		metadata("general.architecture", gguf.ValueTypeString, "rwkv6qwen2"),
		metadata(prefix+"block_count", gguf.ValueTypeUint32, uint32(2)),
		metadata(prefix+"context_length", gguf.ValueTypeUint32, uint32(4096)),
		metadata(prefix+"embedding_length", gguf.ValueTypeUint32, uint32(8)),
		metadata(prefix+"feed_forward_length", gguf.ValueTypeUint32, uint32(12)),
		metadata(prefix+"vocab_size", gguf.ValueTypeUint32, uint32(32)),
		metadata(prefix+"attention.head_count", gguf.ValueTypeUint32, uint32(2)),
		metadata(prefix+"attention.head_count_kv", gguf.ValueTypeUint32, uint32(1)),
		metadata(prefix+"attention.layer_norm_rms_epsilon", gguf.ValueTypeFloat32, float32(1e-6)),
		metadata(prefix+"wkv.head_size", gguf.ValueTypeUint32, uint32(4)),
		metadata(prefix+"time_mix_extra_dim", gguf.ValueTypeUint32, uint32(3)),
		metadata(prefix+"time_decay_extra_dim", gguf.ValueTypeUint32, uint32(2)),
		metadata(prefix+"rescale_every_n_layers", gguf.ValueTypeUint32, uint32(2)),
		metadata(prefix+"token_shift_count", gguf.ValueTypeUint32, uint32(1)),
	}}
	spec, err := ReadSpec(file)
	if err != nil {
		t.Fatal(err)
	}
	if spec.Architecture != "rwkv6qwen2" || !spec.RopeDisabled || spec.WKVHeadSize != 4 ||
		spec.TimeMixExtraDim != 3 || spec.TimeDecayExtraDim != 2 || spec.RescaleEvery != 2 ||
		spec.TokenShiftCount != 1 || spec.KeyLength != 4 || spec.ValueLength != 4 {
		t.Fatalf("unexpected RWKV6-Qwen2 spec: %+v", spec)
	}
}

func TestReadRWKV6Spec(t *testing.T) {
	prefix := "rwkv6."
	file := &gguf.File{Metadata: []gguf.Metadata{
		metadata("general.architecture", gguf.ValueTypeString, "rwkv6"),
		metadata(prefix+"block_count", gguf.ValueTypeUint32, uint32(2)),
		metadata(prefix+"context_length", gguf.ValueTypeUint32, uint32(4096)),
		metadata(prefix+"embedding_length", gguf.ValueTypeUint32, uint32(8)),
		metadata(prefix+"feed_forward_length", gguf.ValueTypeUint32, uint32(12)),
		metadata(prefix+"vocab_size", gguf.ValueTypeUint32, uint32(32)),
		metadata(prefix+"attention.head_count", gguf.ValueTypeUint32, uint32(2)),
		metadata(prefix+"attention.head_count_kv", gguf.ValueTypeUint32, uint32(2)),
		metadata(prefix+"attention.layer_norm_epsilon", gguf.ValueTypeFloat32, float32(1e-5)),
		metadata(prefix+"wkv.head_size", gguf.ValueTypeUint32, uint32(4)),
		metadata(prefix+"time_mix_extra_dim", gguf.ValueTypeUint32, uint32(3)),
		metadata(prefix+"time_decay_extra_dim", gguf.ValueTypeUint32, uint32(2)),
		metadata(prefix+"rescale_every_n_layers", gguf.ValueTypeUint32, uint32(2)),
		metadata(prefix+"token_shift_count", gguf.ValueTypeUint32, uint32(2)),
	}}
	spec, err := ReadSpec(file)
	if err != nil {
		t.Fatal(err)
	}
	if spec.Architecture != "rwkv6" || !spec.RopeDisabled || !spec.UsesLayerNorm() ||
		spec.WKVHeadSize != 4 || spec.TimeMixExtraDim != 3 || spec.TimeDecayExtraDim != 2 ||
		spec.RescaleEvery != 2 || spec.TokenShiftCount != 2 || spec.KeyLength != 4 || spec.ValueLength != 4 {
		t.Fatalf("unexpected RWKV6 spec: %+v", spec)
	}
}

func TestReadRWKV7FamilySpec(t *testing.T) {
	for _, test := range []struct {
		architecture string
		shiftCount   uint32
		gateRank     uint32
		layerNorm    bool
	}{{"rwkv7", 2, 2, true}, {"arwkv7", 1, 0, false}} {
		t.Run(test.architecture, func(t *testing.T) {
			prefix := test.architecture + "."
			items := []gguf.Metadata{
				metadata("general.architecture", gguf.ValueTypeString, test.architecture),
				metadata(prefix+"block_count", gguf.ValueTypeUint32, uint32(2)),
				metadata(prefix+"context_length", gguf.ValueTypeUint32, uint32(4096)),
				metadata(prefix+"embedding_length", gguf.ValueTypeUint32, uint32(8)),
				metadata(prefix+"feed_forward_length", gguf.ValueTypeUint32, uint32(12)),
				metadata(prefix+"vocab_size", gguf.ValueTypeUint32, uint32(32)),
				metadata(prefix+"attention.head_count", gguf.ValueTypeUint32, uint32(2)),
				metadata(prefix+"attention.head_count_kv", gguf.ValueTypeUint32, uint32(2)),
				metadata(prefix+"wkv.head_size", gguf.ValueTypeUint32, uint32(4)),
				metadata(prefix+"attention.decay_lora_rank", gguf.ValueTypeUint32, uint32(3)),
				metadata(prefix+"attention.iclr_lora_rank", gguf.ValueTypeUint32, uint32(2)),
				metadata(prefix+"attention.value_residual_mix_lora_rank", gguf.ValueTypeUint32, uint32(3)),
			}
			if test.layerNorm {
				items = append(items,
					metadata(prefix+"attention.layer_norm_epsilon", gguf.ValueTypeFloat32, float32(1e-5)),
					metadata(prefix+"attention.gate_lora_rank", gguf.ValueTypeUint32, test.gateRank),
				)
			} else {
				items = append(items, metadata(prefix+"attention.layer_norm_rms_epsilon", gguf.ValueTypeFloat32, float32(1e-6)))
			}
			spec, err := ReadSpec(&gguf.File{Metadata: items})
			if err != nil {
				t.Fatal(err)
			}
			if spec.Architecture != test.architecture || !spec.RopeDisabled ||
				spec.WKVHeadSize != 4 || spec.DecayLoRARank != 3 || spec.ICLRLoRARank != 2 ||
				spec.ValueMixLoRARank != 3 || spec.GateLoRARank != test.gateRank ||
				spec.TokenShiftCount != test.shiftCount || spec.UsesLayerNorm() != test.layerNorm {
				t.Fatalf("unexpected %s spec: %+v", test.architecture, spec)
			}
		})
	}
}

func TestReadQwen3NextSpec(t *testing.T) {
	file := &gguf.File{Metadata: []gguf.Metadata{
		metadata("general.architecture", gguf.ValueTypeString, "qwen3next"),
		metadata("qwen3next.block_count", gguf.ValueTypeUint32, uint32(48)),
		metadata("qwen3next.context_length", gguf.ValueTypeUint32, uint32(262144)),
		metadata("qwen3next.embedding_length", gguf.ValueTypeUint32, uint32(8)),
		metadata("qwen3next.feed_forward_length", gguf.ValueTypeUint32, uint32(12)),
		metadata("qwen3next.attention.head_count", gguf.ValueTypeUint32, uint32(2)),
		metadata("qwen3next.attention.head_count_kv", gguf.ValueTypeUint32, uint32(1)),
		metadata("qwen3next.attention.key_length", gguf.ValueTypeUint32, uint32(4)),
		metadata("qwen3next.attention.value_length", gguf.ValueTypeUint32, uint32(4)),
		metadata("qwen3next.rope.freq_base", gguf.ValueTypeFloat32, float32(10_000_000)),
		metadata("qwen3next.rope.dimension_count", gguf.ValueTypeUint32, uint32(4)),
		metadata("qwen3next.attention.layer_norm_rms_epsilon", gguf.ValueTypeFloat32, float32(1e-6)),
		metadata("qwen3next.ssm.conv_kernel", gguf.ValueTypeUint32, uint32(3)),
		metadata("qwen3next.ssm.inner_size", gguf.ValueTypeUint32, uint32(4)),
		metadata("qwen3next.ssm.state_size", gguf.ValueTypeUint32, uint32(2)),
		metadata("qwen3next.ssm.time_step_rank", gguf.ValueTypeUint32, uint32(2)),
		metadata("qwen3next.ssm.group_count", gguf.ValueTypeUint32, uint32(1)),
		metadata("qwen3next.expert_count", gguf.ValueTypeUint32, uint32(4)),
		metadata("qwen3next.expert_used_count", gguf.ValueTypeUint32, uint32(2)),
		metadata("qwen3next.expert_feed_forward_length", gguf.ValueTypeUint32, uint32(6)),
		metadata("qwen3next.expert_shared_feed_forward_length", gguf.ValueTypeUint32, uint32(10)),
		metadata("qwen3next.expert_weights_scale", gguf.ValueTypeFloat32, float32(1.25)),
	}}
	spec, err := ReadSpec(file)
	if err != nil {
		t.Fatal(err)
	}
	if spec.Architecture != "qwen3next" || spec.BlockCount != 48 ||
		spec.RopeDimensionCount != 4 || spec.ExpertCount != 4 ||
		spec.ExpertUsedCount != 2 || spec.ExpertFeedForward != 6 ||
		spec.SharedExpertFF != 10 || spec.ExpertWeightsScale != 1.25 ||
		!spec.IsRecurrentLayer(0) || spec.IsRecurrentLayer(3) {
		t.Fatalf("unexpected Qwen3-Next spec: %+v", spec)
	}
}

func TestReadQwen35MoESpec(t *testing.T) {
	file := &gguf.File{Metadata: []gguf.Metadata{
		metadata("general.architecture", gguf.ValueTypeString, "qwen35moe"),
		metadata("qwen35moe.block_count", gguf.ValueTypeUint32, uint32(4)),
		metadata("qwen35moe.context_length", gguf.ValueTypeUint32, uint32(1024)),
		metadata("qwen35moe.embedding_length", gguf.ValueTypeUint32, uint32(8)),
		metadata("qwen35moe.feed_forward_length", gguf.ValueTypeUint32, uint32(12)),
		metadata("qwen35moe.attention.head_count", gguf.ValueTypeUint32, uint32(2)),
		metadata("qwen35moe.attention.head_count_kv", gguf.ValueTypeUint32, uint32(1)),
		metadata("qwen35moe.attention.key_length", gguf.ValueTypeUint32, uint32(4)),
		metadata("qwen35moe.attention.value_length", gguf.ValueTypeUint32, uint32(4)),
		metadata("qwen35moe.rope.freq_base", gguf.ValueTypeFloat32, float32(10000)),
		metadata("qwen35moe.rope.dimension_count", gguf.ValueTypeUint32, uint32(4)),
		metadata("qwen35moe.attention.layer_norm_rms_epsilon", gguf.ValueTypeFloat32, float32(1e-6)),
		metadata("qwen35moe.ssm.conv_kernel", gguf.ValueTypeUint32, uint32(3)),
		metadata("qwen35moe.ssm.inner_size", gguf.ValueTypeUint32, uint32(4)),
		metadata("qwen35moe.ssm.state_size", gguf.ValueTypeUint32, uint32(2)),
		metadata("qwen35moe.ssm.time_step_rank", gguf.ValueTypeUint32, uint32(2)),
		metadata("qwen35moe.ssm.group_count", gguf.ValueTypeUint32, uint32(1)),
		metadata("qwen35moe.expert_count", gguf.ValueTypeUint32, uint32(4)),
		metadata("qwen35moe.expert_used_count", gguf.ValueTypeUint32, uint32(2)),
		metadata("qwen35moe.expert_feed_forward_length", gguf.ValueTypeUint32, uint32(6)),
		metadata("qwen35moe.expert_shared_feed_forward_length", gguf.ValueTypeUint32, uint32(10)),
		metadata("qwen35moe.expert_weights_scale", gguf.ValueTypeFloat32, float32(1.25)),
		{
			Key: "qwen35moe.rope.dimension_sections",
			Value: gguf.Value{
				Type: gguf.ValueTypeArray, ArrayType: gguf.ValueTypeInt32,
				Data: []int32{1, 1, 0, 0},
			},
		},
	}}
	spec, err := ReadSpec(file)
	if err != nil {
		t.Fatal(err)
	}
	if spec.Architecture != "qwen35moe" || spec.ExpertCount != 4 ||
		spec.ExpertUsedCount != 2 || spec.ExpertFeedForward != 6 ||
		spec.SharedExpertFF != 10 || spec.ExpertWeightsScale != 1.25 ||
		!spec.IsRecurrentLayer(0) || spec.IsRecurrentLayer(3) {
		t.Fatalf("unexpected Qwen3.5-MoE spec: %+v", spec)
	}
}

func TestReadQwen35ExplicitRecurrentLayers(t *testing.T) {
	file := &gguf.File{Metadata: []gguf.Metadata{
		metadata("general.architecture", gguf.ValueTypeString, "qwen35"),
		metadata("qwen35.block_count", gguf.ValueTypeUint32, uint32(3)),
		metadata("qwen35.nextn_predict_layers", gguf.ValueTypeUint32, uint32(1)),
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
				Data: []bool{false, true, false},
			},
		},
	}}
	spec, err := ReadSpec(file)
	if err != nil {
		t.Fatal(err)
	}
	if spec.BlockCount != 2 || spec.NextNPredictLayers != 1 ||
		spec.IsRecurrentLayer(0) || !spec.IsRecurrentLayer(1) {
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

func TestReadT5Spec(t *testing.T) {
	file := &gguf.File{Metadata: []gguf.Metadata{
		metadata("general.architecture", gguf.ValueTypeString, "t5"),
		metadata("t5.block_count", gguf.ValueTypeUint32, uint32(2)),
		metadata("t5.decoder_block_count", gguf.ValueTypeUint32, uint32(3)),
		metadata("t5.decoder_start_token_id", gguf.ValueTypeUint32, uint32(7)),
		metadata("t5.context_length", gguf.ValueTypeUint32, uint32(512)),
		metadata("t5.embedding_length", gguf.ValueTypeUint32, uint32(8)),
		metadata("t5.feed_forward_length", gguf.ValueTypeUint32, uint32(16)),
		metadata("t5.attention.head_count", gguf.ValueTypeUint32, uint32(2)),
		metadata("t5.attention.key_length", gguf.ValueTypeUint32, uint32(4)),
		metadata("t5.attention.value_length", gguf.ValueTypeUint32, uint32(4)),
		metadata("t5.attention.layer_norm_rms_epsilon", gguf.ValueTypeFloat32, float32(1e-6)),
		metadata("t5.attention.relative_buckets_count", gguf.ValueTypeUint32, uint32(32)),
	}}
	spec, err := ReadSpec(file)
	if err != nil {
		t.Fatal(err)
	}
	if spec.DecoderBlockCount != 3 || spec.DecoderStartTokenID != 7 ||
		spec.HeadCountKV != 2 || spec.RopeFrequencyBase != 0 {
		t.Fatalf("unexpected T5 spec: %+v", spec)
	}
}

func TestReadWavTokenizerDecoderSpec(t *testing.T) {
	file := &gguf.File{Metadata: []gguf.Metadata{
		metadata("general.architecture", gguf.ValueTypeString, "wavtokenizer-dec"),
		metadata("wavtokenizer-dec.block_count", gguf.ValueTypeUint32, uint32(6)),
		metadata("wavtokenizer-dec.context_length", gguf.ValueTypeUint32, uint32(128)),
		metadata("wavtokenizer-dec.embedding_length", gguf.ValueTypeUint32, uint32(16)),
		metadata("wavtokenizer-dec.features_length", gguf.ValueTypeUint32, uint32(8)),
		metadata("wavtokenizer-dec.feed_forward_length", gguf.ValueTypeUint32, uint32(24)),
		metadata("wavtokenizer-dec.posnet.embedding_length", gguf.ValueTypeUint32, uint32(12)),
		metadata("wavtokenizer-dec.posnet.block_count", gguf.ValueTypeUint32, uint32(6)),
		metadata("wavtokenizer-dec.convnext.embedding_length", gguf.ValueTypeUint32, uint32(12)),
		metadata("wavtokenizer-dec.convnext.block_count", gguf.ValueTypeUint32, uint32(4)),
		metadata("wavtokenizer-dec.attention.layer_norm_epsilon", gguf.ValueTypeFloat32, float32(1e-5)),
		metadata("wavtokenizer-dec.attention.group_norm_epsilon", gguf.ValueTypeFloat32, float32(1e-5)),
		metadata("wavtokenizer-dec.attention.group_norm_groups", gguf.ValueTypeUint32, uint32(3)),
	}}
	spec, err := ReadSpec(file)
	if err != nil {
		t.Fatal(err)
	}
	if spec.Architecture != "wavtokenizer-dec" || spec.EmbeddingLength != 8 ||
		spec.OutputEmbeddingLength != 16 || spec.PosNetEmbeddingLength != 12 ||
		spec.PosNetBlockCount != 6 || spec.ConvNextBlockCount != 4 ||
		spec.GroupNormGroups != 3 || !spec.NonCausalAttention || !spec.RopeDisabled ||
		spec.HeadCount != 1 || spec.HeadCountKV != 1 || spec.KeyLength != 12 || spec.ValueLength != 12 {
		t.Fatalf("unexpected WavTokenizer decoder spec: %+v", spec)
	}
}

func TestReadDFlashSpec(t *testing.T) {
	file := &gguf.File{Metadata: []gguf.Metadata{
		metadata("general.architecture", gguf.ValueTypeString, "dflash"),
		metadata("dflash.block_count", gguf.ValueTypeUint32, uint32(2)),
		metadata("dflash.context_length", gguf.ValueTypeUint32, uint32(128)),
		metadata("dflash.embedding_length", gguf.ValueTypeUint32, uint32(8)),
		metadata("dflash.feed_forward_length", gguf.ValueTypeUint32, uint32(16)),
		metadata("dflash.attention.head_count", gguf.ValueTypeUint32, uint32(2)),
		metadata("dflash.attention.head_count_kv", gguf.ValueTypeUint32, uint32(1)),
		metadata("dflash.rope.freq_base", gguf.ValueTypeFloat32, float32(10000)),
		metadata("dflash.attention.layer_norm_rms_epsilon", gguf.ValueTypeFloat32, float32(1e-6)),
		metadata("dflash.block_size", gguf.ValueTypeUint32, uint32(8)),
		{
			Key:   "dflash.target_layers",
			Value: gguf.Value{Type: gguf.ValueTypeArray, ArrayType: gguf.ValueTypeInt32, Data: []int32{2, 7}},
		},
	}}
	spec, err := ReadSpec(file)
	if err != nil {
		t.Fatal(err)
	}
	if spec.Architecture != "dflash" || !spec.NonCausalAttention || spec.DFlashBlockSize != 8 ||
		len(spec.TargetLayers) != 2 || spec.TargetLayers[1] != 7 || spec.RopeDimensionCount != 4 {
		t.Fatalf("unexpected DFlash spec: %+v", spec)
	}
}

func TestReadEagle3Spec(t *testing.T) {
	file := &gguf.File{Metadata: []gguf.Metadata{
		metadata("general.architecture", gguf.ValueTypeString, "eagle3"),
		metadata("eagle3.block_count", gguf.ValueTypeUint32, uint32(1)),
		metadata("eagle3.context_length", gguf.ValueTypeUint32, uint32(128)),
		metadata("eagle3.embedding_length", gguf.ValueTypeUint32, uint32(8)),
		metadata("eagle3.feed_forward_length", gguf.ValueTypeUint32, uint32(16)),
		metadata("eagle3.attention.head_count", gguf.ValueTypeUint32, uint32(2)),
		metadata("eagle3.attention.head_count_kv", gguf.ValueTypeUint32, uint32(1)),
		metadata("eagle3.rope.freq_base", gguf.ValueTypeFloat32, float32(10000)),
		metadata("eagle3.attention.layer_norm_rms_epsilon", gguf.ValueTypeFloat32, float32(1e-6)),
		metadata("eagle3.target_hidden_size", gguf.ValueTypeUint32, uint32(12)),
		metadata("eagle3.norm_before_residual", gguf.ValueTypeBool, true),
		{Key: "eagle3.target_layers", Value: gguf.Value{
			Type: gguf.ValueTypeArray, ArrayType: gguf.ValueTypeInt32, Data: []int32{2, 7, 11},
		}},
	}}
	spec, err := ReadSpec(file)
	if err != nil {
		t.Fatal(err)
	}
	if spec.Architecture != "eagle3" || spec.TargetHiddenSize != 12 ||
		len(spec.TargetLayers) != 3 || !spec.NormBeforeResidual || spec.RopeDimensionCount != 4 {
		t.Fatalf("unexpected Eagle3 spec: %+v", spec)
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

func TestReadQwen2VLSpecUsesMRoPESections(t *testing.T) {
	file := &gguf.File{Metadata: []gguf.Metadata{
		metadata("general.architecture", gguf.ValueTypeString, "qwen2vl"),
		metadata("qwen2vl.block_count", gguf.ValueTypeUint32, uint32(2)),
		metadata("qwen2vl.context_length", gguf.ValueTypeUint32, uint32(4096)),
		metadata("qwen2vl.embedding_length", gguf.ValueTypeUint32, uint32(8)),
		metadata("qwen2vl.feed_forward_length", gguf.ValueTypeUint32, uint32(12)),
		metadata("qwen2vl.attention.head_count", gguf.ValueTypeUint32, uint32(2)),
		metadata("qwen2vl.attention.head_count_kv", gguf.ValueTypeUint32, uint32(1)),
		metadata("qwen2vl.attention.key_length", gguf.ValueTypeUint32, uint32(4)),
		metadata("qwen2vl.attention.value_length", gguf.ValueTypeUint32, uint32(4)),
		metadata("qwen2vl.rope.freq_base", gguf.ValueTypeFloat32, float32(10000)),
		metadata("qwen2vl.rope.scaling.type", gguf.ValueTypeString, "linear"),
		metadata("qwen2vl.rope.scaling.factor", gguf.ValueTypeFloat32, float32(4)),
		metadata("qwen2vl.attention.layer_norm_rms_epsilon", gguf.ValueTypeFloat32, float32(1e-6)),
		metadata("qwen2vl.vocab_size", gguf.ValueTypeUint32, uint32(32)),
		{Key: "qwen2vl.rope.dimension_sections", Value: gguf.Value{
			Type: gguf.ValueTypeArray, ArrayType: gguf.ValueTypeInt32, Data: []int32{1, 1, 0, 0},
		}},
	}}
	spec, err := ReadSpec(file)
	if err != nil {
		t.Fatal(err)
	}
	if spec.Architecture != "qwen2vl" || spec.RopeDimensionCount != 4 ||
		spec.RopeSections != [4]int32{1, 1, 0, 0} || spec.RopeScalingFactor != 4 ||
		usesNormalRoPE(spec.Architecture) {
		t.Fatalf("unexpected Qwen2-VL spec: %+v", spec)
	}
}

func TestReadQwen3VLSpecUsesMRoPEAndDeepstackMetadata(t *testing.T) {
	file := &gguf.File{Metadata: []gguf.Metadata{
		metadata("general.architecture", gguf.ValueTypeString, "qwen3vl"),
		metadata("qwen3vl.block_count", gguf.ValueTypeUint32, uint32(4)),
		metadata("qwen3vl.context_length", gguf.ValueTypeUint32, uint32(4096)),
		metadata("qwen3vl.embedding_length", gguf.ValueTypeUint32, uint32(8)),
		metadata("qwen3vl.feed_forward_length", gguf.ValueTypeUint32, uint32(12)),
		metadata("qwen3vl.attention.head_count", gguf.ValueTypeUint32, uint32(2)),
		metadata("qwen3vl.attention.head_count_kv", gguf.ValueTypeUint32, uint32(1)),
		metadata("qwen3vl.attention.key_length", gguf.ValueTypeUint32, uint32(4)),
		metadata("qwen3vl.attention.value_length", gguf.ValueTypeUint32, uint32(4)),
		metadata("qwen3vl.rope.freq_base", gguf.ValueTypeFloat32, float32(10000)),
		metadata("qwen3vl.rope.scaling.type", gguf.ValueTypeString, "linear"),
		metadata("qwen3vl.rope.scaling.factor", gguf.ValueTypeFloat32, float32(4)),
		metadata("qwen3vl.attention.layer_norm_rms_epsilon", gguf.ValueTypeFloat32, float32(1e-6)),
		metadata("qwen3vl.n_deepstack_layers", gguf.ValueTypeUint32, uint32(3)),
		metadata("qwen3vl.vocab_size", gguf.ValueTypeUint32, uint32(32)),
		{Key: "qwen3vl.rope.dimension_sections", Value: gguf.Value{
			Type: gguf.ValueTypeArray, ArrayType: gguf.ValueTypeInt32, Data: []int32{1, 1, 0, 0},
		}},
	}}
	spec, err := ReadSpec(file)
	if err != nil {
		t.Fatal(err)
	}
	if spec.Architecture != "qwen3vl" || spec.RopeDimensionCount != 4 ||
		spec.RopeSections != [4]int32{1, 1, 0, 0} || spec.RopeScalingFactor != 4 ||
		spec.DeepstackLayerCount != 3 || usesNormalRoPE(spec.Architecture) {
		t.Fatalf("unexpected Qwen3-VL spec: %+v", spec)
	}
}

func TestReadQwen3VLMoESpecUsesMRoPEExpertsAndDeepstackMetadata(t *testing.T) {
	file := &gguf.File{Metadata: []gguf.Metadata{
		metadata("general.architecture", gguf.ValueTypeString, "qwen3vlmoe"),
		metadata("qwen3vlmoe.block_count", gguf.ValueTypeUint32, uint32(4)),
		metadata("qwen3vlmoe.context_length", gguf.ValueTypeUint32, uint32(4096)),
		metadata("qwen3vlmoe.embedding_length", gguf.ValueTypeUint32, uint32(8)),
		metadata("qwen3vlmoe.feed_forward_length", gguf.ValueTypeUint32, uint32(24)),
		metadata("qwen3vlmoe.expert_feed_forward_length", gguf.ValueTypeUint32, uint32(12)),
		metadata("qwen3vlmoe.expert_count", gguf.ValueTypeUint32, uint32(4)),
		metadata("qwen3vlmoe.expert_used_count", gguf.ValueTypeUint32, uint32(2)),
		metadata("qwen3vlmoe.expert_weights_scale", gguf.ValueTypeFloat32, float32(1.25)),
		metadata("qwen3vlmoe.attention.head_count", gguf.ValueTypeUint32, uint32(2)),
		metadata("qwen3vlmoe.attention.head_count_kv", gguf.ValueTypeUint32, uint32(1)),
		metadata("qwen3vlmoe.attention.key_length", gguf.ValueTypeUint32, uint32(4)),
		metadata("qwen3vlmoe.attention.value_length", gguf.ValueTypeUint32, uint32(4)),
		metadata("qwen3vlmoe.rope.freq_base", gguf.ValueTypeFloat32, float32(10000)),
		metadata("qwen3vlmoe.rope.scaling.type", gguf.ValueTypeString, "linear"),
		metadata("qwen3vlmoe.rope.scaling.factor", gguf.ValueTypeFloat32, float32(4)),
		metadata("qwen3vlmoe.attention.layer_norm_rms_epsilon", gguf.ValueTypeFloat32, float32(1e-6)),
		metadata("qwen3vlmoe.n_deepstack_layers", gguf.ValueTypeUint32, uint32(3)),
		metadata("qwen3vlmoe.vocab_size", gguf.ValueTypeUint32, uint32(32)),
		{Key: "qwen3vlmoe.rope.dimension_sections", Value: gguf.Value{
			Type: gguf.ValueTypeArray, ArrayType: gguf.ValueTypeInt32, Data: []int32{1, 1, 0, 0},
		}},
	}}
	spec, err := ReadSpec(file)
	if err != nil {
		t.Fatal(err)
	}
	if spec.Architecture != "qwen3vlmoe" || spec.RopeDimensionCount != 4 ||
		spec.RopeSections != [4]int32{1, 1, 0, 0} || spec.RopeScalingFactor != 4 ||
		spec.DeepstackLayerCount != 3 || spec.ExpertCount != 4 || spec.ExpertUsedCount != 2 ||
		spec.ExpertFeedForward != 12 || spec.ExpertWeightsScale != 1.25 || usesNormalRoPE(spec.Architecture) {
		t.Fatalf("unexpected Qwen3-VL-MoE spec: %+v", spec)
	}
}

func TestReadSpecRejectsUnsupportedArchitecture(t *testing.T) {
	for _, architecture := range []string{
		"unsupported-test", "gptj",
	} {
		t.Run(architecture, func(t *testing.T) {
			file := &gguf.File{Metadata: []gguf.Metadata{
				metadata("general.architecture", gguf.ValueTypeString, architecture),
			}}
			_, err := ReadSpec(file)
			var unsupported *UnsupportedArchitectureError
			if !errors.As(err, &unsupported) || unsupported.Architecture != architecture {
				t.Fatalf("error = %v, want UnsupportedArchitectureError for %q", err, architecture)
			}
		})
	}
}

func TestReadMambaSpec(t *testing.T) {
	file := &gguf.File{Metadata: []gguf.Metadata{
		metadata("general.architecture", gguf.ValueTypeString, "mamba"),
		metadata("mamba.block_count", gguf.ValueTypeUint32, uint32(2)),
		metadata("mamba.context_length", gguf.ValueTypeUint32, uint32(4096)),
		metadata("mamba.embedding_length", gguf.ValueTypeUint32, uint32(4)),
		metadata("mamba.feed_forward_length", gguf.ValueTypeUint32, uint32(0)),
		metadata("mamba.attention.head_count", gguf.ValueTypeUint32, uint32(0)),
		metadata("mamba.attention.layer_norm_rms_epsilon", gguf.ValueTypeFloat32, float32(1e-5)),
		metadata("mamba.ssm.conv_kernel", gguf.ValueTypeUint32, uint32(3)),
		metadata("mamba.ssm.inner_size", gguf.ValueTypeUint32, uint32(8)),
		metadata("mamba.ssm.state_size", gguf.ValueTypeUint32, uint32(2)),
		metadata("mamba.ssm.time_step_rank", gguf.ValueTypeUint32, uint32(2)),
		metadata("mamba.ssm.dt_b_c_rms", gguf.ValueTypeBool, true),
		metadata("mamba.vocab_size", gguf.ValueTypeUint32, uint32(32)),
	}}
	spec, err := ReadSpec(file)
	if err != nil {
		t.Fatal(err)
	}
	if spec.Architecture != "mamba" || !spec.RopeDisabled || spec.HeadCount != 0 || spec.HeadCountKV != 0 ||
		spec.FeedForwardLength != 0 || spec.SSMConvKernel != 3 || spec.SSMInnerSize != 8 ||
		spec.SSMStateSize != 2 || spec.SSMTimeStepRank != 2 || spec.SSMGroupCount != 1 || !spec.SSMDtBCNorm {
		t.Fatalf("unexpected Mamba spec: %+v", spec)
	}
}

func TestReadMamba2Spec(t *testing.T) {
	file := &gguf.File{Metadata: []gguf.Metadata{
		metadata("general.architecture", gguf.ValueTypeString, "mamba2"),
		metadata("mamba2.block_count", gguf.ValueTypeUint32, uint32(2)),
		metadata("mamba2.context_length", gguf.ValueTypeUint32, uint32(4096)),
		metadata("mamba2.embedding_length", gguf.ValueTypeUint32, uint32(4)),
		metadata("mamba2.feed_forward_length", gguf.ValueTypeUint32, uint32(0)),
		metadata("mamba2.attention.head_count", gguf.ValueTypeUint32, uint32(0)),
		metadata("mamba2.attention.layer_norm_rms_epsilon", gguf.ValueTypeFloat32, float32(1e-5)),
		metadata("mamba2.ssm.conv_kernel", gguf.ValueTypeUint32, uint32(3)),
		metadata("mamba2.ssm.inner_size", gguf.ValueTypeUint32, uint32(8)),
		metadata("mamba2.ssm.state_size", gguf.ValueTypeUint32, uint32(2)),
		metadata("mamba2.ssm.time_step_rank", gguf.ValueTypeUint32, uint32(4)),
		metadata("mamba2.ssm.group_count", gguf.ValueTypeUint32, uint32(2)),
		metadata("mamba2.vocab_size", gguf.ValueTypeUint32, uint32(32)),
	}}
	spec, err := ReadSpec(file)
	if err != nil {
		t.Fatal(err)
	}
	if spec.Architecture != "mamba2" || !spec.RopeDisabled || spec.HeadCount != 0 || spec.HeadCountKV != 0 ||
		spec.FeedForwardLength != 0 || spec.SSMConvKernel != 3 || spec.SSMInnerSize != 8 ||
		spec.SSMStateSize != 2 || spec.SSMTimeStepRank != 4 || spec.SSMGroupCount != 2 {
		t.Fatalf("unexpected Mamba2 spec: %+v", spec)
	}
}

func TestReadFalconH1Spec(t *testing.T) {
	file := &gguf.File{Metadata: []gguf.Metadata{
		metadata("general.architecture", gguf.ValueTypeString, "falcon-h1"),
		metadata("falcon-h1.block_count", gguf.ValueTypeUint32, uint32(2)),
		metadata("falcon-h1.context_length", gguf.ValueTypeUint32, uint32(4096)),
		metadata("falcon-h1.embedding_length", gguf.ValueTypeUint32, uint32(8)),
		metadata("falcon-h1.feed_forward_length", gguf.ValueTypeUint32, uint32(12)),
		metadata("falcon-h1.attention.head_count", gguf.ValueTypeUint32, uint32(2)),
		metadata("falcon-h1.attention.head_count_kv", gguf.ValueTypeUint32, uint32(1)),
		metadata("falcon-h1.attention.layer_norm_rms_epsilon", gguf.ValueTypeFloat32, float32(1e-5)),
		metadata("falcon-h1.rope.freq_base", gguf.ValueTypeFloat32, float32(10000)),
		metadata("falcon-h1.ssm.conv_kernel", gguf.ValueTypeUint32, uint32(3)),
		metadata("falcon-h1.ssm.inner_size", gguf.ValueTypeUint32, uint32(8)),
		metadata("falcon-h1.ssm.state_size", gguf.ValueTypeUint32, uint32(2)),
		metadata("falcon-h1.ssm.time_step_rank", gguf.ValueTypeUint32, uint32(4)),
		metadata("falcon-h1.ssm.group_count", gguf.ValueTypeUint32, uint32(2)),
		metadata("falcon-h1.vocab_size", gguf.ValueTypeUint32, uint32(32)),
	}}
	spec, err := ReadSpec(file)
	if err != nil {
		t.Fatal(err)
	}
	if spec.Architecture != "falcon-h1" || spec.RopeDisabled || spec.HeadCount != 2 ||
		spec.HeadCountKV != 1 || spec.KeyLength != 4 || spec.ValueLength != 4 ||
		spec.RopeDimensionCount != 4 || spec.SSMConvKernel != 3 || spec.SSMInnerSize != 8 ||
		spec.SSMStateSize != 2 || spec.SSMTimeStepRank != 4 || spec.SSMGroupCount != 2 {
		t.Fatalf("unexpected Falcon-H1 spec: %+v", spec)
	}
}

func TestReadJambaSpec(t *testing.T) {
	file := &gguf.File{Metadata: []gguf.Metadata{
		metadata("general.architecture", gguf.ValueTypeString, "jamba"),
		metadata("jamba.block_count", gguf.ValueTypeUint32, uint32(2)),
		metadata("jamba.context_length", gguf.ValueTypeUint32, uint32(4096)),
		metadata("jamba.embedding_length", gguf.ValueTypeUint32, uint32(4)),
		metadata("jamba.feed_forward_length", gguf.ValueTypeUint32, uint32(6)),
		metadata("jamba.attention.head_count", gguf.ValueTypeUint32, uint32(2)),
		{Key: "jamba.attention.head_count_kv", Value: gguf.Value{
			Type: gguf.ValueTypeArray, ArrayType: gguf.ValueTypeUint32, Data: []uint32{0, 1},
		}},
		metadata("jamba.attention.layer_norm_rms_epsilon", gguf.ValueTypeFloat32, float32(1e-5)),
		metadata("jamba.ssm.conv_kernel", gguf.ValueTypeUint32, uint32(3)),
		metadata("jamba.ssm.inner_size", gguf.ValueTypeUint32, uint32(8)),
		metadata("jamba.ssm.state_size", gguf.ValueTypeUint32, uint32(2)),
		metadata("jamba.ssm.time_step_rank", gguf.ValueTypeUint32, uint32(2)),
		metadata("jamba.expert_count", gguf.ValueTypeUint32, uint32(4)),
		metadata("jamba.expert_used_count", gguf.ValueTypeUint32, uint32(2)),
		metadata("jamba.vocab_size", gguf.ValueTypeUint32, uint32(32)),
	}}
	spec, err := ReadSpec(file)
	if err != nil {
		t.Fatal(err)
	}
	if spec.Architecture != "jamba" || !spec.RopeDisabled || !spec.IsRecurrentLayer(0) ||
		spec.IsRecurrentLayer(1) || spec.LayerKVHeadCount(1) != 1 || spec.SSMInnerSize != 8 ||
		spec.SSMGroupCount != 1 || spec.ExpertFeedForward != 6 || spec.ExpertWeightsNorm {
		t.Fatalf("unexpected Jamba spec: %+v", spec)
	}
}

func TestReadGraniteHybridSpec(t *testing.T) {
	file := &gguf.File{Metadata: []gguf.Metadata{
		metadata("general.architecture", gguf.ValueTypeString, "granitehybrid"),
		metadata("granitehybrid.block_count", gguf.ValueTypeUint32, uint32(2)),
		metadata("granitehybrid.context_length", gguf.ValueTypeUint32, uint32(4096)),
		metadata("granitehybrid.embedding_length", gguf.ValueTypeUint32, uint32(4)),
		metadata("granitehybrid.feed_forward_length", gguf.ValueTypeUint32, uint32(6)),
		metadata("granitehybrid.attention.head_count", gguf.ValueTypeUint32, uint32(2)),
		{Key: "granitehybrid.attention.head_count_kv", Value: gguf.Value{
			Type: gguf.ValueTypeArray, ArrayType: gguf.ValueTypeUint32, Data: []uint32{0, 1},
		}},
		metadata("granitehybrid.rope.freq_base", gguf.ValueTypeFloat32, float32(10000)),
		metadata("granitehybrid.rope.scaling.finetuned", gguf.ValueTypeBool, false),
		metadata("granitehybrid.attention.layer_norm_rms_epsilon", gguf.ValueTypeFloat32, float32(1e-5)),
		metadata("granitehybrid.attention.scale", gguf.ValueTypeFloat32, float32(0.25)),
		metadata("granitehybrid.embedding_scale", gguf.ValueTypeFloat32, float32(2)),
		metadata("granitehybrid.residual_scale", gguf.ValueTypeFloat32, float32(0.5)),
		metadata("granitehybrid.logit_scale", gguf.ValueTypeFloat32, float32(8)),
		metadata("granitehybrid.ssm.conv_kernel", gguf.ValueTypeUint32, uint32(3)),
		metadata("granitehybrid.ssm.inner_size", gguf.ValueTypeUint32, uint32(8)),
		metadata("granitehybrid.ssm.state_size", gguf.ValueTypeUint32, uint32(2)),
		metadata("granitehybrid.ssm.time_step_rank", gguf.ValueTypeUint32, uint32(4)),
		metadata("granitehybrid.ssm.group_count", gguf.ValueTypeUint32, uint32(2)),
		metadata("granitehybrid.expert_count", gguf.ValueTypeUint32, uint32(4)),
		metadata("granitehybrid.expert_used_count", gguf.ValueTypeUint32, uint32(2)),
		metadata("granitehybrid.expert_shared_feed_forward_length", gguf.ValueTypeUint32, uint32(5)),
		metadata("granitehybrid.vocab_size", gguf.ValueTypeUint32, uint32(32)),
	}}
	spec, err := ReadSpec(file)
	if err != nil {
		t.Fatal(err)
	}
	if spec.Architecture != "granitehybrid" || !spec.RopeDisabled || !spec.IsRecurrentLayer(0) ||
		spec.IsRecurrentLayer(1) || spec.LayerKVHeadCount(1) != 1 || spec.SSMInnerSize != 8 ||
		spec.SSMTimeStepRank != 4 || spec.SSMGroupCount != 2 || spec.ExpertFeedForward != 6 ||
		!spec.ExpertWeightsNorm || spec.SharedExpertFF != 5 || spec.AttentionScale != 0.25 ||
		spec.EmbeddingScale != 2 || spec.ResidualScale != 0.5 || spec.OutputLogitMultiplier() != 0.125 {
		t.Fatalf("unexpected Granite Hybrid spec: %+v", spec)
	}
}

func TestReadPLaMo2Spec(t *testing.T) {
	file := &gguf.File{Metadata: []gguf.Metadata{
		metadata("general.architecture", gguf.ValueTypeString, "plamo2"),
		metadata("plamo2.block_count", gguf.ValueTypeUint32, uint32(2)),
		metadata("plamo2.context_length", gguf.ValueTypeUint32, uint32(4096)),
		metadata("plamo2.embedding_length", gguf.ValueTypeUint32, uint32(4)),
		metadata("plamo2.feed_forward_length", gguf.ValueTypeUint32, uint32(6)),
		metadata("plamo2.attention.head_count", gguf.ValueTypeUint32, uint32(2)),
		{Key: "plamo2.attention.head_count_kv", Value: gguf.Value{
			Type: gguf.ValueTypeArray, ArrayType: gguf.ValueTypeUint32, Data: []uint32{0, 1},
		}},
		metadata("plamo2.attention.key_length", gguf.ValueTypeUint32, uint32(2)),
		metadata("plamo2.attention.value_length", gguf.ValueTypeUint32, uint32(2)),
		metadata("plamo2.rope.freq_base", gguf.ValueTypeFloat32, float32(10000)),
		metadata("plamo2.attention.layer_norm_rms_epsilon", gguf.ValueTypeFloat32, float32(1e-5)),
		metadata("plamo2.ssm.conv_kernel", gguf.ValueTypeUint32, uint32(3)),
		metadata("plamo2.ssm.inner_size", gguf.ValueTypeUint32, uint32(8)),
		metadata("plamo2.ssm.state_size", gguf.ValueTypeUint32, uint32(2)),
		metadata("plamo2.ssm.time_step_rank", gguf.ValueTypeUint32, uint32(4)),
		metadata("plamo2.ssm.group_count", gguf.ValueTypeUint32, uint32(0)),
		metadata("plamo2.vocab_size", gguf.ValueTypeUint32, uint32(32)),
	}}
	spec, err := ReadSpec(file)
	if err != nil {
		t.Fatal(err)
	}
	if spec.Architecture != "plamo2" || !spec.IsRecurrentLayer(0) || spec.IsRecurrentLayer(1) ||
		spec.LayerKVHeadCount(1) != 1 || spec.SSMInnerSize != 8 || spec.SSMStateSize != 2 ||
		spec.SSMTimeStepRank != 4 || spec.SSMGroupCount != 0 ||
		math.Abs(float64(spec.AttentionScale)-1/math.Sqrt(2)) > 1e-6 || !usesNormalRoPE(spec.Architecture) {
		t.Fatalf("unexpected PLaMo2 spec: %+v", spec)
	}
}

func TestReadNemotronHMoESpec(t *testing.T) {
	file := &gguf.File{Metadata: []gguf.Metadata{
		metadata("general.architecture", gguf.ValueTypeString, "nemotron_h_moe"),
		metadata("nemotron_h_moe.block_count", gguf.ValueTypeUint32, uint32(3)),
		metadata("nemotron_h_moe.context_length", gguf.ValueTypeUint32, uint32(4096)),
		metadata("nemotron_h_moe.embedding_length", gguf.ValueTypeUint32, uint32(8)),
		{Key: "nemotron_h_moe.feed_forward_length", Value: gguf.Value{
			Type: gguf.ValueTypeArray, ArrayType: gguf.ValueTypeUint32, Data: []uint32{0, 0, 6},
		}},
		{Key: "nemotron_h_moe.attention.head_count", Value: gguf.Value{
			Type: gguf.ValueTypeArray, ArrayType: gguf.ValueTypeUint32, Data: []uint32{2, 0, 2},
		}},
		{Key: "nemotron_h_moe.attention.head_count_kv", Value: gguf.Value{
			Type: gguf.ValueTypeArray, ArrayType: gguf.ValueTypeUint32, Data: []uint32{1, 0, 1},
		}},
		metadata("nemotron_h_moe.attention.key_length", gguf.ValueTypeUint32, uint32(4)),
		metadata("nemotron_h_moe.attention.value_length", gguf.ValueTypeUint32, uint32(4)),
		metadata("nemotron_h_moe.attention.layer_norm_rms_epsilon", gguf.ValueTypeFloat32, float32(1e-5)),
		metadata("nemotron_h_moe.ssm.conv_kernel", gguf.ValueTypeUint32, uint32(3)),
		metadata("nemotron_h_moe.ssm.inner_size", gguf.ValueTypeUint32, uint32(16)),
		metadata("nemotron_h_moe.ssm.state_size", gguf.ValueTypeUint32, uint32(2)),
		metadata("nemotron_h_moe.ssm.time_step_rank", gguf.ValueTypeUint32, uint32(4)),
		metadata("nemotron_h_moe.ssm.group_count", gguf.ValueTypeUint32, uint32(2)),
		metadata("nemotron_h_moe.expert_count", gguf.ValueTypeUint32, uint32(4)),
		metadata("nemotron_h_moe.expert_used_count", gguf.ValueTypeUint32, uint32(2)),
		metadata("nemotron_h_moe.expert_feed_forward_length", gguf.ValueTypeUint32, uint32(6)),
		metadata("nemotron_h_moe.expert_shared_feed_forward_length", gguf.ValueTypeUint32, uint32(5)),
		metadata("nemotron_h_moe.expert_shared_count", gguf.ValueTypeUint32, uint32(1)),
		metadata("nemotron_h_moe.expert_weights_norm", gguf.ValueTypeBool, true),
		metadata("nemotron_h_moe.expert_weights_scale", gguf.ValueTypeFloat32, float32(1.25)),
		metadata("nemotron_h_moe.moe_latent_size", gguf.ValueTypeUint32, uint32(4)),
		metadata("nemotron_h_moe.vocab_size", gguf.ValueTypeUint32, uint32(32)),
	}}
	spec, err := ReadSpec(file)
	if err != nil {
		t.Fatal(err)
	}
	if spec.Architecture != "nemotron_h_moe" || !spec.RopeDisabled || spec.IsRecurrentLayer(0) ||
		!spec.IsRecurrentLayer(1) || spec.IsRecurrentLayer(2) || spec.LayerFeedForwardLength(2) != 6 ||
		spec.LayerHeadCount(1) != 0 || spec.LayerKVHeadCount(0) != 1 || spec.SSMInnerSize != 16 ||
		spec.SSMGroupCount != 2 || spec.ExpertCount != 4 || spec.ExpertUsedCount != 2 ||
		spec.ExpertFeedForward != 6 || spec.SharedExpertFF != 5 || spec.MoELatentSize != 4 ||
		!spec.ExpertWeightsNorm || spec.ExpertWeightsScale != 1.25 || spec.ExpertGatingFunc != 2 {
		t.Fatalf("unexpected Nemotron-H MoE spec: %+v", spec)
	}
}

func TestReadTalkieSpecUsesUnweightedRMSNormAndDirectLogitScale(t *testing.T) {
	file := &gguf.File{Metadata: []gguf.Metadata{
		metadata("general.architecture", gguf.ValueTypeString, "talkie"),
		metadata("talkie.block_count", gguf.ValueTypeUint32, uint32(2)),
		metadata("talkie.context_length", gguf.ValueTypeUint32, uint32(2048)),
		metadata("talkie.embedding_length", gguf.ValueTypeUint32, uint32(8)),
		metadata("talkie.feed_forward_length", gguf.ValueTypeUint32, uint32(12)),
		metadata("talkie.attention.head_count", gguf.ValueTypeUint32, uint32(2)),
		metadata("talkie.attention.head_count_kv", gguf.ValueTypeUint32, uint32(1)),
		metadata("talkie.attention.key_length", gguf.ValueTypeUint32, uint32(4)),
		metadata("talkie.attention.value_length", gguf.ValueTypeUint32, uint32(4)),
		metadata("talkie.rope.freq_base", gguf.ValueTypeFloat32, float32(10000)),
		metadata("talkie.attention.layer_norm_rms_epsilon", gguf.ValueTypeFloat32, float32(1e-6)),
		metadata("talkie.logit_scale", gguf.ValueTypeFloat32, float32(0.125)),
		metadata("talkie.vocab_size", gguf.ValueTypeUint32, uint32(32)),
	}}
	spec, err := ReadSpec(file)
	if err != nil {
		t.Fatal(err)
	}
	if spec.Architecture != "talkie" || !spec.UsesUnweightedRMSNorm() ||
		spec.UsesUnweightedLayerNorm() || spec.RopeDimensionCount != 4 ||
		usesNormalRoPE(spec.Architecture) || spec.OutputLogitMultiplier() != 0.125 {
		t.Fatalf("unexpected Talkie spec: %+v", spec)
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
