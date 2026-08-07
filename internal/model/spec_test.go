package model

import (
	"overgo/internal/gguf"

	"math"

	"testing"
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
	profile, _ := LookupArchitecture("qwen3")
	profile.DenseGraph = DenseGraphTalkie
	resolved, err := ReadSpecWithProfile(file, profile)
	if err != nil {
		t.Fatal(err)
	}
	if resolved.Profile().DenseGraph != DenseGraphTalkie {
		t.Fatal("resolved profile was not pinned to the spec")
	}
	profile.Name = "llama"
	if _, err := ReadSpecWithProfile(file, profile); err == nil {
		t.Fatal("mismatched resolved profile accepted")
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
		spec.RopeFrequencySWA != 10000 || spec.ExpertGatingFunc != expertGatingSelectedSoftmax || spec.ExpertWeightsNorm {
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
	if spec.Architecture != "glm4moe" || spec.BlockCount != 2 || spec.NextNPredictLayers != 1 ||
		spec.LeadingDenseBlocks != 1 || spec.SharedExpertFF != 12 ||
		spec.ExpertGatingFunc != expertGatingSigmoid || !spec.ExpertWeightsNorm || spec.RopeSections[1] != 1 {
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
	if spec.Architecture != "mimo2" || spec.BlockCount != 3 || spec.NextNPredictLayers != 1 ||
		len(spec.LayerKVHeadCounts) != 3 || spec.LayerKVHeadCount(1) != 2 ||
		len(spec.SlidingLayers) != 3 || !spec.IsSlidingLayer(0) || spec.IsSlidingLayer(1) ||
		!spec.IsSlidingLayer(2) || spec.AttentionValueScale != 0.5 ||
		spec.RopeDimensionCount != 4 || spec.RopeFrequencySWA != 20000 ||
		spec.ExpertGatingFunc != expertGatingSigmoid || !spec.ExpertWeightsNorm {
		t.Fatalf("unexpected MiMo2 spec: %+v", spec)
	}
}

func TestReadStep35SpecPreservesMTPArrays(t *testing.T) {
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
	if spec.BlockCount != 2 || spec.NextNPredictLayers != 1 ||
		len(spec.LayerHeadCounts) != 3 || len(spec.LayerKVHeadCounts) != 3 ||
		spec.LayerHeadCount(1) != 4 || spec.LayerKVHeadCount(1) != 2 ||
		spec.LayerHeadCount(2) != 2 || spec.LayerKVHeadCount(2) != 1 ||
		spec.LayerRopeDimensionCount(0) != 2 || spec.LayerRopeDimensionCount(1) != 4 ||
		spec.IsSlidingLayer(0) || !spec.IsSlidingLayer(1) || spec.IsSlidingLayer(2) ||
		spec.LayerExpertSwiGLUClamp(1) != 3 || spec.LayerSharedSwiGLUClampLimit(1) != 5 ||
		spec.LayerExpertSwiGLUClamp(2) != 0 || spec.LayerSharedSwiGLUClampLimit(2) != 0 ||
		spec.ExpertGatingFunc != expertGatingSigmoid || !spec.ExpertWeightsNorm || spec.SharedExpertFF != 8 {
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
		metadata("bailingmoe2.expert_gating_func", gguf.ValueTypeUint32, expertGatingSigmoid),
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
	if spec.Architecture != "bailingmoe2" || spec.BlockCount != 2 || spec.NextNPredictLayers != 1 ||
		spec.LeadingDenseBlocks != 1 || spec.ExpertCount != 8 || spec.ExpertUsedCount != 2 ||
		spec.ExpertFeedForward != 6 || spec.SharedExpertCount != 2 || spec.SharedExpertFF != 10 ||
		spec.ExpertGatingFunc != expertGatingSigmoid || !spec.ExpertWeightsNorm || spec.ExpertWeightsScale != 1.25 ||
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
		metadata("exaone-moe.expert_gating_func", gguf.ValueTypeUint32, expertGatingSigmoid),
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
	if spec.BlockCount != 4 || spec.NextNPredictLayers != 1 ||
		spec.ExpertFeedForward != 6 || spec.SharedExpertFF != 12 ||
		spec.ExpertGatingFunc != expertGatingSigmoid || !spec.ExpertWeightsNorm ||
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
	moeFile := &gguf.File{Metadata: append([]gguf.Metadata(nil), file.Metadata...)}
	moeFile.Metadata = append(moeFile.Metadata,
		metadata("jina-bert-v3.expert_count", gguf.ValueTypeUint32, uint32(4)),
		metadata("jina-bert-v3.expert_used_count", gguf.ValueTypeUint32, uint32(2)),
		metadata("jina-bert-v3.moe_every_n_layers", gguf.ValueTypeUint32, uint32(2)),
	)
	moeSpec, err := ReadSpec(moeFile)
	if err != nil {
		t.Fatal(err)
	}
	if moeSpec.ExpertCount != 4 || moeSpec.ExpertUsedCount != 2 ||
		moeSpec.ExpertFeedForward != 16 || moeSpec.MoELayerStep != 2 ||
		moeSpec.IsInterleavedMoELayer(0) || !moeSpec.IsInterleavedMoELayer(1) {
		t.Fatalf("unexpected JinaBERT v3 MoE spec: %+v", moeSpec)
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
		!spec.ExpertWeightsNorm || spec.ExpertGatingFunc != expertGatingSigmoid ||
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
