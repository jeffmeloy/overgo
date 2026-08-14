package model

import (
	"fmt"
	"overgo/internal/gguf"

	"math"

	"strings"

	"testing"
)

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
			Data: []int32{1, 1, 0, 0}},
	})
	spec, err = ReadSpec(file)
	if err != nil {
		t.Fatal(err)
	}
	if spec.RopeSections != [4]int32{1, 1, 0, 0} {
		t.Fatalf("Hunyuan-Dense MRoPE sections = %v", spec.RopeSections)
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
	if spec.Architecture != "afmoe" || spec.ExpertGatingFunc != expertGatingSigmoid ||
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
		metadata("lfm2moe.expert_gating_func", gguf.ValueTypeUint32, expertGatingSigmoid),
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
		spec.ExpertWeightsScale != 1.25 || spec.ExpertGatingFunc != expertGatingSigmoid ||
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
		spec.SharedExpertFF != 6 || spec.ExpertGatingFunc != expertGatingSoftmax || spec.AttentionTempFloor != 8192 {
		t.Fatalf("unexpected DeepSeek2 spec: %+v", spec)
	}
	wantAttentionFactor := float32(1 / (1 + 0.1*math.Log(4)))
	if math.Abs(float64(spec.YaRNAttentionFactor-wantAttentionFactor)) > 1e-6 {
		t.Fatalf("DeepSeek2 YaRN attention factor = %v, want %v", spec.YaRNAttentionFactor, wantAttentionFactor)
	}
	profile := spec.Profile()
	profile.Experts.Routing = expertRouteSigmoid
	sigmoid, err := ReadSpecWithProfile(file, profile)
	if err != nil {
		t.Fatal(err)
	}
	if sigmoid.ExpertGatingFunc != expertGatingSigmoid {
		t.Fatalf("profile-selected expert gating = %d", sigmoid.ExpertGatingFunc)
	}
}

func TestReadGLMDSASpec(t *testing.T) {
	prefix := "glm-dsa."
	file := &gguf.File{Metadata: []gguf.Metadata{
		metadata("general.architecture", gguf.ValueTypeString, "glm-dsa"),
		metadata(prefix+"block_count", gguf.ValueTypeUint32, uint32(7)),
		metadata(prefix+"nextn_predict_layers", gguf.ValueTypeUint32, uint32(1)),
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
	if spec.Architecture != "glm-dsa" || spec.BlockCount != 6 || spec.NextNPredictLayers != 1 ||
		spec.IndexerHeadCount != 2 || spec.IndexerKeyLength != 8 ||
		spec.IndexerTopK != 4 || spec.ExpertGatingFunc != expertGatingSigmoid || spec.RopeScalingType != "yarn" ||
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
		metadata(prefix+"expert_gating_func", gguf.ValueTypeUint32, expertGatingSigmoid),
		metadata(prefix+"attention.indexer.head_count", gguf.ValueTypeUint32, uint32(2)),
		metadata(prefix+"attention.indexer.key_length", gguf.ValueTypeUint32, uint32(8)),
		metadata(prefix+"attention.indexer.top_k", gguf.ValueTypeUint32, uint32(4)),
	}}
	spec, err := ReadSpec(file)
	if err != nil {
		t.Fatal(err)
	}
	if spec.Architecture != "deepseek32" || spec.BlockCount != 62 || spec.NextNPredictLayers != 1 ||
		spec.IndexerHeadCount != 2 || spec.IndexerKeyLength != 8 || spec.IndexerTopK != 4 ||
		spec.ExpertGatingFunc != expertGatingSigmoid || spec.LayerNormEpsilon != 1e-6 ||
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
		metadata(prefix+"expert_gating_func", gguf.ValueTypeUint32, expertGatingSqrtSoftplus),
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
		spec.ExpertGatingFunc != expertGatingSqrtSoftplus || spec.SharedExpertFF != 8 || spec.LayerExpertSwiGLUClamp(0) != 7 ||
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
	profile, ok := LookupArchitecture("deepseek2")
	if !ok {
		t.Fatal("DeepSeek2 profile is absent")
	}
	profile.Validation.QLoRARankOptional = true
	spec, err := ReadSpecWithProfile(file, profile)
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

func TestReadRefactMoESpec(t *testing.T) {
	file := &gguf.File{Metadata: []gguf.Metadata{
		metadata("general.architecture", gguf.ValueTypeString, "refact"),
		metadata("refact.block_count", gguf.ValueTypeUint32, uint32(2)),
		metadata("refact.context_length", gguf.ValueTypeUint32, uint32(128)),
		metadata("refact.embedding_length", gguf.ValueTypeUint32, uint32(16)),
		metadata("refact.feed_forward_length", gguf.ValueTypeUint32, uint32(64)),
		metadata("refact.attention.head_count", gguf.ValueTypeUint32, uint32(4)),
		metadata("refact.attention.head_count_kv", gguf.ValueTypeUint32, uint32(2)),
		metadata("refact.attention.layer_norm_rms_epsilon", gguf.ValueTypeFloat32, float32(1e-5)),
		metadata("refact.vocab_size", gguf.ValueTypeUint32, uint32(32)),
		metadata("refact.expert_count", gguf.ValueTypeUint32, uint32(8)),
		metadata("refact.expert_used_count", gguf.ValueTypeUint32, uint32(2)),
	}}
	spec, err := ReadSpec(file)
	if err != nil {
		t.Fatal(err)
	}
	if spec.ExpertCount != 8 || spec.ExpertUsedCount != 2 ||
		spec.ExpertFeedForward != 64 || !spec.ExpertWeightsNorm || spec.ExpertWeightsScale != 1 {
		t.Fatalf("unexpected Refact MoE spec: %+v", spec)
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

func TestReadCohere2ArraySlidingSpec(t *testing.T) {
	file := &gguf.File{Metadata: []gguf.Metadata{
		metadata("general.architecture", gguf.ValueTypeString, "cohere2"),
		metadata("cohere2.block_count", gguf.ValueTypeUint32, uint32(4)),
		metadata("cohere2.context_length", gguf.ValueTypeUint32, uint32(4096)),
		metadata("cohere2.embedding_length", gguf.ValueTypeUint32, uint32(8)),
		metadata("cohere2.feed_forward_length", gguf.ValueTypeUint32, uint32(12)),
		metadata("cohere2.vocab_size", gguf.ValueTypeUint32, uint32(32)),
		metadata("cohere2.attention.head_count", gguf.ValueTypeUint32, uint32(2)),
		metadata("cohere2.attention.head_count_kv", gguf.ValueTypeUint32, uint32(1)),
		metadata("cohere2.rope.freq_base", gguf.ValueTypeFloat32, float32(10000)),
		metadata("cohere2.rope.freq_base_swa", gguf.ValueTypeFloat32, float32(20000)),
		metadata("cohere2.rope.dimension_count", gguf.ValueTypeUint32, uint32(4)),
		metadata("cohere2.attention.sliding_window", gguf.ValueTypeUint32, uint32(128)),
		metadata("cohere2.attention.layer_norm_epsilon", gguf.ValueTypeFloat32, float32(1e-5)),
		metadata("cohere2.logit_scale", gguf.ValueTypeFloat32, float32(0.5)),
		{
			Key: "cohere2.attention.sliding_window_pattern",
			Value: gguf.Value{Type: gguf.ValueTypeArray, ArrayType: gguf.ValueTypeBool,
				Data: []bool{true, false, true, false}},
		},
	}}
	spec, err := ReadSpec(file)
	if err != nil {
		t.Fatal(err)
	}
	if len(spec.SlidingLayers) != 4 || spec.NoRopeLayerStep != 0 ||
		!spec.IsSlidingLayer(0) || spec.IsSlidingLayer(1) ||
		!spec.IsSlidingLayer(2) || spec.IsSlidingLayer(3) ||
		!spec.UsesRoPE(0) || spec.UsesRoPE(1) ||
		!spec.UsesRoPE(2) || spec.UsesRoPE(3) {
		t.Fatalf("unexpected Cohere2 array spec: %+v", spec)
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
		spec.NextNPredictLayers != 1 ||
		spec.UsesWeightOnlyLayerNorm() || spec.LeadingDenseBlocks != 1 ||
		spec.ExpertFeedForward != 6 || spec.SharedExpertFF != 6 ||
		spec.ExpertGatingFunc != expertGatingSigmoid || !spec.ExpertWeightsNorm || spec.ExpertWeightsScale != 1.25 ||
		spec.IsSlidingLayer(0) || !spec.IsSlidingLayer(1) ||
		!spec.UsesRoPE(0) || !spec.UsesRoPE(1) || spec.UsesRoPE(2) ||
		spec.OutputLogitMultiplier() != 0.5 {
		t.Fatalf("unexpected Cohere2-MoE spec: %+v", spec)
	}
}

func TestReadHYV3SpecPreservesNextNCount(t *testing.T) {
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
	if spec.Architecture != "hy_v3" || spec.BlockCount != 4 || spec.NextNPredictLayers != 1 || spec.RMSNormEpsilon != 1e-5 ||
		spec.ExpertFeedForward != 6 || spec.SharedExpertFF != 6 || spec.ExpertGatingFunc != expertGatingSigmoid ||
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
		spec.ExpertGatingFunc != expertGatingSoftmax || spec.RopeDimensionCount != 4 ||
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

func TestReadGPTJSpec(t *testing.T) {
	file := &gguf.File{Metadata: []gguf.Metadata{
		metadata("general.architecture", gguf.ValueTypeString, "gptj"),
		metadata("gptj.block_count", gguf.ValueTypeUint32, uint32(28)),
		metadata("gptj.context_length", gguf.ValueTypeUint32, uint32(2048)),
		metadata("gptj.embedding_length", gguf.ValueTypeUint32, uint32(4096)),
		metadata("gptj.feed_forward_length", gguf.ValueTypeUint32, uint32(16384)),
		metadata("gptj.attention.head_count", gguf.ValueTypeUint32, uint32(16)),
		metadata("gptj.rope.freq_base", gguf.ValueTypeFloat32, float32(10000)),
		metadata("gptj.rope.dimension_count", gguf.ValueTypeUint32, uint32(64)),
		metadata("gptj.attention.layer_norm_epsilon", gguf.ValueTypeFloat32, float32(1e-5)),
	}}
	spec, err := ReadSpec(file)
	if err != nil {
		t.Fatal(err)
	}
	if spec.Architecture != "gptj" || spec.HeadCountKV != 16 || spec.KeyLength != 256 ||
		spec.RopeDimensionCount != 64 || spec.RopeFrequencyBase != 10000 ||
		!spec.UsesLayerNorm() || !spec.RequiresLayerNormBias() ||
		!usesNormalRoPE(spec.Architecture) || !usesParallelResidual(spec.Architecture) ||
		!usesGELU(spec.Architecture) {
		t.Fatalf("unexpected GPT-J spec: %+v", spec)
	}
}

func TestReadGPTJSpecRejectsInvalidRotaryWidth(t *testing.T) {
	for _, width := range []uint32{0, 3, 6} {
		t.Run(fmt.Sprint(width), func(t *testing.T) {
			file := &gguf.File{Metadata: []gguf.Metadata{
				metadata("general.architecture", gguf.ValueTypeString, "gptj"),
				metadata("gptj.block_count", gguf.ValueTypeUint32, uint32(1)),
				metadata("gptj.context_length", gguf.ValueTypeUint32, uint32(16)),
				metadata("gptj.embedding_length", gguf.ValueTypeUint32, uint32(8)),
				metadata("gptj.feed_forward_length", gguf.ValueTypeUint32, uint32(12)),
				metadata("gptj.attention.head_count", gguf.ValueTypeUint32, uint32(2)),
				metadata("gptj.rope.freq_base", gguf.ValueTypeFloat32, float32(10000)),
				metadata("gptj.rope.dimension_count", gguf.ValueTypeUint32, width),
				metadata("gptj.attention.layer_norm_epsilon", gguf.ValueTypeFloat32, float32(1e-5)),
			}}
			if _, err := ReadSpec(file); err == nil {
				t.Fatal("expected invalid GPT-J rotary width")
			}
		})
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
	nextNSpec, err := ReadSpec(file)
	if err != nil {
		t.Fatal(err)
	}
	if nextNSpec.BlockCount != 39 || nextNSpec.NextNPredictLayers != 1 {
		t.Fatalf("GLM4 NextN tail was not preserved: %+v", nextNSpec)
	}
	file.Metadata[len(file.Metadata)-1] = gguf.Metadata{
		Key: "glm4.rope.dimension_sections",
		Value: gguf.Value{
			Type: gguf.ValueTypeArray, ArrayType: gguf.ValueTypeInt32,
			Data: []int32{16, 24, 24, 0},
		},
	}
	spec, err = ReadSpec(file)
	if err != nil {
		t.Fatal(err)
	}
	if spec.RopeSections != [4]int32{16, 24, 24, 0} {
		t.Fatalf("GLM4 MRoPE sections = %v", spec.RopeSections)
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
	file.Metadata[1] = metadata("exaone4.block_count", gguf.ValueTypeUint32, uint32(65))
	file.Metadata = append(file.Metadata, metadata("exaone4.nextn_predict_layers", gguf.ValueTypeUint32, uint32(1)))
	nextNSpec, err := ReadSpec(file)
	if err != nil {
		t.Fatal(err)
	}
	if nextNSpec.BlockCount != 64 || nextNSpec.NextNPredictLayers != 1 ||
		nextNSpec.SlidingWindow != 4096 {
		t.Fatalf("EXAONE 4 NextN tail was not preserved: %+v", nextNSpec)
	}
}
