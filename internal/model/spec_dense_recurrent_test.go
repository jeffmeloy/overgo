package model

import (
	"llamacpp2go/internal/gguf"

	"math"

	"slices"

	"testing"
)

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

func TestReadGraniteSameArchitectureMoESpec(t *testing.T) {
	file := &gguf.File{Metadata: append(graniteMetadata(),
		metadata("granite.expert_count", gguf.ValueTypeUint32, uint32(8)),
		metadata("granite.expert_used_count", gguf.ValueTypeUint32, uint32(2)),
		metadata("granite.expert_shared_feed_forward_length", gguf.ValueTypeUint32, uint32(4096)),
	)}
	spec, err := ReadSpec(file)
	if err != nil {
		t.Fatal(err)
	}
	if spec.Architecture != "granite" || spec.ExpertCount != 8 ||
		spec.ExpertUsedCount != 2 || spec.ExpertFeedForward != spec.FeedForwardLength ||
		spec.SharedExpertFF != 4096 || !spec.ExpertWeightsNorm || spec.ExpertWeightsScale != 1 {
		t.Fatalf("unexpected Granite MoE spec: %+v", spec)
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

func TestReadGraniteRejectsIncompleteMoE(t *testing.T) {
	file := &gguf.File{Metadata: append(
		graniteMetadata(),
		metadata("granite.expert_count", gguf.ValueTypeUint32, uint32(8)),
	)}
	if _, err := ReadSpec(file); err == nil {
		t.Fatal("incomplete Granite MoE metadata was accepted")
	}
}

func TestReadGraniteDeepstackMapping(t *testing.T) {
	mapping := make([]int32, 32)
	for index := range mapping {
		mapping[index] = -1
	}
	mapping[0], mapping[3], mapping[7] = 0, 1, 2
	file := &gguf.File{Metadata: append(graniteMetadata(), gguf.Metadata{
		Key: "granite.deepstack_mapping",
		Value: gguf.Value{
			Type: gguf.ValueTypeArray, ArrayType: gguf.ValueTypeInt32, Data: mapping,
		},
	})}
	spec, err := ReadSpec(file)
	if err != nil {
		t.Fatal(err)
	}
	if spec.DeepstackLayerCount != 3 || !slices.Equal(spec.DeepstackMapping, mapping) {
		t.Fatalf("Granite deepstack count/mapping = %d/%v", spec.DeepstackLayerCount, spec.DeepstackMapping)
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

func TestReadMistral3YaRNSpec(t *testing.T) {
	file := &gguf.File{Metadata: append(mistral3Metadata(),
		metadata("mistral3.rope.scaling.type", gguf.ValueTypeString, "yarn"),
		metadata("mistral3.rope.scaling.factor", gguf.ValueTypeFloat32, float32(4)),
		metadata("mistral3.rope.scaling.original_context_length", gguf.ValueTypeUint32, uint32(8192)),
		metadata("mistral3.rope.scaling.attn_factor", gguf.ValueTypeFloat32, float32(1.2)),
		metadata("mistral3.rope.scaling.yarn_log_multiplier", gguf.ValueTypeFloat32, float32(2)),
		metadata("mistral3.rope.scaling.yarn_beta_fast", gguf.ValueTypeFloat32, float32(16)),
		metadata("mistral3.rope.scaling.yarn_beta_slow", gguf.ValueTypeFloat32, float32(2)),
	)}
	spec, err := ReadSpec(file)
	if err != nil {
		t.Fatal(err)
	}
	wantAttention := float32(1.2 / (1 + 0.2*math.Log(4)))
	if spec.RopeScalingType != "yarn" || spec.RopeScalingFactor != 4 ||
		spec.OriginalContextLength != 8192 || spec.YaRNExtFactor != 1 ||
		math.Abs(float64(spec.YaRNAttentionFactor-wantAttention)) > 1e-6 ||
		spec.YaRNBetaFast != 16 || spec.YaRNBetaSlow != 2 ||
		spec.RopeYaRNLogMultiplier != 2 {
		t.Fatalf("unexpected Mistral 3 YaRN spec: %+v", spec)
	}
}

func TestReadNormalRoPEYaRNSpecs(t *testing.T) {
	for _, architecture := range []string{"llama", "llama-embed", "minicpm"} {
		t.Run(architecture, func(t *testing.T) {
			prefix := architecture + "."
			file := &gguf.File{Metadata: []gguf.Metadata{
				metadata("general.architecture", gguf.ValueTypeString, architecture),
				metadata(prefix+"block_count", gguf.ValueTypeUint32, uint32(1)),
				metadata(prefix+"context_length", gguf.ValueTypeUint32, uint32(8192)),
				metadata(prefix+"embedding_length", gguf.ValueTypeUint32, uint32(8)),
				metadata(prefix+"feed_forward_length", gguf.ValueTypeUint32, uint32(16)),
				metadata(prefix+"attention.head_count", gguf.ValueTypeUint32, uint32(2)),
				metadata(prefix+"attention.head_count_kv", gguf.ValueTypeUint32, uint32(1)),
				metadata(prefix+"rope.freq_base", gguf.ValueTypeFloat32, float32(10000)),
				metadata(prefix+"rope.scaling.type", gguf.ValueTypeString, "yarn"),
				metadata(prefix+"rope.scaling.factor", gguf.ValueTypeFloat32, float32(4)),
				metadata(prefix+"rope.scaling.original_context_length", gguf.ValueTypeUint32, uint32(4096)),
				metadata(prefix+"rope.scaling.attn_factor", gguf.ValueTypeFloat32, float32(1.2)),
				metadata(prefix+"attention.layer_norm_rms_epsilon", gguf.ValueTypeFloat32, float32(1e-5)),
			}}
			spec, err := ReadSpec(file)
			if err != nil {
				t.Fatal(err)
			}
			if spec.RopeScalingType != "yarn" || spec.RopeScalingFactor != 4 ||
				spec.OriginalContextLength != 4096 || spec.YaRNExtFactor != 1 ||
				spec.YaRNAttentionFactor != 1.2 || spec.YaRNBetaFast != 32 ||
				spec.YaRNBetaSlow != 1 {
				t.Fatalf("unexpected %s YaRN spec: %+v", architecture, spec)
			}
		})
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
		metadata("modern-bert.hidden_activation", gguf.ValueTypeString, "swish"),
		metadata("modern-bert.vocab_size", gguf.ValueTypeUint32, uint32(32)),
	}}
	spec, err := ReadSpec(file)
	if err != nil {
		t.Fatal(err)
	}
	if spec.Architecture != "modern-bert" || !spec.NonCausalAttention || !spec.IsEncoderOnly() ||
		!spec.UsesWeightOnlyLayerNorm() || spec.HeadCountKV != 2 || spec.RopeDimensionCount != 4 ||
		spec.HiddenActivation != "swiglu" || spec.IsSlidingLayer(0) || !spec.IsSlidingLayer(1) ||
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
