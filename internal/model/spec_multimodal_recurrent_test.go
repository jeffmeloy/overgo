package model

import (
	"errors"

	"overgo/internal/gguf"

	"math"

	"strings"

	"testing"
)

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
		t.Fatalf("unexpected AudioDecoder decoder spec: %+v", spec)
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
		metadata("qwen3vl.pooling_type", gguf.ValueTypeUint32, uint32(4)),
		metadata("qwen3vl.vocab_size", gguf.ValueTypeUint32, uint32(32)),
		{Key: "qwen3vl.classifier.output_labels", Value: gguf.Value{
			Type: gguf.ValueTypeArray, ArrayType: gguf.ValueTypeString, Data: []string{"no", "yes"},
		}},
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
		spec.DeepstackLayerCount != 3 || spec.PoolingType != 4 ||
		len(spec.ClassifierLabels) != 2 || spec.ClassifierLabels[1] != "yes" ||
		usesNormalRoPE(spec.Architecture) {
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
		"unsupported-test",
	} {
		t.Run(architecture, func(t *testing.T) {
			file := &gguf.File{Metadata: []gguf.Metadata{
				metadata("general.architecture", gguf.ValueTypeString, architecture),
			}}
			_, err := ReadSpec(file)
			unsupported, ok := errors.AsType[*UnsupportedArchitectureError](err)
			if !ok || unsupported.Architecture != architecture {
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
		!spec.ExpertWeightsNorm || spec.ExpertWeightsScale != 1.25 || spec.ExpertGatingFunc != expertGatingSigmoid {
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

func TestReadNormalRoPELongRoPESpecs(t *testing.T) {
	for _, architecture := range []string{"llama", "llama-embed", "minicpm", "mistral3"} {
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
				metadata(prefix+"rope.scaling.type", gguf.ValueTypeString, "longrope"),
				metadata(prefix+"rope.scaling.original_context_length", gguf.ValueTypeUint32, uint32(4096)),
				metadata(prefix+"rope.scaling.attn_factor", gguf.ValueTypeFloat32, float32(1.25)),
				metadata(prefix+"attention.layer_norm_rms_epsilon", gguf.ValueTypeFloat32, float32(1e-5)),
			}}
			spec, err := ReadSpec(file)
			if err != nil {
				t.Fatal(err)
			}
			if spec.RopeScalingType != "longrope" || spec.RopeDimensionCount != 4 ||
				spec.OriginalContextLength != 4096 || spec.RopeAttentionFactor != 1.25 ||
				!supportsLongRoPE(architecture) {
				t.Fatalf("unexpected %s LongRoPE spec: %+v", architecture, spec)
			}
		})
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
