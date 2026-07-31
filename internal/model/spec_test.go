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
