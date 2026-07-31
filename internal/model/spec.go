package model

import (
	"errors"
	"fmt"
	"math"

	"llamacpp2go/internal/gguf"
)

// Spec: contains common transformer metadata needed to construct model
type Spec struct {
	Architecture          string
	Name                  string
	BlockCount            uint32
	ContextLength         uint32
	EmbeddingLength       uint32
	FeedForwardLength     uint32
	HeadCount             uint32
	HeadCountKV           uint32
	KeyLength             uint32
	ValueLength           uint32
	RopeFrequencyBase     float32
	RopeFrequencySWA      float32
	RopeScalingType       string
	RopeScalingFactor     float32
	RopeAttentionFactor   float32
	OriginalContextLength uint32
	AttentionScale        float32
	AttentionClamp        float32
	MaxALiBiBias          float32
	EmbeddingScale        float32
	ResidualScale         float32
	LogitScale            float32
	AttentionSoftcap      float32
	FinalLogitSoftcap     float32
	RMSNormEpsilon        float32
	LayerNormEpsilon      float32
	VocabularySize        uint32
	ExpertCount           uint32
	ExpertUsedCount       uint32
	ExpertFeedForward     uint32
	ExpertWeightsScale    float32
	LeadingDenseBlocks    uint32
	SharedExpertFF        uint32
	SharedExpertCount     uint32
	ExpertGatingFunc      uint32
	ExpertWeightsNorm     bool
	ShortConvCacheLength  uint32
	KVLoRARank            uint32
	QKNormEpsilon         float32
	SlidingWindow         uint32
	SlidingPattern        uint32
	RelativeBuckets       uint32
	NoRopeLayerStep       uint32
	RopeDisabled          bool
	ParallelResidual      bool
	NonCausalAttention    bool
	SandwichNorm          bool
	YaRNExtFactor         float32
	YaRNAttentionFactor   float32
	YaRNBetaFast          float32
	YaRNBetaSlow          float32
	RopeDimensionSWA      uint32
	LayerHeadCounts       []uint32
	LayerKVHeadCounts     []uint32
	LayerFeedForward      []uint32
	SlidingLayers         []bool
	XIELUAlphaN           []float32
	XIELUAlphaP           []float32
	XIELUBeta             []float32
	XIELUEpsilon          []float32

	// Qwen3.5 hybrid recurrent-attention metadata
	RopeDimensionCount    uint32
	RopeSections          [4]int32
	SSMConvKernel         uint32
	SSMInnerSize          uint32
	SSMStateSize          uint32
	SSMTimeStepRank       uint32
	SSMGroupCount         uint32
	FullAttentionInterval uint32
	RecurrentLayers       []bool
}

// UnsupportedArchitectureError: identifies valid GGUF architecture that
// runtime cannot execute yet
type UnsupportedArchitectureError struct {
	Architecture string
}

func (e *UnsupportedArchitectureError) Error() string {
	return fmt.Sprintf("model architecture %q is not supported", e.Architecture)
}

// ReadSpec: validates common metadata for initial Llama and Qwen3
// architecture families
func ReadSpec(file *gguf.File) (Spec, error) {
	if file == nil {
		return Spec{}, errors.New("model file is nil")
	}
	values := make(map[string]gguf.Value, len(file.Metadata))
	for _, item := range file.Metadata {
		values[item.Key] = item.Value
	}
	architecture, err := required[string](values, "general.architecture", gguf.ValueTypeString)
	if err != nil {
		return Spec{}, err
	}
	if architecture != "llama" && architecture != "internlm2" && architecture != "jais" &&
		architecture != "arcee" &&
		architecture != "apertus" &&
		architecture != "arctic" &&
		architecture != "baichuan" &&
		architecture != "bailingmoe" &&
		architecture != "bailingmoe2" &&
		architecture != "bitnet" &&
		architecture != "bloom" &&
		architecture != "codeshell" &&
		architecture != "chameleon" &&
		architecture != "dream" &&
		architecture != "deepseek" &&
		architecture != "deci" &&
		architecture != "dbrx" &&
		architecture != "dots1" &&
		architecture != "cohere2" &&
		architecture != "command-r" &&
		architecture != "jais2" &&
		architecture != "afmoe" &&
		architecture != "laguna" &&
		architecture != "lfm2" &&
		architecture != "lfm2moe" &&
		architecture != "llada" &&
		architecture != "llada-moe" &&
		architecture != "xverse" &&
		architecture != "exaone" && architecture != "olmo2" &&
		architecture != "exaone4" &&
		architecture != "exaone-moe" &&
		architecture != "smollm3" &&
		architecture != "smallthinker" &&
		architecture != "minicpm" &&
		architecture != "minimax-m2" &&
		architecture != "granite" &&
		architecture != "granitemoe" &&
		architecture != "glm4" &&
		architecture != "gpt2" &&
		architecture != "gptneox" &&
		architecture != "grok" &&
		architecture != "hunyuan-moe" &&
		architecture != "maincoder" &&
		architecture != "mellum" &&
		architecture != "mistral3" &&
		architecture != "mpt" &&
		architecture != "nemotron" &&
		architecture != "olmo" &&
		architecture != "olmoe" &&
		architecture != "openelm" &&
		architecture != "orion" &&
		architecture != "phi2" &&
		architecture != "phi3" &&
		architecture != "phimoe" &&
		architecture != "plamo" &&
		architecture != "plm" &&
		architecture != "seed_oss" &&
		architecture != "stablelm" &&
		architecture != "starcoder" &&
		architecture != "starcoder2" &&
		architecture != "qwen2" &&
		architecture != "qwen3" &&
		architecture != "qwen3moe" &&
		architecture != "qwen2moe" &&
		architecture != "qwen35" && architecture != "gemma" &&
		architecture != "refact" &&
		architecture != "rnd1" &&
		architecture != "gemma2" &&
		architecture != "gemma3" &&
		architecture != "falcon" &&
		architecture != "t5encoder" {
		return Spec{}, &UnsupportedArchitectureError{Architecture: architecture}
	}
	spec := Spec{Architecture: architecture}
	if architecture == "dream" || architecture == "llada" || architecture == "llada-moe" || architecture == "rnd1" {
		spec.NonCausalAttention = true
	}
	if architecture == "chameleon" {
		spec.QKNormEpsilon = 1e-5
		spec.SandwichNorm, _ = optional[bool](values, "chameleon.swin_norm", gguf.ValueTypeBool)
	}
	if value, ok := optional[string](values, "general.name", gguf.ValueTypeString); ok {
		spec.Name = value
	}
	prefix := architecture + "."
	isLlamaMoE := false
	if architecture == "llama" {
		if count, ok := optional[uint32](values, prefix+"expert_count", gguf.ValueTypeUint32); ok && count > 0 {
			isLlamaMoE = true
		}
	}
	if spec.BlockCount, err = required[uint32](values, prefix+"block_count", gguf.ValueTypeUint32); err != nil {
		return Spec{}, err
	}
	if spec.ContextLength, err = required[uint32](values, prefix+"context_length", gguf.ValueTypeUint32); err != nil {
		return Spec{}, err
	}
	if spec.EmbeddingLength, err = required[uint32](values, prefix+"embedding_length", gguf.ValueTypeUint32); err != nil {
		return Spec{}, err
	}
	if architecture == "deci" || architecture == "openelm" {
		if spec.LayerFeedForward, err = requiredLayerUint32(values, prefix+"feed_forward_length", spec.BlockCount); err != nil {
			return Spec{}, err
		}
		spec.FeedForwardLength = firstPositive(spec.LayerFeedForward)
	} else if spec.FeedForwardLength, err = required[uint32](values, prefix+"feed_forward_length", gguf.ValueTypeUint32); err != nil {
		return Spec{}, err
	}
	if architecture == "deci" || architecture == "laguna" || architecture == "openelm" {
		if spec.LayerHeadCounts, err = requiredLayerUint32(
			values, prefix+"attention.head_count", spec.BlockCount,
		); err != nil {
			return Spec{}, err
		}
		spec.HeadCount = firstPositive(spec.LayerHeadCounts)
	} else if spec.HeadCount, err = required[uint32](values, prefix+"attention.head_count", gguf.ValueTypeUint32); err != nil {
		return Spec{}, err
	}
	if architecture == "t5encoder" || architecture == "bloom" || architecture == "gpt2" || architecture == "jais" || architecture == "mpt" ||
		architecture == "starcoder" || architecture == "gptneox" || architecture == "falcon" {
		spec.HeadCountKV = spec.HeadCount
		if architecture == "gptneox" || architecture == "falcon" || architecture == "mpt" {
			if value, ok := optional[uint32](
				values,
				prefix+"attention.head_count_kv",
				gguf.ValueTypeUint32,
			); ok {
				spec.HeadCountKV = value
			}
		}
	} else if architecture == "deci" || architecture == "laguna" || architecture == "openelm" {
		if spec.LayerKVHeadCounts, err = requiredLayerUint32(
			values, prefix+"attention.head_count_kv", spec.BlockCount,
		); err != nil {
			return Spec{}, err
		}
		spec.HeadCountKV = firstPositive(spec.LayerKVHeadCounts)
	} else if architecture == "lfm2" || architecture == "lfm2moe" {
		counts, countErr := requiredArray[uint32](
			values, prefix+"attention.head_count_kv", gguf.ValueTypeUint32,
		)
		if countErr != nil {
			return Spec{}, countErr
		}
		if len(counts) != int(spec.BlockCount) {
			return Spec{}, fmt.Errorf(
				"metadata %q has %d values, need %d",
				prefix+"attention.head_count_kv", len(counts), spec.BlockCount,
			)
		}
		spec.RecurrentLayers = make([]bool, len(counts))
		for index, count := range counts {
			if count == 0 {
				spec.RecurrentLayers[index] = true
				continue
			}
			if spec.HeadCountKV == 0 {
				spec.HeadCountKV = count
			} else if spec.HeadCountKV != count {
				return Spec{}, errors.New("LFM2 attention layers use differing positive KV head counts")
			}
		}
	} else {
		if spec.HeadCountKV, err = required[uint32](values, prefix+"attention.head_count_kv", gguf.ValueTypeUint32); err != nil {
			return Spec{}, err
		}
	}
	spec.KeyLength, _ = optional[uint32](values, prefix+"attention.key_length", gguf.ValueTypeUint32)
	spec.ValueLength, _ = optional[uint32](values, prefix+"attention.value_length", gguf.ValueTypeUint32)
	if spec.KeyLength == 0 || spec.ValueLength == 0 {
		if spec.HeadCount == 0 || spec.EmbeddingLength%spec.HeadCount != 0 {
			return Spec{}, errors.New("embedding length is not divisible by attention head count")
		}
		headLength := spec.EmbeddingLength / spec.HeadCount
		if spec.KeyLength == 0 {
			spec.KeyLength = headLength
		}
		if spec.ValueLength == 0 {
			spec.ValueLength = headLength
		}
	}
	if architecture == "bloom" || architecture == "gpt2" || architecture == "jais" || architecture == "mpt" ||
		architecture == "refact" || architecture == "starcoder" {
		// architectures use ALiBi or learned absolute rows instead of RoPE
		spec.RopeDisabled = true
	} else if architecture != "t5encoder" {
		if architecture == "gptneox" || architecture == "falcon" {
			spec.RopeFrequencyBase = 10000
			if value, ok := optional[float32](
				values,
				prefix+"rope.freq_base",
				gguf.ValueTypeFloat32,
			); ok {
				spec.RopeFrequencyBase = value
			}
		} else if spec.RopeFrequencyBase, err = required[float32](values, prefix+"rope.freq_base", gguf.ValueTypeFloat32); err != nil {
			return Spec{}, err
		}
		if scalingType, ok := optional[string](
			values,
			prefix+"rope.scaling.type",
			gguf.ValueTypeString,
		); ok && scalingType != "" && scalingType != "none" {
			if architecture == "qwen35" ||
				(scalingType != "linear" && !(supportsLongRoPE(architecture) && scalingType == "longrope")) {
				if (architecture != "laguna" && architecture != "grok" && architecture != "mellum") || scalingType != "yarn" {
					return Spec{}, fmt.Errorf(
						"model architecture %q uses unsupported RoPE scaling type %q",
						architecture,
						scalingType,
					)
				}
			}
			spec.RopeScalingType = scalingType
			if scalingType == "linear" || scalingType == "yarn" {
				if spec.RopeScalingFactor, err = required[float32](
					values,
					prefix+"rope.scaling.factor",
					gguf.ValueTypeFloat32,
				); err != nil {
					return Spec{}, err
				}
			}
			if scalingType == "yarn" {
				if spec.OriginalContextLength, err = required[uint32](
					values, prefix+"rope.scaling.original_context_length", gguf.ValueTypeUint32,
				); err != nil {
					return Spec{}, err
				}
				spec.YaRNExtFactor = 1
				// ggml's YaRN primitive applies its logarithmic magnitude factor
				// internally; llama.cpp cancels it in context parameters for
				// ordinary YaRN models, leaving rotation interpolation without
				// unintended residual-vector scale
				spec.YaRNAttentionFactor = 1 / (1 + 0.1*float32(math.Log(float64(spec.RopeScalingFactor))))
				spec.YaRNBetaFast = 32
				if architecture == "grok" {
					spec.YaRNBetaFast = 8
				}
				spec.YaRNBetaSlow = 1
				for key, destination := range map[string]*float32{
					"rope.scaling.yarn_ext_factor":  &spec.YaRNExtFactor,
					"rope.scaling.yarn_attn_factor": &spec.YaRNAttentionFactor,
					"rope.scaling.yarn_beta_fast":   &spec.YaRNBetaFast,
					"rope.scaling.yarn_beta_slow":   &spec.YaRNBetaSlow,
				} {
					if value, ok := optional[float32](values, prefix+key, gguf.ValueTypeFloat32); ok {
						*destination = value
					}
				}
			}
		}
	}
	if architecture == "bloom" || architecture == "jais" || architecture == "mpt" || architecture == "refact" {
		if architecture != "mpt" {
			spec.MaxALiBiBias = 8
		}
		if value, ok := optional[float32](
			values, prefix+"attention.max_alibi_bias", gguf.ValueTypeFloat32,
		); ok {
			spec.MaxALiBiBias = value
		}
	}
	if architecture == "mpt" {
		if clamp, ok := optional[float32](
			values, prefix+"attention.clamp_kqv", gguf.ValueTypeFloat32,
		); ok && clamp != 0 {
			return Spec{}, errors.New("MPT attention QKV clamping is not supported")
		}
	}
	if architecture == "dbrx" {
		if spec.AttentionClamp, err = required[float32](
			values, prefix+"attention.clamp_kqv", gguf.ValueTypeFloat32,
		); err != nil {
			return Spec{}, err
		}
	}
	if architecture == "refact" {
		if expertCount, ok := optional[uint32](
			values, prefix+"expert_count", gguf.ValueTypeUint32,
		); ok && expertCount > 0 {
			return Spec{}, errors.New("Refact expert layers are not supported")
		}
	}
	if spec.UsesLayerNorm() || spec.UsesWeightOnlyLayerNorm() || spec.UsesUnweightedLayerNorm() {
		if spec.LayerNormEpsilon, err = required[float32](
			values,
			prefix+"attention.layer_norm_epsilon",
			gguf.ValueTypeFloat32,
		); err != nil {
			return Spec{}, err
		}
	} else {
		if spec.RMSNormEpsilon, err = required[float32](
			values,
			prefix+"attention.layer_norm_rms_epsilon",
			gguf.ValueTypeFloat32,
		); err != nil {
			return Spec{}, err
		}
	}
	spec.FinalLogitSoftcap, _ = optional[float32](
		values,
		prefix+"final_logit_softcapping",
		gguf.ValueTypeFloat32,
	)
	spec.AttentionSoftcap, _ = optional[float32](
		values,
		prefix+"attn_logit_softcapping",
		gguf.ValueTypeFloat32,
	)
	if architecture == "jais" {
		spec.AttentionScale = 1 / float32(spec.KeyLength)
	}
	if value, ok := optional[float32](values, prefix+"attention.scale", gguf.ValueTypeFloat32); ok {
		spec.AttentionScale = value
	}
	if architecture == "grok" {
		spec.AttentionSoftcap = 30
		if value, ok := optional[float32](values, prefix+"attn_logit_softcapping", gguf.ValueTypeFloat32); ok {
			spec.AttentionSoftcap = value
		}
		spec.AttentionScale = 0.08838834764831845
		if value, ok := optional[float32](values, prefix+"attention.output_scale", gguf.ValueTypeFloat32); ok {
			spec.AttentionScale = value
		}
		spec.EmbeddingScale = 78.38367176906169
		if value, ok := optional[float32](values, prefix+"embedding_scale", gguf.ValueTypeFloat32); ok {
			spec.EmbeddingScale = value
		}
		spec.LogitScale = 0.5773502691896257
		if value, ok := optional[float32](values, prefix+"logit_scale", gguf.ValueTypeFloat32); ok {
			spec.LogitScale = value
		}
		spec.RopeDimensionCount = spec.KeyLength
		if value, ok := optional[uint32](
			values, prefix+"rope.dimension_count", gguf.ValueTypeUint32,
		); ok {
			spec.RopeDimensionCount = value
		}
	}
	if architecture == "minicpm" {
		spec.EmbeddingScale = 12
		spec.ResidualScale = float32(1.4 / math.Sqrt(float64(spec.BlockCount)))
		spec.LogitScale = 256 / float32(spec.EmbeddingLength)
		if value, ok := optional[float32](
			values,
			prefix+"embedding_scale",
			gguf.ValueTypeFloat32,
		); ok {
			spec.EmbeddingScale = value
		}
		if value, ok := optional[float32](
			values,
			prefix+"residual_scale",
			gguf.ValueTypeFloat32,
		); ok {
			spec.ResidualScale = value
		}
		if value, ok := optional[float32](
			values,
			prefix+"logit_scale",
			gguf.ValueTypeFloat32,
		); ok {
			spec.LogitScale = value
		}
	}
	if architecture == "granite" || architecture == "granitemoe" {
		if spec.LogitScale, err = required[float32](
			values,
			prefix+"logit_scale",
			gguf.ValueTypeFloat32,
		); err != nil {
			return Spec{}, err
		}
		spec.EmbeddingScale, _ = optional[float32](
			values,
			prefix+"embedding_scale",
			gguf.ValueTypeFloat32,
		)
		spec.ResidualScale, _ = optional[float32](
			values,
			prefix+"residual_scale",
			gguf.ValueTypeFloat32,
		)
		spec.RopeDimensionCount = spec.KeyLength
		spec.RopeDimensionCount, _ = optional[uint32](
			values, prefix+"rope.dimension_count", gguf.ValueTypeUint32,
		)
		if spec.RopeDimensionCount == 0 {
			spec.RopeDimensionCount = spec.KeyLength
		}
		spec.OriginalContextLength = spec.ContextLength
		spec.OriginalContextLength, _ = optional[uint32](
			values, prefix+"rope.scaling.original_context_length", gguf.ValueTypeUint32,
		)
		if spec.OriginalContextLength == 0 {
			spec.OriginalContextLength = spec.ContextLength
		}
		spec.RopeAttentionFactor = 1
		spec.RopeAttentionFactor, _ = optional[float32](
			values, prefix+"rope.scaling.attn_factor", gguf.ValueTypeFloat32,
		)
		if spec.RopeAttentionFactor == 0 {
			spec.RopeAttentionFactor = 1
		}
		ropeEnabled := true
		if value, ok := optional[bool](
			values,
			prefix+"rope.scaling.finetuned",
			gguf.ValueTypeBool,
		); ok {
			ropeEnabled = value
		}
		spec.RopeDisabled = !ropeEnabled
		if architecture == "granite" {
			if expertCount, ok := optional[uint32](
				values,
				prefix+"expert_count",
				gguf.ValueTypeUint32,
			); ok && expertCount > 0 {
				return Spec{}, errors.New("Granite expert layers are not supported")
			}
		}
		if mapping, ok, mappingErr := optionalArray[int32](
			values,
			prefix+"deepstack_mapping",
			gguf.ValueTypeInt32,
		); mappingErr != nil {
			return Spec{}, mappingErr
		} else if ok && len(mapping) > 0 {
			return Spec{}, errors.New("Granite vision deepstack is not supported")
		}
	}
	if architecture == "cohere2" {
		if spec.LogitScale, err = required[float32](
			values,
			prefix+"logit_scale",
			gguf.ValueTypeFloat32,
		); err != nil {
			return Spec{}, err
		}
		if spec.RopeDimensionCount, err = required[uint32](
			values,
			prefix+"rope.dimension_count",
			gguf.ValueTypeUint32,
		); err != nil {
			return Spec{}, err
		}
		if pattern, ok := values[prefix+"attention.sliding_window_pattern"]; ok &&
			pattern.Type != gguf.ValueTypeUint32 {
			return Spec{}, errors.New("Cohere2 array sliding attention patterns are not supported")
		}
	}
	if architecture == "stablelm" {
		if spec.RopeDimensionCount, err = required[uint32](
			values,
			prefix+"rope.dimension_count",
			gguf.ValueTypeUint32,
		); err != nil {
			return Spec{}, err
		}
	}
	if architecture == "phi2" {
		if spec.RopeDimensionCount, err = required[uint32](
			values,
			prefix+"rope.dimension_count",
			gguf.ValueTypeUint32,
		); err != nil {
			return Spec{}, err
		}
	}
	if architecture == "phi3" || architecture == "phimoe" {
		if spec.RopeDimensionCount, err = required[uint32](
			values, prefix+"rope.dimension_count", gguf.ValueTypeUint32,
		); err != nil {
			return Spec{}, err
		}
		if spec.OriginalContextLength, err = required[uint32](
			values, prefix+"rope.scaling.original_context_length", gguf.ValueTypeUint32,
		); err != nil {
			return Spec{}, err
		}
		spec.RopeAttentionFactor = 1
		if value, ok := optional[float32](
			values, prefix+"rope.scaling.attn_factor", gguf.ValueTypeFloat32,
		); ok {
			spec.RopeAttentionFactor = value
		}
	}
	if architecture == "deci" {
		spec.RopeDimensionCount = spec.KeyLength
		if value, ok := optional[uint32](
			values, prefix+"rope.dimension_count", gguf.ValueTypeUint32,
		); ok {
			spec.RopeDimensionCount = value
		}
		spec.OriginalContextLength = spec.ContextLength
		if value, ok := optional[uint32](
			values, prefix+"rope.scaling.original_context_length", gguf.ValueTypeUint32,
		); ok {
			spec.OriginalContextLength = value
		}
		spec.RopeAttentionFactor = 1
		if value, ok := optional[float32](
			values, prefix+"rope.scaling.attn_factor", gguf.ValueTypeFloat32,
		); ok {
			spec.RopeAttentionFactor = value
		}
	}
	if architecture == "apertus" {
		spec.RopeDimensionCount = spec.KeyLength
		if value, ok := optional[uint32](
			values, prefix+"rope.dimension_count", gguf.ValueTypeUint32,
		); ok {
			spec.RopeDimensionCount = value
		}
		spec.OriginalContextLength = spec.ContextLength
		if value, ok := optional[uint32](
			values, prefix+"rope.scaling.original_context_length", gguf.ValueTypeUint32,
		); ok {
			spec.OriginalContextLength = value
		}
		spec.RopeAttentionFactor = 1
		if value, ok := optional[float32](
			values, prefix+"rope.scaling.attn_factor", gguf.ValueTypeFloat32,
		); ok {
			spec.RopeAttentionFactor = value
		}
		for key, destination := range map[string]*[]float32{
			"xielu.alpha_n": &spec.XIELUAlphaN,
			"xielu.alpha_p": &spec.XIELUAlphaP,
			"xielu.beta":    &spec.XIELUBeta,
			"xielu.eps":     &spec.XIELUEpsilon,
		} {
			*destination, err = requiredLayerFloat32(values, prefix+key, spec.BlockCount)
			if err != nil {
				return Spec{}, err
			}
		}
	}
	if architecture == "gptneox" {
		spec.RopeDimensionCount, _ = optional[uint32](
			values,
			prefix+"rope.dimension_count",
			gguf.ValueTypeUint32,
		)
		if spec.ParallelResidual, err = required[bool](
			values,
			prefix+"use_parallel_residual",
			gguf.ValueTypeBool,
		); err != nil {
			return Spec{}, err
		}
	}
	if architecture == "glm4" {
		spec.RopeDimensionCount = spec.KeyLength
		if value, ok := optional[uint32](
			values, prefix+"rope.dimension_count", gguf.ValueTypeUint32,
		); ok {
			spec.RopeDimensionCount = value
		}
		if nextN, ok := optional[uint32](
			values, prefix+"nextn_predict_layers", gguf.ValueTypeUint32,
		); ok && nextN > 0 {
			return Spec{}, errors.New("GLM4 NextN/MTP layers are not supported")
		}
		if sections, ok, sectionsErr := optionalArray[int32](
			values, prefix+"rope.dimension_sections", gguf.ValueTypeInt32,
		); sectionsErr != nil {
			return Spec{}, sectionsErr
		} else if ok {
			if len(sections) != 4 {
				return Spec{}, fmt.Errorf("metadata %q has %d values, need 4", prefix+"rope.dimension_sections", len(sections))
			}
			if sections[0] > 0 && sections[1] > 0 {
				return Spec{}, errors.New("GLM4 multimodal RoPE is not supported")
			}
		}
	}
	if architecture == "exaone4" {
		spec.RopeDimensionCount = spec.KeyLength
		if value, ok := optional[uint32](
			values, prefix+"rope.dimension_count", gguf.ValueTypeUint32,
		); ok {
			spec.RopeDimensionCount = value
		}
		if nextN, ok := optional[uint32](
			values, prefix+"nextn_predict_layers", gguf.ValueTypeUint32,
		); ok && nextN > 0 {
			return Spec{}, errors.New("EXAONE 4 NextN/MTP layers are not supported")
		}
		spec.RopeFrequencySWA = spec.RopeFrequencyBase
		if value, ok := optional[float32](
			values, prefix+"rope.freq_base_swa", gguf.ValueTypeFloat32,
		); ok {
			spec.RopeFrequencySWA = value
		}
		if spec.BlockCount == 64 {
			spec.SlidingWindow = 4096
		}
		if value, ok := optional[uint32](
			values, prefix+"attention.sliding_window", gguf.ValueTypeUint32,
		); ok {
			spec.SlidingWindow = value
		}
		if spec.SlidingWindow > 0 {
			spec.SlidingPattern = 4
			if value, ok := optional[uint32](
				values, prefix+"attention.sliding_window_pattern", gguf.ValueTypeUint32,
			); ok {
				spec.SlidingPattern = value
			}
			spec.NoRopeLayerStep = spec.SlidingPattern
		}
	}
	if architecture == "exaone-moe" {
		spec.RopeDimensionCount = spec.KeyLength
		if value, ok := optional[uint32](values, prefix+"rope.dimension_count", gguf.ValueTypeUint32); ok {
			spec.RopeDimensionCount = value
		}
		if nextN, ok := optional[uint32](values, prefix+"nextn_predict_layers", gguf.ValueTypeUint32); ok && nextN > 0 {
			if nextN >= spec.BlockCount {
				return Spec{}, errors.New("EXAONE-MoE NextN/MTP layer count is invalid")
			}
			spec.BlockCount -= nextN
		}
		spec.RopeFrequencySWA = spec.RopeFrequencyBase
		if value, ok := optional[float32](values, prefix+"rope.freq_base_swa", gguf.ValueTypeFloat32); ok {
			spec.RopeFrequencySWA = value
		}
		if spec.SlidingWindow, err = required[uint32](values, prefix+"attention.sliding_window", gguf.ValueTypeUint32); err != nil {
			return Spec{}, err
		}
		spec.SlidingPattern = 4
		if value, ok := optional[uint32](values, prefix+"attention.sliding_window_pattern", gguf.ValueTypeUint32); ok {
			spec.SlidingPattern = value
		} else if layers, ok, arrayErr := optionalArray[bool](values, prefix+"attention.sliding_window_pattern", gguf.ValueTypeBool); arrayErr != nil {
			return Spec{}, arrayErr
		} else if ok {
			if len(layers) != int(spec.BlockCount) {
				return Spec{}, fmt.Errorf("metadata %q has %d values, need %d", prefix+"attention.sliding_window_pattern", len(layers), spec.BlockCount)
			}
			spec.SlidingLayers = append([]bool(nil), layers...)
		}
	}
	if architecture == "bailingmoe2" {
		spec.RopeDimensionCount = spec.KeyLength
		if value, ok := optional[uint32](values, prefix+"rope.dimension_count", gguf.ValueTypeUint32); ok {
			spec.RopeDimensionCount = value
		}
		if nextN, ok := optional[uint32](values, prefix+"nextn_predict_layers", gguf.ValueTypeUint32); ok && nextN > 0 {
			if nextN >= spec.BlockCount {
				return Spec{}, errors.New("BailingMoE2 NextN/MTP layer count is invalid")
			}
			spec.BlockCount -= nextN
		}
	}
	if architecture == "falcon" {
		spec.RopeDimensionCount, _ = optional[uint32](
			values,
			prefix+"rope.dimension_count",
			gguf.ValueTypeUint32,
		)
	}
	if architecture == "command-r" {
		spec.LogitScale, _ = optional[float32](
			values,
			prefix+"logit_scale",
			gguf.ValueTypeFloat32,
		)
	}
	if architecture == "baichuan" && spec.BlockCount != 32 {
		return Spec{}, errors.New("only the 32-layer Baichuan RoPE variant is supported")
	}
	if architecture == "mistral3" {
		if temperatureScale, ok := optional[float32](
			values,
			prefix+"attention.temperature_scale",
			gguf.ValueTypeFloat32,
		); ok && temperatureScale != 0 {
			return Spec{}, errors.New("Mistral 3 attention temperature scaling is not supported")
		}
		if expertCount, ok := optional[uint32](
			values,
			prefix+"expert_count",
			gguf.ValueTypeUint32,
		); ok && expertCount > 0 {
			return Spec{}, errors.New("Mistral 3 expert layers are not supported")
		}
	}
	if architecture == "smollm3" {
		spec.NoRopeLayerStep = 4
	}
	if architecture == "smallthinker" {
		spec.RopeDimensionCount = spec.KeyLength
		spec.NoRopeLayerStep = spec.BlockCount
		if value, ok := optional[uint32](
			values, prefix+"attention.sliding_window", gguf.ValueTypeUint32,
		); ok && value > 0 {
			spec.SlidingWindow = 4096
			spec.SlidingPattern = 4
			if pattern, patternOK := optional[uint32](
				values, prefix+"attention.sliding_window_pattern", gguf.ValueTypeUint32,
			); patternOK {
				spec.SlidingPattern = pattern
			}
			spec.NoRopeLayerStep = spec.SlidingPattern
			spec.RopeFrequencySWA = spec.RopeFrequencyBase
			if frequency, frequencyOK := optional[float32](
				values, prefix+"rope.freq_base_swa", gguf.ValueTypeFloat32,
			); frequencyOK {
				spec.RopeFrequencySWA = frequency
			}
		}
	}
	if architecture == "mellum" {
		spec.RopeDimensionCount = spec.KeyLength
		if value, ok := optional[uint32](values, prefix+"rope.dimension_count", gguf.ValueTypeUint32); ok {
			spec.RopeDimensionCount = value
		}
		if value, ok := optional[uint32](values, prefix+"attention.sliding_window", gguf.ValueTypeUint32); ok {
			spec.SlidingWindow = value
		}
		if spec.SlidingWindow > 0 {
			spec.SlidingPattern = 4
			if value, ok := optional[uint32](values, prefix+"attention.sliding_window_pattern", gguf.ValueTypeUint32); ok {
				spec.SlidingPattern = value
			} else if layers, ok, arrayErr := optionalArray[bool](values, prefix+"attention.sliding_window_pattern", gguf.ValueTypeBool); arrayErr != nil {
				return Spec{}, arrayErr
			} else if ok {
				if len(layers) != int(spec.BlockCount) {
					return Spec{}, fmt.Errorf("metadata %q has %d values, need %d", prefix+"attention.sliding_window_pattern", len(layers), spec.BlockCount)
				}
				spec.SlidingLayers = append([]bool(nil), layers...)
			}
			spec.RopeFrequencySWA = spec.RopeFrequencyBase
			if value, ok := optional[float32](values, prefix+"rope.freq_base_swa", gguf.ValueTypeFloat32); ok {
				spec.RopeFrequencySWA = value
			}
		}
	}
	if architecture == "afmoe" {
		spec.NoRopeLayerStep = 4
	}
	if architecture == "olmo" {
		if clamp, ok := optional[float32](
			values,
			prefix+"attention.clamp_kqv",
			gguf.ValueTypeFloat32,
		); ok && clamp != 0 {
			return Spec{}, errors.New("OLMo attention QKV clamping is not supported")
		}
	}
	if architecture == "t5encoder" {
		if spec.RelativeBuckets, err = required[uint32](
			values,
			prefix+"attention.relative_buckets_count",
			gguf.ValueTypeUint32,
		); err != nil {
			return Spec{}, err
		}
	}
	if architecture == "qwen35" {
		if spec.RopeDimensionCount, err = required[uint32](
			values,
			prefix+"rope.dimension_count",
			gguf.ValueTypeUint32,
		); err != nil {
			return Spec{}, err
		}
		sections, sectionsErr := requiredArray[int32](
			values,
			prefix+"rope.dimension_sections",
			gguf.ValueTypeInt32,
		)
		if sectionsErr != nil {
			return Spec{}, sectionsErr
		}
		if len(sections) != len(spec.RopeSections) {
			return Spec{}, fmt.Errorf(
				"metadata %q has %d values, need %d",
				prefix+"rope.dimension_sections",
				len(sections),
				len(spec.RopeSections),
			)
		}
		copy(spec.RopeSections[:], sections)
		for key, destination := range map[string]*uint32{
			"ssm.conv_kernel":    &spec.SSMConvKernel,
			"ssm.inner_size":     &spec.SSMInnerSize,
			"ssm.state_size":     &spec.SSMStateSize,
			"ssm.time_step_rank": &spec.SSMTimeStepRank,
			"ssm.group_count":    &spec.SSMGroupCount,
		} {
			*destination, err = required[uint32](values, prefix+key, gguf.ValueTypeUint32)
			if err != nil {
				return Spec{}, err
			}
		}
		spec.FullAttentionInterval = 4
		if interval, ok := optional[uint32](
			values,
			prefix+"full_attention_interval",
			gguf.ValueTypeUint32,
		); ok {
			spec.FullAttentionInterval = interval
		}
		if recurrent, ok, recurrentErr := optionalArray[bool](
			values,
			prefix+"attention.recurrent_layers",
			gguf.ValueTypeBool,
		); recurrentErr != nil {
			return Spec{}, recurrentErr
		} else if ok {
			if len(recurrent) != int(spec.BlockCount) {
				return Spec{}, fmt.Errorf(
					"metadata %q has %d values, need %d",
					prefix+"attention.recurrent_layers",
					len(recurrent),
					spec.BlockCount,
				)
			}
			spec.RecurrentLayers = append([]bool(nil), recurrent...)
		}
	}
	if isLlamaMoE || architecture == "arctic" || architecture == "bailingmoe" || architecture == "bailingmoe2" || architecture == "deepseek" || architecture == "dbrx" || architecture == "dots1" || architecture == "granitemoe" || architecture == "grok" || architecture == "hunyuan-moe" || architecture == "llada-moe" || architecture == "mellum" || architecture == "minimax-m2" || architecture == "qwen3moe" || architecture == "qwen2moe" || architecture == "olmoe" || architecture == "phimoe" || architecture == "exaone-moe" || architecture == "rnd1" || architecture == "afmoe" || architecture == "laguna" || architecture == "lfm2moe" || architecture == "smallthinker" {
		if spec.ExpertCount, err = required[uint32](
			values, prefix+"expert_count", gguf.ValueTypeUint32,
		); err != nil {
			return Spec{}, err
		}
		if spec.ExpertUsedCount, err = required[uint32](
			values, prefix+"expert_used_count", gguf.ValueTypeUint32,
		); err != nil {
			return Spec{}, err
		}
		if spec.ExpertUsedCount == 0 {
			return Spec{}, errors.New("expert used count is zero")
		}
		spec.ExpertFeedForward = spec.FeedForwardLength / spec.ExpertUsedCount
		if value, ok := optional[uint32](
			values, prefix+"expert_feed_forward_length", gguf.ValueTypeUint32,
		); ok {
			spec.ExpertFeedForward = value
		}
		if architecture == "arctic" {
			spec.ExpertFeedForward = spec.FeedForwardLength
		}
		spec.ExpertWeightsScale = 1
		if value, ok := optional[float32](
			values, prefix+"expert_weights_scale", gguf.ValueTypeFloat32,
		); ok {
			spec.ExpertWeightsScale = value
		}
	}
	if architecture == "mellum" {
		if spec.ExpertFeedForward, err = required[uint32](values, prefix+"expert_feed_forward_length", gguf.ValueTypeUint32); err != nil {
			return Spec{}, err
		}
		spec.ExpertWeightsNorm = true
	}
	if architecture == "hunyuan-moe" {
		if spec.ExpertFeedForward, err = required[uint32](values, prefix+"expert_feed_forward_length", gguf.ValueTypeUint32); err != nil {
			return Spec{}, err
		}
		spec.SharedExpertFF = spec.FeedForwardLength
		if value, ok := optional[uint32](values, prefix+"expert_shared_feed_forward_length", gguf.ValueTypeUint32); ok {
			spec.SharedExpertFF = value
		}
		spec.ExpertWeightsNorm = true
		spec.RopeDimensionCount = spec.KeyLength
	}
	if architecture == "grok" {
		if _, ok := optional[uint32](
			values, prefix+"expert_feed_forward_length", gguf.ValueTypeUint32,
		); !ok {
			spec.ExpertFeedForward = spec.FeedForwardLength
		}
		spec.ExpertWeightsNorm = true
	}
	if architecture == "dbrx" {
		spec.ExpertFeedForward = spec.FeedForwardLength
		spec.ExpertWeightsNorm = true
	}
	if architecture == "granitemoe" {
		spec.ExpertFeedForward = spec.FeedForwardLength
		spec.ExpertWeightsNorm = true
		spec.SharedExpertFF, _ = optional[uint32](
			values, prefix+"expert_shared_feed_forward_length", gguf.ValueTypeUint32,
		)
	}
	if architecture == "smallthinker" {
		spec.ExpertFeedForward = spec.FeedForwardLength
		spec.ExpertWeightsNorm = true
		if spec.ExpertGatingFunc, err = required[uint32](
			values, prefix+"expert_gating_func", gguf.ValueTypeUint32,
		); err != nil {
			return Spec{}, err
		}
	}
	if architecture == "dots1" {
		if spec.ExpertFeedForward, err = required[uint32](
			values, prefix+"expert_feed_forward_length", gguf.ValueTypeUint32,
		); err != nil {
			return Spec{}, err
		}
		if spec.SharedExpertCount, err = required[uint32](
			values, prefix+"expert_shared_count", gguf.ValueTypeUint32,
		); err != nil {
			return Spec{}, err
		}
		spec.SharedExpertFF = spec.ExpertFeedForward * spec.SharedExpertCount
		if spec.ExpertGatingFunc, err = required[uint32](
			values, prefix+"expert_gating_func", gguf.ValueTypeUint32,
		); err != nil {
			return Spec{}, err
		}
		spec.ExpertWeightsNorm, _ = optional[bool](
			values, prefix+"expert_weights_norm", gguf.ValueTypeBool,
		)
		spec.LeadingDenseBlocks, _ = optional[uint32](
			values, prefix+"leading_dense_block_count", gguf.ValueTypeUint32,
		)
	}
	if architecture == "minimax-m2" {
		expertWidth, widthErr := required[uint32](
			values, prefix+"expert_feed_forward_length", gguf.ValueTypeUint32,
		)
		if widthErr != nil {
			return Spec{}, widthErr
		}
		if expertWidth != spec.FeedForwardLength {
			return Spec{}, errors.New("MiniMax-M2 expert width differs from packed tensor width")
		}
		spec.ExpertFeedForward = spec.FeedForwardLength
		spec.ExpertWeightsNorm = true
		if spec.ExpertGatingFunc, err = required[uint32](
			values, prefix+"expert_gating_func", gguf.ValueTypeUint32,
		); err != nil {
			return Spec{}, err
		}
		if spec.RopeDimensionCount, err = required[uint32](
			values, prefix+"rope.dimension_count", gguf.ValueTypeUint32,
		); err != nil {
			return Spec{}, err
		}
	}
	if architecture == "bailingmoe" {
		if spec.ExpertFeedForward, err = required[uint32](
			values, prefix+"expert_feed_forward_length", gguf.ValueTypeUint32,
		); err != nil {
			return Spec{}, err
		}
		if spec.SharedExpertCount, err = required[uint32](
			values, prefix+"expert_shared_count", gguf.ValueTypeUint32,
		); err != nil {
			return Spec{}, err
		}
		spec.SharedExpertFF = spec.ExpertFeedForward * spec.SharedExpertCount
		spec.ExpertWeightsNorm, _ = optional[bool](values, prefix+"expert_weights_norm", gguf.ValueTypeBool)
		spec.LeadingDenseBlocks, _ = optional[uint32](values, prefix+"leading_dense_block_count", gguf.ValueTypeUint32)
	}
	if architecture == "deepseek" {
		if spec.ExpertFeedForward, err = required[uint32](
			values, prefix+"expert_feed_forward_length", gguf.ValueTypeUint32,
		); err != nil {
			return Spec{}, err
		}
		if spec.SharedExpertCount, err = required[uint32](
			values, prefix+"expert_shared_count", gguf.ValueTypeUint32,
		); err != nil {
			return Spec{}, err
		}
		spec.SharedExpertFF = spec.ExpertFeedForward * spec.SharedExpertCount
		spec.LeadingDenseBlocks, _ = optional[uint32](values, prefix+"leading_dense_block_count", gguf.ValueTypeUint32)
	}
	if architecture == "lfm2moe" {
		if spec.ExpertFeedForward, err = required[uint32](
			values, prefix+"expert_feed_forward_length", gguf.ValueTypeUint32,
		); err != nil {
			return Spec{}, err
		}
		if spec.ExpertGatingFunc, err = required[uint32](
			values, prefix+"expert_gating_func", gguf.ValueTypeUint32,
		); err != nil {
			return Spec{}, err
		}
		spec.LeadingDenseBlocks, _ = optional[uint32](values, prefix+"leading_dense_block_count", gguf.ValueTypeUint32)
	}
	if architecture == "bailingmoe2" {
		if spec.ExpertFeedForward, err = required[uint32](
			values, prefix+"expert_feed_forward_length", gguf.ValueTypeUint32,
		); err != nil {
			return Spec{}, err
		}
		if spec.SharedExpertCount, err = required[uint32](
			values, prefix+"expert_shared_count", gguf.ValueTypeUint32,
		); err != nil {
			return Spec{}, err
		}
		sharedWidth := spec.ExpertFeedForward
		if value, ok := optional[uint32](values, prefix+"expert_shared_feed_forward_length", gguf.ValueTypeUint32); ok {
			sharedWidth = value
		}
		spec.SharedExpertFF = sharedWidth * spec.SharedExpertCount
		if spec.ExpertGatingFunc, err = required[uint32](
			values, prefix+"expert_gating_func", gguf.ValueTypeUint32,
		); err != nil {
			return Spec{}, err
		}
		spec.ExpertWeightsNorm, _ = optional[bool](values, prefix+"expert_weights_norm", gguf.ValueTypeBool)
		spec.LeadingDenseBlocks, _ = optional[uint32](values, prefix+"leading_dense_block_count", gguf.ValueTypeUint32)
	}
	if architecture == "qwen2moe" {
		if spec.ExpertFeedForward == 0 {
			spec.ExpertFeedForward = spec.FeedForwardLength
		}
		spec.SharedExpertCount = 1
		spec.SharedExpertFF = spec.FeedForwardLength
		if value, ok := optional[uint32](values, prefix+"expert_shared_feed_forward_length", gguf.ValueTypeUint32); ok {
			spec.SharedExpertFF = value
		}
	}
	if isLlamaMoE || architecture == "olmoe" || architecture == "phimoe" {
		spec.ExpertFeedForward = spec.FeedForwardLength
	}
	if architecture == "afmoe" {
		if spec.ExpertFeedForward, err = required[uint32](
			values, prefix+"expert_feed_forward_length", gguf.ValueTypeUint32,
		); err != nil {
			return Spec{}, err
		}
		if value, ok := optional[uint32](values, prefix+"leading_dense_block_count", gguf.ValueTypeUint32); ok {
			spec.LeadingDenseBlocks = value
		}
		if spec.SharedExpertCount, err = required[uint32](
			values, prefix+"expert_shared_count", gguf.ValueTypeUint32,
		); err != nil {
			return Spec{}, err
		}
		spec.SharedExpertFF = spec.ExpertFeedForward * spec.SharedExpertCount
		spec.ExpertGatingFunc = 2
		if value, ok := optional[uint32](values, prefix+"expert_gating_func", gguf.ValueTypeUint32); ok && value != 0 {
			spec.ExpertGatingFunc = value
		}
		spec.ExpertWeightsNorm, _ = optional[bool](values, prefix+"expert_weights_norm", gguf.ValueTypeBool)
		spec.RopeDimensionCount = spec.KeyLength
		if value, ok := optional[uint32](values, prefix+"rope.dimension_count", gguf.ValueTypeUint32); ok {
			spec.RopeDimensionCount = value
		}
		if value, ok := optional[uint32](values, prefix+"attention.sliding_window", gguf.ValueTypeUint32); ok {
			spec.SlidingWindow = value
		}
		if spec.SlidingWindow > 0 {
			spec.SlidingPattern = 4
			if value, ok := optional[uint32](values, prefix+"attention.sliding_window_pattern", gguf.ValueTypeUint32); ok {
				spec.SlidingPattern = value
			}
			spec.RopeFrequencySWA = spec.RopeFrequencyBase
			if value, ok := optional[float32](values, prefix+"rope.freq_base_swa", gguf.ValueTypeFloat32); ok {
				spec.RopeFrequencySWA = value
			}
		}
	}
	if architecture == "exaone-moe" {
		if spec.ExpertFeedForward, err = required[uint32](values, prefix+"expert_feed_forward_length", gguf.ValueTypeUint32); err != nil {
			return Spec{}, err
		}
		spec.SharedExpertFF = spec.ExpertFeedForward
		if value, ok := optional[uint32](values, prefix+"expert_shared_feed_forward_length", gguf.ValueTypeUint32); ok {
			spec.SharedExpertFF = value
		}
		spec.SharedExpertCount, _ = optional[uint32](values, prefix+"expert_shared_count", gguf.ValueTypeUint32)
		spec.LeadingDenseBlocks, _ = optional[uint32](values, prefix+"leading_dense_block_count", gguf.ValueTypeUint32)
		if spec.ExpertGatingFunc, err = required[uint32](values, prefix+"expert_gating_func", gguf.ValueTypeUint32); err != nil {
			return Spec{}, err
		}
		spec.ExpertWeightsNorm, _ = optional[bool](values, prefix+"expert_weights_norm", gguf.ValueTypeBool)
	}
	if architecture == "laguna" {
		if spec.ExpertFeedForward, err = required[uint32](
			values, prefix+"expert_feed_forward_length", gguf.ValueTypeUint32,
		); err != nil {
			return Spec{}, err
		}
		if spec.LeadingDenseBlocks, err = required[uint32](
			values, prefix+"leading_dense_block_count", gguf.ValueTypeUint32,
		); err != nil {
			return Spec{}, err
		}
		if spec.SharedExpertFF, err = required[uint32](
			values, prefix+"expert_shared_feed_forward_length", gguf.ValueTypeUint32,
		); err != nil {
			return Spec{}, err
		}
		spec.ExpertGatingFunc = 2
		spec.ExpertGatingFunc, _ = optional[uint32](values, prefix+"expert_gating_func", gguf.ValueTypeUint32)
		if spec.ExpertGatingFunc == 0 {
			spec.ExpertGatingFunc = 2
		}
		spec.ExpertWeightsNorm, _ = optional[bool](values, prefix+"expert_weights_norm", gguf.ValueTypeBool)
		sharedCount := uint32(1)
		if value, ok := optional[uint32](values, prefix+"expert_shared_count", gguf.ValueTypeUint32); ok {
			sharedCount = value
		}
		if sharedCount != 1 {
			return Spec{}, errors.New("Laguna requires exactly one shared expert")
		}
		spec.RopeDimensionCount = spec.KeyLength
		if value, ok := optional[uint32](values, prefix+"rope.dimension_count", gguf.ValueTypeUint32); ok {
			spec.RopeDimensionCount = value
		}
		if value, ok := optional[uint32](values, prefix+"attention.sliding_window", gguf.ValueTypeUint32); ok {
			spec.SlidingWindow = value
		}
		if spec.SlidingWindow > 0 {
			spec.SlidingPattern = 4
			if value, ok := optional[uint32](values, prefix+"attention.sliding_window_pattern", gguf.ValueTypeUint32); ok {
				spec.SlidingPattern = value
			}
			spec.RopeFrequencySWA = spec.RopeFrequencyBase
			if value, ok := optional[float32](values, prefix+"rope.freq_base_swa", gguf.ValueTypeFloat32); ok {
				spec.RopeFrequencySWA = value
			}
			spec.RopeDimensionSWA = spec.KeyLength
			if value, ok := optional[uint32](values, prefix+"rope.dimension_count_swa", gguf.ValueTypeUint32); ok {
				spec.RopeDimensionSWA = value
			}
		}
	}
	if architecture == "lfm2" || architecture == "lfm2moe" {
		if spec.ShortConvCacheLength, err = required[uint32](
			values, prefix+"shortconv.l_cache", gguf.ValueTypeUint32,
		); err != nil {
			return Spec{}, err
		}
		if value, ok := optional[uint32](
			values, prefix+"attention.sliding_window", gguf.ValueTypeUint32,
		); ok {
			spec.SlidingWindow = value
		}
	}
	if architecture == "plm" {
		if spec.KVLoRARank, err = required[uint32](
			values, prefix+"attention.kv_lora_rank", gguf.ValueTypeUint32,
		); err != nil {
			return Spec{}, err
		}
		if spec.RopeDimensionCount, err = required[uint32](
			values, prefix+"rope.dimension_count", gguf.ValueTypeUint32,
		); err != nil {
			return Spec{}, err
		}
	}
	if architecture == "gemma2" || architecture == "gemma3" ||
		architecture == "olmo2" || architecture == "cohere2" {
		spec.RopeFrequencySWA = spec.RopeFrequencyBase
		if architecture == "gemma3" {
			spec.RopeFrequencySWA = 10000
		}
		if value, ok := optional[float32](
			values,
			prefix+"rope.freq_base_swa",
			gguf.ValueTypeFloat32,
		); ok {
			spec.RopeFrequencySWA = value
		}
		if architecture == "gemma2" {
			spec.SlidingWindow = 4096
		}
		if value, ok := optional[uint32](
			values,
			prefix+"attention.sliding_window",
			gguf.ValueTypeUint32,
		); ok {
			spec.SlidingWindow = value
		}
		if spec.SlidingWindow > 0 {
			spec.SlidingPattern = 4
			if architecture == "gemma2" {
				spec.SlidingPattern = 2
			}
			if architecture == "gemma3" {
				spec.SlidingPattern = 6
			}
			if value, ok := optional[uint32](
				values,
				prefix+"attention.sliding_window_pattern",
				gguf.ValueTypeUint32,
			); ok {
				spec.SlidingPattern = value
			}
			if architecture == "cohere2" {
				spec.NoRopeLayerStep = spec.SlidingPattern
			}
		}
	}
	spec.VocabularySize, _ = optional[uint32](values, prefix+"vocab_size", gguf.ValueTypeUint32)
	if tokens, ok := values["tokenizer.ggml.tokens"]; ok && spec.VocabularySize == 0 {
		if tokens.Type != gguf.ValueTypeArray || tokens.ArrayType != gguf.ValueTypeString {
			return Spec{}, errors.New(`metadata "tokenizer.ggml.tokens" must be a string array`)
		}
		if tokens.Count() > int(^uint32(0)) {
			return Spec{}, errors.New("tokenizer vocabulary exceeds uint32")
		}
		spec.VocabularySize = uint32(tokens.Count())
	}
	if err := spec.validate(); err != nil {
		return Spec{}, err
	}
	return spec, nil
}

func (s Spec) IsRecurrentLayer(block uint32) bool {
	if block >= s.BlockCount {
		return false
	}
	if len(s.RecurrentLayers) == int(s.BlockCount) {
		return s.RecurrentLayers[block]
	}
	return s.Architecture == "qwen35" &&
		s.FullAttentionInterval > 0 &&
		(block+1)%s.FullAttentionInterval != 0
}

func (s Spec) IsSlidingLayer(block uint32) bool {
	if s.Architecture == "lfm2" || s.Architecture == "lfm2moe" {
		return block < s.BlockCount && s.SlidingWindow > 0 && !s.IsRecurrentLayer(block)
	}
	if s.Architecture == "laguna" || s.Architecture == "smallthinker" {
		return block < s.BlockCount &&
			s.SlidingWindow > 0 &&
			s.SlidingPattern > 0 &&
			block%s.SlidingPattern != 0
	}
	if block < uint32(len(s.SlidingLayers)) {
		return s.SlidingLayers[block]
	}
	return usesSlidingAttention(s.Architecture) &&
		block < s.BlockCount &&
		s.SlidingWindow > 0 &&
		s.SlidingPattern > 0 &&
		block%s.SlidingPattern < s.SlidingPattern-1
}

// LayerHeadCount: returns query-head count selected for layer; Laguna
// stores this metadata as either scalar or one value per layer
func (s Spec) LayerHeadCount(block uint32) uint32 {
	if block < uint32(len(s.LayerHeadCounts)) {
		return s.LayerHeadCounts[block]
	}
	return s.HeadCount
}

// LayerKVHeadCount: returns key/value-head count selected for layer
func (s Spec) LayerKVHeadCount(block uint32) uint32 {
	if block < uint32(len(s.LayerKVHeadCounts)) {
		return s.LayerKVHeadCounts[block]
	}
	return s.HeadCountKV
}

func (s Spec) LayerFeedForwardLength(block uint32) uint32 {
	if block < uint32(len(s.LayerFeedForward)) {
		return s.LayerFeedForward[block]
	}
	return s.FeedForwardLength
}

func (s Spec) UsesRoPE(block uint32) bool {
	if s.Architecture == "smallthinker" {
		return !s.RopeDisabled && block < s.BlockCount &&
			(s.SlidingWindow == 0 || s.NoRopeLayerStep == 0 || block%s.NoRopeLayerStep != 0)
	}
	if s.Architecture == "exaone-moe" {
		return s.IsSlidingLayer(block)
	}
	return !s.RopeDisabled &&
		(s.BlockCount == 0 || block < s.BlockCount) &&
		(s.NoRopeLayerStep == 0 || (block+1)%s.NoRopeLayerStep != 0)
}

func (s Spec) InputEmbeddingScale() float32 {
	if s.EmbeddingScale > 0 {
		return s.EmbeddingScale
	}
	if s.Architecture == "afmoe" || isGemmaArchitecture(s.Architecture) {
		return float32(math.Sqrt(float64(s.EmbeddingLength)))
	}
	return 1
}

func (s Spec) OutputLogitMultiplier() float32 {
	if s.LogitScale > 0 {
		if s.Architecture == "cohere2" || s.Architecture == "command-r" || s.Architecture == "grok" {
			return s.LogitScale
		}
		return 1 / s.LogitScale
	}
	return 1
}

func (s Spec) UsesLayerNorm() bool {
	return s.Architecture == "dbrx" ||
		s.Architecture == "falcon" ||
		s.Architecture == "jais" ||
		s.Architecture == "nemotron" ||
		s.Architecture == "jais2" ||
		s.Architecture == "orion" ||
		s.Architecture == "stablelm" ||
		s.Architecture == "mpt" ||
		usesSequentialGELU(s.Architecture)
}

func (s Spec) RequiresLayerNormBias() bool {
	return s.Architecture == "phimoe" ||
		(s.UsesLayerNorm() && s.Architecture != "dbrx" && s.Architecture != "mpt")
}

func (s Spec) UsesUnweightedLayerNorm() bool {
	return s.Architecture == "olmo"
}

func (s Spec) UsesWeightOnlyLayerNorm() bool {
	return s.Architecture == "cohere2" || s.Architecture == "command-r"
}

func (s Spec) validate() error {
	switch {
	case s.BlockCount == 0:
		return errors.New("model block count is zero")
	case s.ContextLength == 0:
		return errors.New("model context length is zero")
	case s.EmbeddingLength == 0:
		return errors.New("model embedding length is zero")
	case s.FeedForwardLength == 0:
		return errors.New("model feed-forward length is zero")
	case s.HeadCount == 0:
		return errors.New("model attention head count is zero")
	case s.HeadCountKV == 0:
		return errors.New("model KV head count is zero")
	case s.HeadCount%s.HeadCountKV != 0:
		return errors.New("attention head count is not divisible by KV head count")
	case s.KeyLength == 0 || s.ValueLength == 0:
		return errors.New("model attention key/value length is zero")
	case s.Architecture != "t5encoder" && !s.RopeDisabled && s.RopeFrequencyBase <= 0:
		return errors.New("model RoPE frequency base must be positive")
	case (s.UsesLayerNorm() || s.UsesWeightOnlyLayerNorm() || s.UsesUnweightedLayerNorm()) && s.LayerNormEpsilon <= 0:
		return errors.New("model LayerNorm epsilon must be positive")
	case !s.UsesLayerNorm() && !s.UsesWeightOnlyLayerNorm() && !s.UsesUnweightedLayerNorm() && s.RMSNormEpsilon <= 0:
		return errors.New("model RMSNorm epsilon must be positive")
	}
	if s.Architecture == "t5encoder" && s.RelativeBuckets == 0 {
		return errors.New("T5 encoder relative attention bucket count is zero")
	}
	if s.Architecture == "qwen35" {
		switch {
		case s.RopeDimensionCount == 0 || s.RopeDimensionCount > s.KeyLength ||
			s.RopeDimensionCount%2 != 0:
			return errors.New("Qwen3.5 rotary dimension count is invalid")
		case s.SSMConvKernel == 0:
			return errors.New("Qwen3.5 SSM convolution kernel is zero")
		case s.SSMInnerSize == 0:
			return errors.New("Qwen3.5 SSM inner size is zero")
		case s.SSMStateSize == 0:
			return errors.New("Qwen3.5 SSM state size is zero")
		case s.SSMTimeStepRank == 0 || s.SSMInnerSize%s.SSMTimeStepRank != 0:
			return errors.New("Qwen3.5 SSM inner size is not divisible by time-step rank")
		case s.SSMInnerSize/s.SSMTimeStepRank != s.SSMStateSize:
			return errors.New("Qwen3.5 SSM value-head width differs from state size")
		case s.SSMGroupCount == 0 || s.SSMTimeStepRank%s.SSMGroupCount != 0:
			return errors.New("Qwen3.5 SSM value heads are not divisible by key groups")
		case s.FullAttentionInterval == 0:
			return errors.New("Qwen3.5 full-attention interval is zero")
		}
		var sectionPairs int64
		for _, section := range s.RopeSections {
			if section < 0 {
				return errors.New("Qwen3.5 RoPE section count is negative")
			}
			sectionPairs += int64(section)
		}
		if sectionPairs == 0 || sectionPairs > int64(s.RopeDimensionCount/2) {
			return errors.New("Qwen3.5 RoPE sections exceed rotary pair count")
		}
	}
	if (s.Architecture == "qwen3moe" || s.Architecture == "rnd1") &&
		(s.ExpertCount == 0 || s.ExpertUsedCount == 0 ||
			s.ExpertUsedCount > s.ExpertCount || s.ExpertUsedCount > 16 ||
			s.ExpertFeedForward == 0 || s.ExpertWeightsScale == 0 ||
			math.IsNaN(float64(s.ExpertWeightsScale)) ||
			math.IsInf(float64(s.ExpertWeightsScale), 0)) {
		return errors.New("Qwen3-MoE expert metadata is invalid")
	}
	if s.Architecture == "llada-moe" &&
		(s.ExpertCount == 0 || s.ExpertUsedCount == 0 || s.ExpertUsedCount > s.ExpertCount ||
			s.ExpertUsedCount > 16 || s.ExpertFeedForward == 0 || s.ExpertWeightsScale <= 0 ||
			math.IsNaN(float64(s.ExpertWeightsScale)) || math.IsInf(float64(s.ExpertWeightsScale), 0)) {
		return errors.New("LLaDA-MoE expert metadata is invalid")
	}
	if s.Architecture == "qwen2moe" &&
		(s.ExpertCount == 0 || s.ExpertUsedCount == 0 || s.ExpertUsedCount > s.ExpertCount ||
			s.ExpertUsedCount > 16 || s.ExpertFeedForward == 0 || s.SharedExpertFF == 0 ||
			s.ExpertWeightsScale <= 0 || math.IsNaN(float64(s.ExpertWeightsScale)) ||
			math.IsInf(float64(s.ExpertWeightsScale), 0)) {
		return errors.New("Qwen2-MoE expert metadata is invalid")
	}
	if s.Architecture == "arctic" &&
		(s.ExpertCount == 0 || s.ExpertUsedCount == 0 || s.ExpertUsedCount > s.ExpertCount ||
			s.ExpertUsedCount > 16 || s.ExpertFeedForward == 0 || s.ExpertWeightsScale <= 0 ||
			math.IsNaN(float64(s.ExpertWeightsScale)) || math.IsInf(float64(s.ExpertWeightsScale), 0)) {
		return errors.New("Arctic expert metadata is invalid")
	}
	if s.Architecture == "bailingmoe" {
		switch {
		case s.ExpertCount == 0 || s.ExpertUsedCount == 0 || s.ExpertUsedCount > s.ExpertCount ||
			s.ExpertUsedCount > 16 || s.ExpertFeedForward == 0 || s.SharedExpertCount == 0 ||
			s.SharedExpertFF == 0:
			return errors.New("BailingMoE expert metadata is invalid")
		case s.SharedExpertFF/s.SharedExpertCount != s.ExpertFeedForward:
			return errors.New("BailingMoE shared expert width overflows")
		case s.ExpertWeightsScale <= 0 || math.IsNaN(float64(s.ExpertWeightsScale)) ||
			math.IsInf(float64(s.ExpertWeightsScale), 0):
			return errors.New("BailingMoE expert weight scale is invalid")
		}
	}
	if s.Architecture == "deepseek" {
		switch {
		case s.LeadingDenseBlocks >= s.BlockCount:
			return errors.New("DeepSeek leading dense block count leaves no MoE layers")
		case s.ExpertCount == 0 || s.ExpertUsedCount == 0 || s.ExpertUsedCount > s.ExpertCount ||
			s.ExpertUsedCount > 16 || s.ExpertFeedForward == 0 || s.SharedExpertCount == 0 ||
			s.SharedExpertFF == 0:
			return errors.New("DeepSeek expert metadata is invalid")
		case s.SharedExpertFF/s.SharedExpertCount != s.ExpertFeedForward:
			return errors.New("DeepSeek shared expert width overflows")
		case s.ExpertWeightsScale <= 0 || math.IsNaN(float64(s.ExpertWeightsScale)) ||
			math.IsInf(float64(s.ExpertWeightsScale), 0):
			return errors.New("DeepSeek expert weight scale is invalid")
		}
	}
	if s.Architecture == "granitemoe" &&
		(s.ExpertCount == 0 || s.ExpertUsedCount == 0 || s.ExpertUsedCount > s.ExpertCount ||
			s.ExpertUsedCount > 16 || s.ExpertFeedForward == 0 || s.ExpertWeightsScale <= 0 ||
			math.IsNaN(float64(s.ExpertWeightsScale)) || math.IsInf(float64(s.ExpertWeightsScale), 0)) {
		return errors.New("GraniteMoE expert metadata is invalid")
	}
	if s.Architecture == "dbrx" &&
		(s.ExpertCount == 0 || s.ExpertUsedCount == 0 || s.ExpertUsedCount > s.ExpertCount ||
			s.ExpertUsedCount > 16 || s.ExpertFeedForward == 0 || s.ExpertWeightsScale <= 0 ||
			s.AttentionClamp < 0 || math.IsNaN(float64(s.AttentionClamp)) ||
			math.IsInf(float64(s.AttentionClamp), 0) || math.IsNaN(float64(s.ExpertWeightsScale)) ||
			math.IsInf(float64(s.ExpertWeightsScale), 0)) {
		return errors.New("DBRX expert or attention metadata is invalid")
	}
	if s.Architecture == "grok" {
		switch {
		case s.ExpertCount == 0 || s.ExpertUsedCount == 0 || s.ExpertUsedCount > s.ExpertCount ||
			s.ExpertUsedCount > 16 || s.ExpertFeedForward == 0:
			return errors.New("Grok expert metadata is invalid")
		case s.ExpertWeightsScale <= 0 || math.IsNaN(float64(s.ExpertWeightsScale)) ||
			math.IsInf(float64(s.ExpertWeightsScale), 0):
			return errors.New("Grok expert weight scale is invalid")
		case s.RopeDimensionCount != s.KeyLength || s.KeyLength != s.ValueLength ||
			s.RopeDimensionCount%2 != 0:
			return errors.New("Grok rotary/head dimensions are invalid")
		case s.AttentionScale <= 0 || s.AttentionSoftcap <= 0:
			return errors.New("Grok attention scaling metadata is invalid")
		case s.RopeScalingType == "yarn" &&
			(s.RopeScalingFactor <= 0 || s.OriginalContextLength == 0 ||
				s.YaRNExtFactor < 0 || s.YaRNAttentionFactor <= 0 ||
				s.YaRNBetaFast <= 0 || s.YaRNBetaSlow <= 0 ||
				math.IsNaN(float64(s.RopeScalingFactor)) || math.IsInf(float64(s.RopeScalingFactor), 0) ||
				math.IsNaN(float64(s.YaRNExtFactor)) || math.IsInf(float64(s.YaRNExtFactor), 0) ||
				math.IsNaN(float64(s.YaRNAttentionFactor)) || math.IsInf(float64(s.YaRNAttentionFactor), 0) ||
				math.IsNaN(float64(s.YaRNBetaFast)) || math.IsInf(float64(s.YaRNBetaFast), 0) ||
				math.IsNaN(float64(s.YaRNBetaSlow)) || math.IsInf(float64(s.YaRNBetaSlow), 0)):
			return errors.New("Grok YaRN metadata is invalid")
		}
	}
	if s.Architecture == "mellum" {
		switch {
		case s.ExpertCount == 0 || s.ExpertUsedCount == 0 || s.ExpertUsedCount > s.ExpertCount ||
			s.ExpertUsedCount > 16 || s.ExpertFeedForward == 0:
			return errors.New("Mellum expert metadata is invalid")
		case s.RopeDimensionCount != s.KeyLength || s.KeyLength != s.ValueLength || s.RopeDimensionCount%2 != 0:
			return errors.New("Mellum rotary/head dimensions are invalid")
		case s.SlidingWindow > 0 && (s.RopeFrequencySWA <= 0 ||
			(len(s.SlidingLayers) == 0 && s.SlidingPattern < 2)):
			return errors.New("Mellum sliding-attention metadata is invalid")
		case s.RopeScalingType == "yarn" &&
			(s.RopeScalingFactor <= 0 || s.OriginalContextLength == 0 || s.YaRNExtFactor < 0 ||
				s.YaRNAttentionFactor <= 0 || s.YaRNBetaFast <= 0 || s.YaRNBetaSlow <= 0):
			return errors.New("Mellum YaRN metadata is invalid")
		}
	}
	if s.Architecture == "hunyuan-moe" &&
		(s.ExpertCount == 0 || s.ExpertUsedCount == 0 || s.ExpertUsedCount > s.ExpertCount ||
			s.ExpertUsedCount > 16 || s.ExpertFeedForward == 0 || s.SharedExpertFF == 0 ||
			s.RopeDimensionCount != s.KeyLength || s.KeyLength != s.ValueLength ||
			s.RopeDimensionCount%2 != 0) {
		return errors.New("Hunyuan-MoE metadata is invalid")
	}
	if s.Architecture == "smallthinker" {
		switch {
		case s.ExpertCount == 0 || s.ExpertUsedCount == 0 || s.ExpertUsedCount > s.ExpertCount ||
			s.ExpertUsedCount > 16 || s.ExpertFeedForward == 0:
			return errors.New("SmallThinker expert metadata is invalid")
		case s.ExpertGatingFunc != 1 && s.ExpertGatingFunc != 2:
			return errors.New("SmallThinker expert routing function is unsupported")
		case s.ExpertWeightsScale <= 0 || math.IsNaN(float64(s.ExpertWeightsScale)) ||
			math.IsInf(float64(s.ExpertWeightsScale), 0):
			return errors.New("SmallThinker expert weight scale is invalid")
		case s.RopeDimensionCount != s.KeyLength || s.KeyLength != s.ValueLength ||
			s.RopeDimensionCount%2 != 0:
			return errors.New("SmallThinker rotary/head dimensions are invalid")
		case s.SlidingWindow > 0 && (s.SlidingPattern < 2 || s.RopeFrequencySWA <= 0):
			return errors.New("SmallThinker sliding-attention metadata is invalid")
		}
	}
	if s.Architecture == "dots1" {
		switch {
		case s.LeadingDenseBlocks >= s.BlockCount:
			return errors.New("DOTS1 leading dense block count leaves no MoE layers")
		case s.ExpertCount == 0 || s.ExpertUsedCount == 0 || s.ExpertUsedCount > s.ExpertCount ||
			s.ExpertUsedCount > 16 || s.ExpertFeedForward == 0 || s.SharedExpertCount == 0 ||
			s.SharedExpertFF == 0:
			return errors.New("DOTS1 expert metadata is invalid")
		case s.SharedExpertFF/s.SharedExpertCount != s.ExpertFeedForward:
			return errors.New("DOTS1 shared expert width overflows")
		case s.ExpertGatingFunc != 1 && s.ExpertGatingFunc != 2:
			return errors.New("DOTS1 expert routing function is unsupported")
		case s.ExpertWeightsScale <= 0 || math.IsNaN(float64(s.ExpertWeightsScale)) ||
			math.IsInf(float64(s.ExpertWeightsScale), 0):
			return errors.New("DOTS1 expert weight scale is invalid")
		case s.RopeDimensionCount > 0 && s.RopeDimensionCount != s.KeyLength:
			return errors.New("DOTS1 rotary dimension must equal key length")
		case s.HeadCountKV != s.HeadCount || s.KeyLength != s.ValueLength:
			return errors.New("DOTS1 requires full-head matching key/value attention")
		}
	}
	if s.Architecture == "minimax-m2" {
		switch {
		case s.ExpertCount == 0 || s.ExpertUsedCount == 0 || s.ExpertUsedCount > s.ExpertCount ||
			s.ExpertUsedCount > 16 || s.ExpertFeedForward == 0:
			return errors.New("MiniMax-M2 expert metadata is invalid")
		case s.ExpertGatingFunc != 1 && s.ExpertGatingFunc != 2:
			return errors.New("MiniMax-M2 expert routing function is unsupported")
		case s.ExpertWeightsScale <= 0 || math.IsNaN(float64(s.ExpertWeightsScale)) ||
			math.IsInf(float64(s.ExpertWeightsScale), 0):
			return errors.New("MiniMax-M2 expert weight scale is invalid")
		case s.RopeDimensionCount == 0 || s.RopeDimensionCount > s.KeyLength ||
			s.RopeDimensionCount%2 != 0 || s.KeyLength != s.ValueLength:
			return errors.New("MiniMax-M2 rotary/head dimensions are invalid")
		}
	}
	if (s.Architecture == "granite" || s.Architecture == "granitemoe") &&
		s.RopeScalingType == "longrope" &&
		(s.RopeDimensionCount == 0 || s.RopeDimensionCount > s.KeyLength ||
			s.RopeDimensionCount%2 != 0 || s.OriginalContextLength == 0 ||
			s.RopeAttentionFactor <= 0 || math.IsNaN(float64(s.RopeAttentionFactor)) ||
			math.IsInf(float64(s.RopeAttentionFactor), 0)) {
		return errors.New("Granite LongRoPE metadata is invalid")
	}
	if s.Architecture == "bailingmoe2" {
		switch {
		case s.LeadingDenseBlocks >= s.BlockCount:
			return errors.New("BailingMoE2 leading dense block count leaves no MoE layers")
		case s.ExpertCount == 0 || s.ExpertUsedCount == 0 || s.ExpertUsedCount > s.ExpertCount ||
			s.ExpertUsedCount > 16 || s.ExpertFeedForward == 0 || s.SharedExpertCount == 0 ||
			s.SharedExpertFF == 0:
			return errors.New("BailingMoE2 expert metadata is invalid")
		case s.ExpertGatingFunc != 1 && s.ExpertGatingFunc != 2:
			return errors.New("BailingMoE2 expert routing function is unsupported")
		case s.SharedExpertFF%s.SharedExpertCount != 0:
			return errors.New("BailingMoE2 shared expert width is invalid")
		case s.ExpertWeightsScale <= 0 || math.IsNaN(float64(s.ExpertWeightsScale)) ||
			math.IsInf(float64(s.ExpertWeightsScale), 0):
			return errors.New("BailingMoE2 expert weight scale is invalid")
		case s.RopeDimensionCount == 0 || s.RopeDimensionCount > s.KeyLength ||
			s.RopeDimensionCount%2 != 0 || s.KeyLength != s.ValueLength:
			return errors.New("BailingMoE2 rotary/head dimensions are invalid")
		}
	}
	if s.Architecture == "olmoe" &&
		(s.HeadCountKV != s.HeadCount || s.ExpertCount == 0 || s.ExpertUsedCount == 0 ||
			s.ExpertUsedCount > s.ExpertCount || s.ExpertUsedCount > 16 ||
			s.ExpertFeedForward == 0 || s.ExpertWeightsScale <= 0 ||
			math.IsNaN(float64(s.ExpertWeightsScale)) || math.IsInf(float64(s.ExpertWeightsScale), 0)) {
		return errors.New("OLMoE expert or attention metadata is invalid")
	}
	if s.Architecture == "llama" && s.ExpertCount > 0 &&
		(s.ExpertUsedCount == 0 || s.ExpertUsedCount > s.ExpertCount || s.ExpertUsedCount > 16 ||
			s.ExpertFeedForward == 0 || s.ExpertWeightsScale <= 0 ||
			math.IsNaN(float64(s.ExpertWeightsScale)) || math.IsInf(float64(s.ExpertWeightsScale), 0)) {
		return errors.New("Llama MoE expert metadata is invalid")
	}
	if s.Architecture == "phimoe" &&
		(s.ExpertCount == 0 || s.ExpertUsedCount == 0 || s.ExpertUsedCount > s.ExpertCount ||
			s.ExpertUsedCount > 16 || s.ExpertFeedForward == 0 || s.ExpertWeightsScale <= 0 ||
			math.IsNaN(float64(s.ExpertWeightsScale)) || math.IsInf(float64(s.ExpertWeightsScale), 0)) {
		return errors.New("PhiMoE expert metadata is invalid")
	}
	if s.Architecture == "laguna" {
		if len(s.LayerHeadCounts) != int(s.BlockCount) ||
			len(s.LayerKVHeadCounts) != int(s.BlockCount) {
			return errors.New("Laguna per-layer attention head metadata is invalid")
		}
		for block := uint32(0); block < s.BlockCount; block++ {
			heads := s.LayerHeadCount(block)
			kvHeads := s.LayerKVHeadCount(block)
			if heads == 0 || kvHeads == 0 || heads%kvHeads != 0 {
				return fmt.Errorf("Laguna layer %d attention head metadata is invalid", block)
			}
		}
		switch {
		case s.LeadingDenseBlocks >= s.BlockCount:
			return errors.New("Laguna leading dense block count leaves no MoE layers")
		case s.ExpertCount == 0 || s.ExpertUsedCount == 0 ||
			s.ExpertUsedCount > s.ExpertCount || s.ExpertUsedCount > 16 ||
			s.ExpertFeedForward == 0 || s.SharedExpertFF == 0:
			return errors.New("Laguna expert metadata is invalid")
		case s.ExpertGatingFunc != 2:
			return errors.New("Laguna requires sigmoid expert routing")
		case s.ExpertWeightsScale <= 0 || math.IsNaN(float64(s.ExpertWeightsScale)) ||
			math.IsInf(float64(s.ExpertWeightsScale), 0):
			return errors.New("Laguna expert weight scale is invalid")
		case s.RopeDimensionCount == 0 || s.RopeDimensionCount > s.KeyLength ||
			s.RopeDimensionCount%2 != 0:
			return errors.New("Laguna full-attention rotary dimension count is invalid")
		case s.RopeScalingType != "yarn":
			return errors.New("Laguna full-attention layers require YaRN RoPE")
		case s.KeyLength != s.ValueLength:
			return errors.New("Laguna requires matching attention key and value lengths")
		case s.RopeScalingFactor <= 0 || s.OriginalContextLength == 0 ||
			s.YaRNExtFactor < 0 || s.YaRNAttentionFactor <= 0 ||
			s.YaRNBetaFast <= 0 || s.YaRNBetaSlow <= 0 ||
			math.IsNaN(float64(s.YaRNExtFactor)) || math.IsInf(float64(s.YaRNExtFactor), 0) ||
			math.IsNaN(float64(s.YaRNAttentionFactor)) || math.IsInf(float64(s.YaRNAttentionFactor), 0) ||
			math.IsNaN(float64(s.YaRNBetaFast)) || math.IsInf(float64(s.YaRNBetaFast), 0) ||
			math.IsNaN(float64(s.YaRNBetaSlow)) || math.IsInf(float64(s.YaRNBetaSlow), 0):
			return errors.New("Laguna YaRN metadata is invalid")
		case s.SlidingWindow > 0 && (s.SlidingPattern < 2 || s.RopeFrequencySWA <= 0 ||
			s.RopeDimensionSWA == 0 || s.RopeDimensionSWA > s.KeyLength || s.RopeDimensionSWA%2 != 0):
			return errors.New("Laguna sliding-attention metadata is invalid")
		}
	}
	if s.Architecture == "afmoe" {
		switch {
		case s.LeadingDenseBlocks >= s.BlockCount:
			return errors.New("AFMoE leading dense block count leaves no MoE layers")
		case s.ExpertCount == 0 || s.ExpertUsedCount == 0 ||
			s.ExpertUsedCount > s.ExpertCount || s.ExpertUsedCount > 16 ||
			s.ExpertFeedForward == 0:
			return errors.New("AFMoE expert metadata is invalid")
		case s.ExpertGatingFunc != 2:
			return errors.New("AFMoE requires sigmoid expert routing")
		case s.ExpertWeightsScale <= 0 || math.IsNaN(float64(s.ExpertWeightsScale)) ||
			math.IsInf(float64(s.ExpertWeightsScale), 0):
			return errors.New("AFMoE expert weight scale is invalid")
		case s.SharedExpertCount > 0 && s.SharedExpertFF/s.SharedExpertCount != s.ExpertFeedForward:
			return errors.New("AFMoE shared expert width overflows")
		case s.RopeDimensionCount == 0 || s.RopeDimensionCount > s.KeyLength ||
			s.RopeDimensionCount%2 != 0 || s.KeyLength != s.ValueLength:
			return errors.New("AFMoE rotary/head dimensions are invalid")
		case s.SlidingWindow > 0 && (s.SlidingPattern < 2 || s.RopeFrequencySWA <= 0):
			return errors.New("AFMoE sliding-attention metadata is invalid")
		}
	}
	if s.Architecture == "exaone-moe" {
		switch {
		case s.LeadingDenseBlocks >= s.BlockCount:
			return errors.New("EXAONE-MoE leading dense block count leaves no MoE layers")
		case s.ExpertCount == 0 || s.ExpertUsedCount == 0 || s.ExpertUsedCount > s.ExpertCount ||
			s.ExpertUsedCount > 16 || s.ExpertFeedForward == 0 || s.SharedExpertFF == 0:
			return errors.New("EXAONE-MoE expert metadata is invalid")
		case s.ExpertGatingFunc != 1 && s.ExpertGatingFunc != 2:
			return errors.New("EXAONE-MoE expert routing function is unsupported")
		case s.ExpertWeightsScale <= 0 || math.IsNaN(float64(s.ExpertWeightsScale)) || math.IsInf(float64(s.ExpertWeightsScale), 0):
			return errors.New("EXAONE-MoE expert weight scale is invalid")
		case s.RopeDimensionCount != s.KeyLength || s.KeyLength != s.ValueLength || s.RopeDimensionCount%2 != 0:
			return errors.New("EXAONE-MoE rotary/head dimensions are invalid")
		case s.SlidingWindow == 0 || (len(s.SlidingLayers) == 0 && s.SlidingPattern < 2) || s.RopeFrequencySWA <= 0:
			return errors.New("EXAONE-MoE sliding attention metadata is invalid")
		}
	}
	if s.Architecture == "lfm2" || s.Architecture == "lfm2moe" {
		if s.ShortConvCacheLength < 2 || len(s.RecurrentLayers) != int(s.BlockCount) {
			return errors.New("LFM2 short-convolution metadata is invalid")
		}
		var recurrent, attention bool
		for _, item := range s.RecurrentLayers {
			recurrent = recurrent || item
			attention = attention || !item
		}
		if !recurrent || !attention {
			return errors.New("LFM2 requires both convolution and attention layers")
		}
		if s.Architecture == "lfm2moe" {
			switch {
			case s.LeadingDenseBlocks >= s.BlockCount:
				return errors.New("LFM2-MoE leading dense block count leaves no MoE layers")
			case s.ExpertCount == 0 || s.ExpertUsedCount == 0 || s.ExpertUsedCount > s.ExpertCount ||
				s.ExpertUsedCount > 16 || s.ExpertFeedForward == 0:
				return errors.New("LFM2-MoE expert metadata is invalid")
			case s.ExpertGatingFunc != 1 && s.ExpertGatingFunc != 2:
				return errors.New("LFM2-MoE expert routing function is unsupported")
			case s.ExpertWeightsScale <= 0 || math.IsNaN(float64(s.ExpertWeightsScale)) ||
				math.IsInf(float64(s.ExpertWeightsScale), 0):
				return errors.New("LFM2-MoE expert weight scale is invalid")
			}
		}
	}
	if s.Architecture == "plm" &&
		(s.KVLoRARank == 0 || s.RopeDimensionCount == 0 ||
			s.RopeDimensionCount >= s.KeyLength || s.HeadCountKV != s.HeadCount) {
		return errors.New("PLM MLA metadata is invalid")
	}
	if s.Architecture == "chameleon" && s.QKNormEpsilon <= 0 {
		return errors.New("Chameleon Q/K LayerNorm epsilon must be positive")
	}
	if s.Architecture == "jais2" && s.HeadCountKV != s.HeadCount {
		return errors.New("Jais2 requires matching attention and KV head counts")
	}
	if s.Architecture == "openelm" {
		if len(s.LayerHeadCounts) != int(s.BlockCount) ||
			len(s.LayerKVHeadCounts) != int(s.BlockCount) ||
			len(s.LayerFeedForward) != int(s.BlockCount) {
			return errors.New("OpenELM per-layer metadata is invalid")
		}
		for block := uint32(0); block < s.BlockCount; block++ {
			heads := s.LayerHeadCount(block)
			kvHeads := s.LayerKVHeadCount(block)
			if heads == 0 || kvHeads == 0 || heads%kvHeads != 0 || s.LayerFeedForwardLength(block) == 0 {
				return fmt.Errorf("OpenELM layer %d dimensions are invalid", block)
			}
		}
	}
	if s.Architecture == "deci" {
		if len(s.LayerHeadCounts) != int(s.BlockCount) ||
			len(s.LayerKVHeadCounts) != int(s.BlockCount) ||
			len(s.LayerFeedForward) != int(s.BlockCount) {
			return errors.New("Deci per-layer metadata is invalid")
		}
		var fullAttention bool
		for block := uint32(0); block < s.BlockCount; block++ {
			heads := s.LayerHeadCount(block)
			kvHeads := s.LayerKVHeadCount(block)
			if heads == 0 && kvHeads != 0 || kvHeads > 0 && (heads == 0 || heads%kvHeads != 0) {
				return fmt.Errorf("Deci layer %d attention head metadata is invalid", block)
			}
			fullAttention = fullAttention || kvHeads > 0
		}
		if !fullAttention {
			return errors.New("Deci requires at least one full-attention layer")
		}
		if s.RopeDimensionCount == 0 || s.RopeDimensionCount > s.KeyLength ||
			s.RopeDimensionCount%2 != 0 || s.KeyLength != s.ValueLength {
			return errors.New("Deci rotary/head dimensions are invalid")
		}
		if s.RopeScalingType == "longrope" &&
			(s.OriginalContextLength == 0 || s.RopeAttentionFactor <= 0 ||
				math.IsNaN(float64(s.RopeAttentionFactor)) || math.IsInf(float64(s.RopeAttentionFactor), 0)) {
			return errors.New("Deci LongRoPE metadata is invalid")
		}
	}
	if s.Architecture == "gemma3" {
		switch {
		case s.RopeFrequencySWA <= 0:
			return errors.New("Gemma 3 sliding RoPE frequency base must be positive")
		case s.RopeScalingFactor <= 0:
			return errors.New("Gemma 3 RoPE scaling factor must be positive")
		case s.SlidingWindow > 0 && s.SlidingPattern < 2:
			return errors.New("Gemma 3 sliding attention pattern must be at least 2")
		}
	}
	if s.Architecture == "gemma2" {
		switch {
		case s.RopeFrequencySWA <= 0:
			return errors.New("Gemma 2 sliding RoPE frequency base must be positive")
		case s.SlidingWindow > 0 && s.SlidingPattern < 2:
			return errors.New("Gemma 2 sliding attention pattern must be at least 2")
		}
	}
	if s.Architecture == "olmo2" {
		switch {
		case s.SlidingWindow > 0 && s.RopeFrequencySWA <= 0:
			return errors.New("OLMo2 sliding RoPE frequency base must be positive")
		case s.SlidingWindow > 0 && s.SlidingPattern < 2:
			return errors.New("OLMo2 sliding attention pattern must be at least 2")
		}
	}
	if s.Architecture == "cohere2" {
		switch {
		case s.RopeDimensionCount == 0 || s.RopeDimensionCount > s.KeyLength ||
			s.RopeDimensionCount%2 != 0:
			return errors.New("Cohere2 rotary dimension count is invalid")
		case s.RopeFrequencySWA <= 0:
			return errors.New("Cohere2 sliding RoPE frequency base must be positive")
		case s.SlidingWindow == 0:
			return errors.New("Cohere2 sliding attention window is zero")
		case s.SlidingPattern < 2:
			return errors.New("Cohere2 sliding attention pattern must be at least 2")
		}
	}
	if s.Architecture == "stablelm" &&
		(s.RopeDimensionCount == 0 || s.RopeDimensionCount > s.KeyLength ||
			s.RopeDimensionCount%2 != 0) {
		return errors.New("StableLM rotary dimension count is invalid")
	}
	if s.Architecture == "phi2" &&
		(s.RopeDimensionCount == 0 || s.RopeDimensionCount > s.KeyLength ||
			s.RopeDimensionCount%2 != 0) {
		return errors.New("Phi-2 rotary dimension count is invalid")
	}
	if (s.Architecture == "phi3" || s.Architecture == "phimoe") &&
		(s.RopeDimensionCount == 0 || s.RopeDimensionCount > s.KeyLength ||
			s.RopeDimensionCount%2 != 0 || s.OriginalContextLength == 0 ||
			s.RopeAttentionFactor <= 0 || math.IsNaN(float64(s.RopeAttentionFactor)) ||
			math.IsInf(float64(s.RopeAttentionFactor), 0)) {
		return errors.New("Phi-3 RoPE metadata is invalid")
	}
	if s.Architecture == "apertus" {
		if s.RopeDimensionCount != s.KeyLength || s.RopeDimensionCount%2 != 0 ||
			s.OriginalContextLength == 0 || s.RopeAttentionFactor <= 0 ||
			math.IsNaN(float64(s.RopeAttentionFactor)) || math.IsInf(float64(s.RopeAttentionFactor), 0) {
			return errors.New("Apertus RoPE metadata is invalid")
		}
		for name, values := range map[string][]float32{
			"alpha_n": s.XIELUAlphaN,
			"alpha_p": s.XIELUAlphaP,
			"beta":    s.XIELUBeta,
			"epsilon": s.XIELUEpsilon,
		} {
			if len(values) != int(s.BlockCount) {
				return fmt.Errorf("Apertus xIELU %s has %d values, need %d", name, len(values), s.BlockCount)
			}
			for _, value := range values {
				if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
					return fmt.Errorf("Apertus xIELU %s values must be finite", name)
				}
			}
		}
	}
	if s.Architecture == "gptneox" && s.RopeDimensionCount > 0 &&
		(s.RopeDimensionCount > s.KeyLength || s.RopeDimensionCount%2 != 0) {
		return errors.New("GPT-NeoX rotary dimension count is invalid")
	}
	if s.Architecture == "glm4" &&
		(s.RopeDimensionCount == 0 || s.RopeDimensionCount > s.KeyLength ||
			s.RopeDimensionCount%2 != 0) {
		return errors.New("GLM4 rotary dimension count is invalid")
	}
	if s.Architecture == "exaone4" {
		switch {
		case s.RopeDimensionCount != s.KeyLength || s.RopeDimensionCount%2 != 0:
			return errors.New("EXAONE 4 rotary dimension count must equal the key length")
		case s.SlidingWindow > 0 && s.SlidingPattern < 2:
			return errors.New("EXAONE 4 sliding attention pattern must be at least 2")
		case s.SlidingWindow > 0 && s.RopeFrequencySWA <= 0:
			return errors.New("EXAONE 4 sliding RoPE frequency base must be positive")
		}
	}
	if s.Architecture == "falcon" && s.RopeDimensionCount > 0 &&
		s.RopeDimensionCount != s.KeyLength {
		return errors.New("Falcon rotary dimension count must equal the key length")
	}
	if s.RopeScalingType == "linear" && s.RopeScalingFactor <= 0 {
		return errors.New("linear RoPE scaling factor must be positive")
	}
	if s.FinalLogitSoftcap < 0 ||
		math.IsNaN(float64(s.FinalLogitSoftcap)) ||
		math.IsInf(float64(s.FinalLogitSoftcap), 0) {
		return errors.New("final logit softcap must be finite and non-negative")
	}
	if s.AttentionSoftcap < 0 ||
		math.IsNaN(float64(s.AttentionSoftcap)) ||
		math.IsInf(float64(s.AttentionSoftcap), 0) {
		return errors.New("attention logit softcap must be finite and non-negative")
	}
	if s.AttentionScale < 0 ||
		math.IsNaN(float64(s.AttentionScale)) ||
		math.IsInf(float64(s.AttentionScale), 0) {
		return errors.New("attention scale must be finite and non-negative")
	}
	if s.MaxALiBiBias < 0 || math.IsNaN(float64(s.MaxALiBiBias)) ||
		math.IsInf(float64(s.MaxALiBiBias), 0) {
		return errors.New("maximum ALiBi bias must be finite and non-negative")
	}
	if s.EmbeddingScale < 0 ||
		math.IsNaN(float64(s.EmbeddingScale)) ||
		math.IsInf(float64(s.EmbeddingScale), 0) {
		return errors.New("embedding scale must be finite and non-negative")
	}
	if s.ResidualScale < 0 ||
		math.IsNaN(float64(s.ResidualScale)) ||
		math.IsInf(float64(s.ResidualScale), 0) {
		return errors.New("residual scale must be finite and non-negative")
	}
	if s.LogitScale < 0 ||
		math.IsNaN(float64(s.LogitScale)) ||
		math.IsInf(float64(s.LogitScale), 0) ||
		((s.Architecture == "minicpm" || s.Architecture == "granite" || s.Architecture == "granitemoe") &&
			s.LogitScale == 0) {
		return errors.New("logit scale must be finite and positive when required")
	}
	return nil
}

func isGemmaArchitecture(architecture string) bool {
	return architecture == "gemma" || architecture == "gemma2" || architecture == "gemma3"
}

func hasPostNorm(architecture string) bool {
	return architecture == "afmoe" || architecture == "exaone4" || architecture == "gemma2" ||
		architecture == "gemma3" || architecture == "glm4" || architecture == "grok"
}

func usesSlidingAttention(architecture string) bool {
	return architecture == "afmoe" || (architecture == "gemma2" || architecture == "gemma3") ||
		architecture == "exaone4" || architecture == "exaone-moe" || architecture == "olmo2" ||
		architecture == "cohere2" || architecture == "mellum" || architecture == "smallthinker"
}

func usesPostOnlyNorm(architecture string) bool {
	return architecture == "exaone4"
}

func usesNormalRoPE(architecture string) bool {
	return architecture == "llama" ||
		architecture == "arctic" ||
		architecture == "deci" ||
		architecture == "llada" ||
		architecture == "internlm2" ||
		architecture == "arcee" ||
		architecture == "baichuan" ||
		architecture == "bailingmoe" ||
		architecture == "deepseek" ||
		architecture == "cohere2" ||
		architecture == "command-r" ||
		architecture == "chameleon" ||
		architecture == "granite" ||
		architecture == "granitemoe" ||
		architecture == "glm4" ||
		architecture == "minicpm" ||
		architecture == "olmo" ||
		architecture == "maincoder" ||
		architecture == "mistral3" ||
		architecture == "smollm3" ||
		architecture == "xverse"
}

func usesParallelResidual(architecture string) bool {
	return architecture == "cohere2" || architecture == "command-r" || architecture == "falcon" ||
		architecture == "phi2" || architecture == "plamo"
}

func usesSequentialGELU(architecture string) bool {
	return architecture == "bloom" || architecture == "codeshell" || architecture == "gpt2" ||
		architecture == "gptneox" || architecture == "phi2" ||
		architecture == "starcoder" || architecture == "starcoder2"
}

func usesGateFreeFFN(architecture string) bool {
	return architecture == "apertus" || usesFusedGateUp(architecture) ||
		usesSquaredReLU(architecture) || usesGELU(architecture)
}

func usesFusedGateUp(architecture string) bool {
	return architecture == "glm4" || architecture == "phi3"
}

func supportsLongRoPE(architecture string) bool {
	return architecture == "apertus" || architecture == "deci" || architecture == "granite" || architecture == "granitemoe" ||
		architecture == "phi3" || architecture == "phimoe"
}

func firstPositive(values []uint32) uint32 {
	for _, value := range values {
		if value > 0 {
			return value
		}
	}
	return 0
}

func usesGELU(architecture string) bool {
	return architecture == "falcon" || architecture == "mpt" || usesSequentialGELU(architecture)
}

func usesSquaredReLU(architecture string) bool {
	return architecture == "arcee" || architecture == "jais2" || architecture == "nemotron" || architecture == "plm"
}

func required[T any](values map[string]gguf.Value, key string, valueType gguf.ValueType) (T, error) {
	value, ok := values[key]
	if !ok {
		var zero T
		return zero, fmt.Errorf("required metadata %q is missing", key)
	}
	if value.Type != valueType {
		var zero T
		return zero, fmt.Errorf("metadata %q has type %s, need %s", key, value.Type, valueType)
	}
	typed, ok := value.Data.(T)
	if !ok {
		var zero T
		return zero, fmt.Errorf("metadata %q has an invalid Go representation", key)
	}
	return typed, nil
}

func optional[T any](values map[string]gguf.Value, key string, valueType gguf.ValueType) (T, bool) {
	value, ok := values[key]
	if !ok || value.Type != valueType {
		var zero T
		return zero, false
	}
	typed, ok := value.Data.(T)
	return typed, ok
}

func requiredArray[T any](
	values map[string]gguf.Value,
	key string,
	elementType gguf.ValueType,
) ([]T, error) {
	value, ok := values[key]
	if !ok {
		return nil, fmt.Errorf("required metadata %q is missing", key)
	}
	if value.Type != gguf.ValueTypeArray || value.ArrayType != elementType {
		return nil, fmt.Errorf(
			"metadata %q must be an array of %s",
			key,
			elementType,
		)
	}
	typed, ok := value.Data.([]T)
	if !ok {
		return nil, fmt.Errorf("metadata %q has an invalid Go representation", key)
	}
	return typed, nil
}

func requiredLayerFloat32(
	values map[string]gguf.Value,
	key string,
	count uint32,
) ([]float32, error) {
	value, ok := values[key]
	if !ok {
		return nil, fmt.Errorf("required metadata %q is missing", key)
	}
	if value.Type == gguf.ValueTypeFloat32 {
		scalar, ok := value.Data.(float32)
		if !ok {
			return nil, fmt.Errorf("metadata %q has an invalid Go representation", key)
		}
		result := make([]float32, count)
		for index := range result {
			result[index] = scalar
		}
		return result, nil
	}
	if value.Type != gguf.ValueTypeArray || value.ArrayType != gguf.ValueTypeFloat32 {
		return nil, fmt.Errorf("metadata %q must be a float32 or float32 array", key)
	}
	items, ok := value.Data.([]float32)
	if !ok {
		return nil, fmt.Errorf("metadata %q has an invalid Go representation", key)
	}
	if len(items) != int(count) {
		return nil, fmt.Errorf("metadata %q has %d values, need %d", key, len(items), count)
	}
	return append([]float32(nil), items...), nil
}

func requiredLayerUint32(
	values map[string]gguf.Value,
	key string,
	count uint32,
) ([]uint32, error) {
	value, ok := values[key]
	if !ok {
		return nil, fmt.Errorf("required metadata %q is missing", key)
	}
	if value.Type == gguf.ValueTypeUint32 {
		scalar, ok := value.Data.(uint32)
		if !ok {
			return nil, fmt.Errorf("metadata %q has an invalid Go representation", key)
		}
		result := make([]uint32, count)
		for index := range result {
			result[index] = scalar
		}
		return result, nil
	}
	if value.Type != gguf.ValueTypeArray || value.ArrayType != gguf.ValueTypeUint32 {
		return nil, fmt.Errorf("metadata %q must be a uint32 or uint32 array", key)
	}
	items, ok := value.Data.([]uint32)
	if !ok {
		return nil, fmt.Errorf("metadata %q has an invalid Go representation", key)
	}
	if len(items) != int(count) {
		return nil, fmt.Errorf("metadata %q has %d values, need %d", key, len(items), count)
	}
	return append([]uint32(nil), items...), nil
}

func optionalArray[T any](
	values map[string]gguf.Value,
	key string,
	elementType gguf.ValueType,
) ([]T, bool, error) {
	value, ok := values[key]
	if !ok {
		return nil, false, nil
	}
	if value.Type != gguf.ValueTypeArray || value.ArrayType != elementType {
		return nil, false, fmt.Errorf(
			"metadata %q must be an array of %s",
			key,
			elementType,
		)
	}
	typed, ok := value.Data.([]T)
	if !ok {
		return nil, false, fmt.Errorf("metadata %q has an invalid Go representation", key)
	}
	return typed, true, nil
}
