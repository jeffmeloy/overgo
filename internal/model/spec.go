package model

import (
	"errors"
	"fmt"
	"math"

	"llamacpp2go/internal/gguf"
)

// Spec contains the common transformer metadata needed to construct a model.
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
	EmbeddingScale        float32
	ResidualScale         float32
	LogitScale            float32
	AttentionSoftcap      float32
	FinalLogitSoftcap     float32
	RMSNormEpsilon        float32
	LayerNormEpsilon      float32
	VocabularySize        uint32
	SlidingWindow         uint32
	SlidingPattern        uint32
	RelativeBuckets       uint32
	NoRopeLayerStep       uint32
	RopeDisabled          bool
	ParallelResidual      bool
	XIELUAlphaN           []float32
	XIELUAlphaP           []float32
	XIELUBeta             []float32
	XIELUEpsilon          []float32

	// Qwen3.5 hybrid recurrent-attention metadata.
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

// UnsupportedArchitectureError identifies a valid GGUF architecture that the
// runtime cannot execute yet.
type UnsupportedArchitectureError struct {
	Architecture string
}

func (e *UnsupportedArchitectureError) Error() string {
	return fmt.Sprintf("model architecture %q is not supported", e.Architecture)
}

// ReadSpec validates the common metadata for the initial Llama and Qwen3
// architecture families.
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
	if architecture != "llama" && architecture != "internlm2" &&
		architecture != "arcee" &&
		architecture != "apertus" &&
		architecture != "baichuan" &&
		architecture != "bitnet" &&
		architecture != "codeshell" &&
		architecture != "cohere2" &&
		architecture != "command-r" &&
		architecture != "jais2" &&
		architecture != "xverse" &&
		architecture != "exaone" && architecture != "olmo2" &&
		architecture != "exaone4" &&
		architecture != "smollm3" &&
		architecture != "minicpm" &&
		architecture != "granite" &&
		architecture != "glm4" &&
		architecture != "gptneox" &&
		architecture != "maincoder" &&
		architecture != "mistral3" &&
		architecture != "nemotron" &&
		architecture != "olmo" &&
		architecture != "orion" &&
		architecture != "phi2" &&
		architecture != "phi3" &&
		architecture != "plamo" &&
		architecture != "seed_oss" &&
		architecture != "stablelm" &&
		architecture != "starcoder2" &&
		architecture != "qwen2" &&
		architecture != "qwen3" &&
		architecture != "qwen35" && architecture != "gemma" &&
		architecture != "gemma2" &&
		architecture != "gemma3" &&
		architecture != "falcon" &&
		architecture != "t5encoder" {
		return Spec{}, &UnsupportedArchitectureError{Architecture: architecture}
	}
	spec := Spec{Architecture: architecture}
	if value, ok := optional[string](values, "general.name", gguf.ValueTypeString); ok {
		spec.Name = value
	}
	prefix := architecture + "."
	if spec.BlockCount, err = required[uint32](values, prefix+"block_count", gguf.ValueTypeUint32); err != nil {
		return Spec{}, err
	}
	if spec.ContextLength, err = required[uint32](values, prefix+"context_length", gguf.ValueTypeUint32); err != nil {
		return Spec{}, err
	}
	if spec.EmbeddingLength, err = required[uint32](values, prefix+"embedding_length", gguf.ValueTypeUint32); err != nil {
		return Spec{}, err
	}
	if spec.FeedForwardLength, err = required[uint32](values, prefix+"feed_forward_length", gguf.ValueTypeUint32); err != nil {
		return Spec{}, err
	}
	if spec.HeadCount, err = required[uint32](values, prefix+"attention.head_count", gguf.ValueTypeUint32); err != nil {
		return Spec{}, err
	}
	if architecture == "t5encoder" || architecture == "gptneox" || architecture == "falcon" {
		spec.HeadCountKV = spec.HeadCount
		if architecture == "gptneox" || architecture == "falcon" {
			if value, ok := optional[uint32](
				values,
				prefix+"attention.head_count_kv",
				gguf.ValueTypeUint32,
			); ok {
				spec.HeadCountKV = value
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
	if architecture != "t5encoder" {
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
				return Spec{}, fmt.Errorf(
					"model architecture %q uses unsupported RoPE scaling type %q",
					architecture,
					scalingType,
				)
			}
			spec.RopeScalingType = scalingType
			if scalingType == "linear" {
				if spec.RopeScalingFactor, err = required[float32](
					values,
					prefix+"rope.scaling.factor",
					gguf.ValueTypeFloat32,
				); err != nil {
					return Spec{}, err
				}
			}
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
	spec.AttentionScale, _ = optional[float32](
		values,
		prefix+"attention.scale",
		gguf.ValueTypeFloat32,
	)
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
	if architecture == "granite" {
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
		ropeEnabled := true
		if value, ok := optional[bool](
			values,
			prefix+"rope.scaling.finetuned",
			gguf.ValueTypeBool,
		); ok {
			ropeEnabled = value
		}
		spec.RopeDisabled = !ropeEnabled
		if expertCount, ok := optional[uint32](
			values,
			prefix+"expert_count",
			gguf.ValueTypeUint32,
		); ok && expertCount > 0 {
			return Spec{}, errors.New("Granite expert layers are not supported")
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
	if architecture == "phi3" {
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
	return usesSlidingAttention(s.Architecture) &&
		block < s.BlockCount &&
		s.SlidingWindow > 0 &&
		s.SlidingPattern > 0 &&
		block%s.SlidingPattern < s.SlidingPattern-1
}

func (s Spec) UsesRoPE(block uint32) bool {
	return !s.RopeDisabled &&
		(s.BlockCount == 0 || block < s.BlockCount) &&
		(s.NoRopeLayerStep == 0 || (block+1)%s.NoRopeLayerStep != 0)
}

func (s Spec) InputEmbeddingScale() float32 {
	if s.EmbeddingScale > 0 {
		return s.EmbeddingScale
	}
	if isGemmaArchitecture(s.Architecture) {
		return float32(math.Sqrt(float64(s.EmbeddingLength)))
	}
	return 1
}

func (s Spec) OutputLogitMultiplier() float32 {
	if s.LogitScale > 0 {
		if s.Architecture == "cohere2" || s.Architecture == "command-r" {
			return s.LogitScale
		}
		return 1 / s.LogitScale
	}
	return 1
}

func (s Spec) UsesLayerNorm() bool {
	return s.Architecture == "falcon" ||
		s.Architecture == "nemotron" ||
		s.Architecture == "jais2" ||
		s.Architecture == "orion" ||
		s.Architecture == "stablelm" ||
		usesSequentialGELU(s.Architecture)
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
	case s.Architecture != "t5encoder" && s.RopeFrequencyBase <= 0:
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
	if s.Architecture == "jais2" && s.HeadCountKV != s.HeadCount {
		return errors.New("Jais2 requires matching attention and KV head counts")
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
	if s.Architecture == "phi3" &&
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
		((s.Architecture == "minicpm" || s.Architecture == "granite") &&
			s.LogitScale == 0) {
		return errors.New("logit scale must be finite and positive when required")
	}
	return nil
}

func isGemmaArchitecture(architecture string) bool {
	return architecture == "gemma" || architecture == "gemma2" || architecture == "gemma3"
}

func hasPostNorm(architecture string) bool {
	return architecture == "exaone4" || architecture == "gemma2" ||
		architecture == "gemma3" || architecture == "glm4"
}

func usesSlidingAttention(architecture string) bool {
	return (architecture == "gemma2" || architecture == "gemma3") ||
		architecture == "exaone4" || architecture == "olmo2" || architecture == "cohere2"
}

func usesPostOnlyNorm(architecture string) bool {
	return architecture == "exaone4"
}

func usesNormalRoPE(architecture string) bool {
	return architecture == "llama" ||
		architecture == "internlm2" ||
		architecture == "arcee" ||
		architecture == "baichuan" ||
		architecture == "cohere2" ||
		architecture == "command-r" ||
		architecture == "granite" ||
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
	return architecture == "codeshell" || architecture == "gptneox" ||
		architecture == "phi2" || architecture == "starcoder2"
}

func usesGateFreeFFN(architecture string) bool {
	return architecture == "apertus" || usesFusedGateUp(architecture) ||
		usesSquaredReLU(architecture) || usesGELU(architecture)
}

func usesFusedGateUp(architecture string) bool {
	return architecture == "glm4" || architecture == "phi3"
}

func supportsLongRoPE(architecture string) bool {
	return architecture == "apertus" || architecture == "phi3"
}

func usesGELU(architecture string) bool {
	return architecture == "falcon" || usesSequentialGELU(architecture)
}

func usesSquaredReLU(architecture string) bool {
	return architecture == "arcee" || architecture == "jais2" || architecture == "nemotron"
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
