package model

import (
	"errors"
	"fmt"

	"llamacpp2go/internal/gguf"
)

// Spec contains the common transformer metadata needed to construct a model.
type Spec struct {
	Architecture      string
	Name              string
	BlockCount        uint32
	ContextLength     uint32
	EmbeddingLength   uint32
	FeedForwardLength uint32
	HeadCount         uint32
	HeadCountKV       uint32
	KeyLength         uint32
	ValueLength       uint32
	RopeFrequencyBase float32
	RopeFrequencySWA  float32
	RopeScalingFactor float32
	RMSNormEpsilon    float32
	VocabularySize    uint32
	SlidingWindow     uint32
	SlidingPattern    uint32
	RelativeBuckets   uint32

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
	if architecture != "llama" && architecture != "qwen3" &&
		architecture != "qwen35" && architecture != "gemma3" &&
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
	if architecture == "t5encoder" {
		spec.HeadCountKV = spec.HeadCount
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
		if spec.RopeFrequencyBase, err = required[float32](values, prefix+"rope.freq_base", gguf.ValueTypeFloat32); err != nil {
			return Spec{}, err
		}
		if scalingType, ok := optional[string](
			values,
			prefix+"rope.scaling.type",
			gguf.ValueTypeString,
		); ok && scalingType != "" && scalingType != "none" {
			if architecture != "gemma3" || scalingType != "linear" {
				return Spec{}, fmt.Errorf(
					"model architecture %q uses unsupported RoPE scaling type %q",
					architecture,
					scalingType,
				)
			}
			if spec.RopeScalingFactor, err = required[float32](
				values,
				prefix+"rope.scaling.factor",
				gguf.ValueTypeFloat32,
			); err != nil {
				return Spec{}, err
			}
		}
	}
	if spec.RMSNormEpsilon, err = required[float32](
		values,
		prefix+"attention.layer_norm_rms_epsilon",
		gguf.ValueTypeFloat32,
	); err != nil {
		return Spec{}, err
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
	if architecture == "gemma3" {
		spec.RopeFrequencySWA = 10000
		if value, ok := optional[float32](
			values,
			prefix+"rope.freq_base_swa",
			gguf.ValueTypeFloat32,
		); ok {
			spec.RopeFrequencySWA = value
		}
		spec.SlidingWindow, _ = optional[uint32](
			values,
			prefix+"attention.sliding_window",
			gguf.ValueTypeUint32,
		)
		if spec.SlidingWindow > 0 {
			spec.SlidingPattern = 6
			if value, ok := optional[uint32](
				values,
				prefix+"attention.sliding_window_pattern",
				gguf.ValueTypeUint32,
			); ok {
				spec.SlidingPattern = value
			}
		}
	}
	if tokens, ok := values["tokenizer.ggml.tokens"]; ok {
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
	return s.Architecture == "gemma3" &&
		block < s.BlockCount &&
		s.SlidingWindow > 0 &&
		s.SlidingPattern > 0 &&
		block%s.SlidingPattern < s.SlidingPattern-1
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
	case s.RMSNormEpsilon <= 0:
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
	return nil
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
