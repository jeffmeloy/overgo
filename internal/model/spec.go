package model

import (
	"errors"
	"fmt"
	"math"
	"slices"

	"llamacpp2go/internal/gguf"
)

const (
	chameleonQKNormEpsilon      = 1e-5
	deepSeek32BlockCount        = 62
	deepSeek32LayerNormEpsilon  = 1e-6
	deepSeekDenseIndexerContext = 1 << 20
	deepSeekInitialFullIndexers = 2
	deepSeekFullIndexerPeriod   = 4
)

// UnsupportedArchitectureError: valid, unsupported GGUF architecture.
type UnsupportedArchitectureError struct {
	Architecture string
}

// LayerHasFullIndexer: DSA full-indexer predicate.
func (s Spec) LayerHasFullIndexer(layer uint32) bool {
	if s.Architecture == "deepseek32" && layer < s.BlockCount+s.NextNPredictLayers {
		return true
	}
	return s.Profile().Attention == AttentionDSA && layerValue(s.IndexerFullLayers, layer, false)
}

func (e *UnsupportedArchitectureError) Error() string {
	return fmt.Sprintf("model architecture %q is not supported", e.Architecture)
}

// ReadSpec: validates model metadata.
func ReadSpec(file *gguf.File) (Spec, error) {
	metadata, err := newSpecMetadata(file)
	if err != nil {
		return Spec{}, err
	}
	architecture, profile := metadata.architecture, metadata.profile
	spec := Spec{CommonSpec: CommonSpec{Architecture: architecture}, AttentionSpec: AttentionSpec{NonCausalAttention: profile.Has(ArchitectureNonCausal),
		RopeDisabled: profile.Has(ArchitectureRoPEDisabled)},
	}
	state, err := metadata.readBase(&spec)
	if err != nil {
		return Spec{}, err
	}
	if err := metadata.readAttentionShape(&spec, state); err != nil {
		return Spec{}, err
	}
	if err := metadata.readPosition(&spec); err != nil {
		return Spec{}, err
	}
	for _, stage := range []func(Spec, specReadState) (Spec, error){
		metadata.readArchitectureCore,
		metadata.readExpertMetadata,
		metadata.readRuntimeMetadata,
	} {
		spec, err = stage(spec, state)
		if err != nil {
			return Spec{}, err
		}
	}
	return spec, nil
}

func (m specMetadata) readArchitectureCore(spec Spec, state specReadState) (Spec, error) {
	values, architecture, prefix := m.values, m.architecture, m.prefix
	declaredBlockCount := state.declaredBlockCount
	var err error
	if architecture == "refact" {
		spec.ExpertCount, _ = optional[uint32](
			values, prefix+"expert_count", gguf.ValueTypeUint32,
		)
		if spec.ExpertCount > 0 {
			if spec.ExpertUsedCount, err = required[uint32](
				values, prefix+"expert_used_count", gguf.ValueTypeUint32,
			); err != nil {
				return Spec{}, err
			}
			spec.ExpertFeedForward = spec.FeedForwardLength
			if value, ok := optional[uint32](
				values, prefix+"expert_feed_forward_length", gguf.ValueTypeUint32,
			); ok {
				spec.ExpertFeedForward = value
			}
			spec.ExpertWeightsNorm = true
			spec.ExpertWeightsScale = 1
		}
	}
	if architecture == "cohere2moe" {
		if value, ok := optional[float32](values, prefix+"attention.layer_norm_rms_epsilon", gguf.ValueTypeFloat32); ok {
			spec.RMSNormEpsilon = value
		} else if value, ok := optional[float32](values, prefix+"attention.layer_norm_epsilon", gguf.ValueTypeFloat32); ok {
			spec.LayerNormEpsilon = value
		} else {
			return Spec{}, errors.New("Cohere2-MoE norm epsilon is missing")
		}
	} else if spec.UsesLayerNorm() || spec.UsesWeightOnlyLayerNorm() || spec.UsesUnweightedLayerNorm() {
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
	if architecture == "talkie" {
		if spec.LogitScale, err = required[float32](values, prefix+"logit_scale", gguf.ValueTypeFloat32); err != nil {
			return Spec{}, err
		}
		spec.RopeDimensionCount = spec.KeyLength
	}
	if architecture == "dflash" {
		spec.RopeDimensionCount = spec.KeyLength
		if value, ok := optional[uint32](values, prefix+"rope.dimension_count", gguf.ValueTypeUint32); ok {
			spec.RopeDimensionCount = value
		}
		if window, ok := optional[uint32](values, prefix+"attention.sliding_window", gguf.ValueTypeUint32); ok && window > 0 {
			spec.SlidingWindow = window
			spec.RopeFrequencySWA = spec.RopeFrequencyBase
			if pattern, patternOK := optional[uint32](values, prefix+"attention.sliding_window_pattern", gguf.ValueTypeUint32); patternOK {
				spec.SlidingPattern = pattern
			} else if layers, layersOK, layersErr := optionalArray[bool](values, prefix+"attention.sliding_window_pattern", gguf.ValueTypeBool); layersErr != nil {
				return Spec{}, layersErr
			} else if layersOK {
				if len(layers) != int(spec.BlockCount) {
					return Spec{}, fmt.Errorf("metadata %q has %d values, need %d", prefix+"attention.sliding_window_pattern", len(layers), spec.BlockCount)
				}
				spec.SlidingLayers = slices.Clone(layers)
			}
		}
	}
	if architecture == "eagle3" {
		spec.RopeDimensionCount = spec.KeyLength
		if value, ok := optional[uint32](values, prefix+"rope.dimension_count", gguf.ValueTypeUint32); ok {
			spec.RopeDimensionCount = value
		}
	}
	if architecture == "gemma3n" {
		spec.RopeDimensionCount = spec.KeyLength
	}
	if architecture == "cogvlm" {
		spec.RopeDimensionCount = spec.KeyLength
	}
	if architecture == "minicpm" || architecture == "minicpm3" {
		spec.EmbeddingScale = 12
		spec.ResidualScale = float32(1.4 / math.Sqrt(float64(spec.BlockCount)))
		spec.LogitScale = 256 / float32(spec.EmbeddingLength)
		if architecture == "minicpm3" {
			spec.OriginalContextLength = spec.ContextLength
			spec.RopeAttentionFactor = 1
			if value, ok := optional[uint32](values, prefix+"rope.scaling.original_context_length", gguf.ValueTypeUint32); ok {
				spec.OriginalContextLength = value
			}
			if value, ok := optional[float32](values, prefix+"rope.scaling.attn_factor", gguf.ValueTypeFloat32); ok {
				spec.RopeAttentionFactor = value
			}
		}
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
	if architecture == "granite" || architecture == "granitemoe" || architecture == "granitehybrid" {
		if architecture == "granitehybrid" {
			spec.LogitScale, _ = optional[float32](values, prefix+"logit_scale", gguf.ValueTypeFloat32)
		} else if spec.LogitScale, err = required[float32](
			values, prefix+"logit_scale", gguf.ValueTypeFloat32,
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
			); ok {
				spec.ExpertCount = expertCount
			}
		}
		if mapping, ok, mappingErr := optionalArray[int32](
			values,
			prefix+"deepstack_mapping",
			gguf.ValueTypeInt32,
		); mappingErr != nil {
			return Spec{}, mappingErr
		} else if ok && len(mapping) > 0 {
			if architecture != "granite" {
				return Spec{}, errors.New("Granite deepstack mapping requires granite architecture")
			}
			if len(mapping) != int(spec.BlockCount) {
				return Spec{}, fmt.Errorf("Granite deepstack mapping has %d entries, need %d", len(mapping), spec.BlockCount)
			}
			unique := make(map[int32]struct{})
			for _, index := range mapping {
				if index < -1 {
					return Spec{}, errors.New("Granite deepstack mapping index is invalid")
				}
				if index >= 0 {
					unique[index] = struct{}{}
				}
			}
			for index := range unique {
				if uint32(index) > uint32(len(unique)) {
					return Spec{}, errors.New("Granite deepstack mapping index exceeds stream count")
				}
			}
			spec.DeepstackLayerCount = uint32(len(unique))
			spec.DeepstackMapping = slices.Clone(mapping)
		}
	}
	if architecture == "cohere2" || architecture == "cohere2moe" {
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
			pattern.Type != gguf.ValueTypeUint32 &&
			(pattern.Type != gguf.ValueTypeArray || pattern.ArrayType != gguf.ValueTypeBool) {
			return Spec{}, errors.New("Cohere2 sliding attention pattern has an invalid type")
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
	if architecture == "gptj" || architecture == "phi2" {
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
	if architecture == "pangu-embedded" {
		spec.RopeDimensionCount = spec.KeyLength
		if value, ok := optional[uint32](values, prefix+"rope.dimension_count", gguf.ValueTypeUint32); ok {
			spec.RopeDimensionCount = value
		}
		spec.OriginalContextLength = spec.ContextLength
		if value, ok := optional[uint32](values, prefix+"rope.scaling.original_context_length", gguf.ValueTypeUint32); ok {
			spec.OriginalContextLength = value
		}
		spec.RopeAttentionFactor = 1
		if value, ok := optional[float32](values, prefix+"rope.scaling.attn_factor", gguf.ValueTypeFloat32); ok {
			spec.RopeAttentionFactor = value
		}
	}
	if architecture == "modern-bert" {
		spec.RopeDimensionCount = spec.KeyLength
		if value, ok := optional[uint32](values, prefix+"rope.dimension_count", gguf.ValueTypeUint32); ok {
			spec.RopeDimensionCount = value
		}
		spec.RopeFrequencySWA = 10000
		if value, ok := optional[float32](values, prefix+"rope.freq_base_swa", gguf.ValueTypeFloat32); ok {
			spec.RopeFrequencySWA = value
		}
		if value, ok := optional[uint32](values, prefix+"attention.sliding_window", gguf.ValueTypeUint32); ok {
			spec.SlidingWindow = value
		}
		if spec.SlidingWindow > 0 {
			spec.SlidingPattern = 3
			if value, ok := optional[uint32](values, prefix+"attention.sliding_window_pattern", gguf.ValueTypeUint32); ok {
				spec.SlidingPattern = value
			}
		}
		spec.HiddenActivation = "geglu"
		if value, ok := optional[string](values, prefix+"hidden_activation", gguf.ValueTypeString); ok {
			switch value {
			case "gelu", "geglu":
				spec.HiddenActivation = "geglu"
			case "silu", "swish", "swiglu":
				spec.HiddenActivation = "swiglu"
			case "reglu":
				spec.HiddenActivation = "reglu"
			default:
				return Spec{}, fmt.Errorf("ModernBERT hidden activation %q is unsupported", value)
			}
		}
	}
	if architecture == "gemma-embedding" {
		spec.RopeDimensionCount = spec.KeyLength
		if value, ok := optional[uint32](values, prefix+"rope.dimension_count", gguf.ValueTypeUint32); ok {
			spec.RopeDimensionCount = value
		}
		spec.RopeFrequencySWA = 10000
		if value, ok := optional[float32](values, prefix+"rope.freq_base_swa", gguf.ValueTypeFloat32); ok {
			spec.RopeFrequencySWA = value
		}
		if spec.SlidingWindow, err = required[uint32](values, prefix+"attention.sliding_window", gguf.ValueTypeUint32); err != nil {
			return Spec{}, err
		}
		spec.SlidingPattern = 6
		if value, ok := optional[uint32](values, prefix+"attention.sliding_window_pattern", gguf.ValueTypeUint32); ok {
			spec.SlidingPattern = value
		}
		spec.Dense2FeatureIn, _ = optional[uint32](values, prefix+"dense_2_feat_in", gguf.ValueTypeUint32)
		spec.Dense2FeatureOut, _ = optional[uint32](values, prefix+"dense_2_feat_out", gguf.ValueTypeUint32)
		spec.Dense3FeatureIn, _ = optional[uint32](values, prefix+"dense_3_feat_in", gguf.ValueTypeUint32)
		spec.Dense3FeatureOut, _ = optional[uint32](values, prefix+"dense_3_feat_out", gguf.ValueTypeUint32)
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
	if architecture == "glm4" || architecture == "glm4moe" {
		spec.RopeDimensionCount = spec.KeyLength
		if value, ok := optional[uint32](
			values, prefix+"rope.dimension_count", gguf.ValueTypeUint32,
		); ok {
			spec.RopeDimensionCount = value
		}
		if nextN, ok := optional[uint32](
			values, prefix+"nextn_predict_layers", gguf.ValueTypeUint32,
		); ok && nextN > 0 {
			if nextN >= spec.BlockCount {
				return Spec{}, errors.New("GLM4 NextN/MTP layer count is invalid")
			}
			spec.NextNPredictLayers = nextN
			spec.BlockCount -= nextN
		}
		if sections, ok, sectionsErr := optionalArray[int32](
			values, prefix+"rope.dimension_sections", gguf.ValueTypeInt32,
		); sectionsErr != nil {
			return Spec{}, sectionsErr
		} else if ok {
			if len(sections) != 4 {
				return Spec{}, fmt.Errorf("metadata %q has %d values, need 4", prefix+"rope.dimension_sections", len(sections))
			}
			copy(spec.RopeSections[:], sections)
		}
	}
	if architecture == "mimo2" {
		spec.RopeDimensionCount = spec.KeyLength
		if value, ok := optional[uint32](values, prefix+"rope.dimension_count", gguf.ValueTypeUint32); ok {
			spec.RopeDimensionCount = value
		}
		if nextN, ok := optional[uint32](values, prefix+"nextn_predict_layers", gguf.ValueTypeUint32); ok && nextN > 0 {
			if nextN >= spec.BlockCount {
				return Spec{}, errors.New("MiMo2 NextN/MTP layer count is invalid")
			}
			spec.NextNPredictLayers = nextN
			spec.BlockCount -= nextN
			spec.LayerKVHeadCounts = spec.LayerKVHeadCounts[:spec.BlockCount]
		}
		if spec.SlidingWindow, err = required[uint32](values, prefix+"attention.sliding_window", gguf.ValueTypeUint32); err != nil {
			return Spec{}, err
		}
		spec.RopeFrequencySWA = spec.RopeFrequencyBase
		if value, ok := optional[float32](values, prefix+"rope.freq_base_swa", gguf.ValueTypeFloat32); ok {
			spec.RopeFrequencySWA = value
		}
		patternKey := prefix + "attention.sliding_window_pattern"
		if pattern, ok := optional[uint32](values, patternKey, gguf.ValueTypeUint32); ok {
			spec.SlidingPattern = pattern
		} else {
			pattern, exists := values[patternKey]
			if !exists || pattern.Type != gguf.ValueTypeArray {
				return Spec{}, fmt.Errorf("required metadata %q is missing", patternKey)
			}
			spec.SlidingLayers = make([]bool, spec.BlockCount)
			switch pattern.ArrayType {
			case gguf.ValueTypeBool:
				layers, valid := pattern.Data.([]bool)
				if !valid || len(layers) != int(declaredBlockCount) {
					return Spec{}, fmt.Errorf("metadata %q has invalid layer values", patternKey)
				}
				copy(spec.SlidingLayers, layers)
			case gguf.ValueTypeUint32:
				layers, valid := pattern.Data.([]uint32)
				if !valid || len(layers) != int(declaredBlockCount) {
					return Spec{}, fmt.Errorf("metadata %q has invalid layer values", patternKey)
				}
				for index := range spec.SlidingLayers {
					spec.SlidingLayers[index] = layers[index] != 0
				}
			case gguf.ValueTypeInt32:
				layers, valid := pattern.Data.([]int32)
				if !valid || len(layers) != int(declaredBlockCount) {
					return Spec{}, fmt.Errorf("metadata %q has invalid layer values", patternKey)
				}
				for index := range spec.SlidingLayers {
					if layers[index] < 0 {
						return Spec{}, fmt.Errorf("metadata %q has a negative layer value", patternKey)
					}
					spec.SlidingLayers[index] = layers[index] != 0
				}
			default:
				return Spec{}, fmt.Errorf("metadata %q must be an integer or bool array", patternKey)
			}
		}
		if value, ok := optional[float32](values, prefix+"attention.value_scale", gguf.ValueTypeFloat32); ok && value != 1 {
			spec.AttentionValueScale = value
		}
	}
	if architecture == "step35" {
		spec.RopeDimensionCount = spec.KeyLength
		if value, ok := optional[uint32](values, prefix+"rope.dimension_count", gguf.ValueTypeUint32); ok {
			spec.RopeDimensionCount = value
		}
		if nextN, ok := optional[uint32](values, prefix+"nextn_predict_layers", gguf.ValueTypeUint32); ok && nextN > 0 {
			if nextN >= spec.BlockCount {
				return Spec{}, errors.New("Step3.5 NextN/MTP layer count is invalid")
			}
			spec.NextNPredictLayers = nextN
			spec.BlockCount -= nextN
		}
		if spec.SlidingWindow, err = required[uint32](values, prefix+"attention.sliding_window", gguf.ValueTypeUint32); err != nil {
			return Spec{}, err
		}
		spec.RopeFrequencySWA = spec.RopeFrequencyBase
		if value, ok := optional[float32](values, prefix+"rope.freq_base_swa", gguf.ValueTypeFloat32); ok {
			spec.RopeFrequencySWA = value
		}
		if spec.SlidingLayers, err = requiredLayerBoolCompatible(
			values, prefix+"attention.sliding_window_pattern", declaredBlockCount,
		); err != nil {
			return Spec{}, err
		}
		if spec.LayerSwiGLUClamp, err = optionalLayerFloat32(
			values, prefix+"swiglu_clamp_exp", declaredBlockCount,
		); err != nil {
			return Spec{}, err
		}
		if spec.LayerSharedSwiGLUClamp, err = optionalLayerFloat32(
			values, prefix+"swiglu_clamp_shexp", declaredBlockCount,
		); err != nil {
			return Spec{}, err
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
			if nextN >= spec.BlockCount {
				return Spec{}, errors.New("EXAONE 4 NextN/MTP layer count is invalid")
			}
			spec.NextNPredictLayers = nextN
			spec.BlockCount -= nextN
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
			spec.NextNPredictLayers = nextN
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
			spec.SlidingLayers = slices.Clone(layers)
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
			spec.NextNPredictLayers = nextN
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
	if architecture == "baichuan" && spec.BlockCount != 32 && spec.BlockCount != 40 {
		return Spec{}, errors.New("Baichuan block count must select the 32-layer RoPE or 40-layer ALiBi variant")
	}
	if architecture == "mistral3" {
		spec.OriginalContextLength = spec.ContextLength
		if value, ok := optional[uint32](
			values, prefix+"rope.scaling.original_context_length", gguf.ValueTypeUint32,
		); ok {
			spec.OriginalContextLength = value
		}
		spec.AttentionTempScale, _ = optional[float32](
			values, prefix+"attention.temperature_scale", gguf.ValueTypeFloat32,
		)
		if spec.AttentionTempScale != 0 {
			spec.AttentionTempFloor = spec.OriginalContextLength
		}
		if spec.RopeScalingType == "yarn" {
			spec.RopeYaRNLogMultiplier, _ = optional[float32](
				values, prefix+"rope.scaling.yarn_log_multiplier", gguf.ValueTypeFloat32,
			)
			rawAttentionFactor := float32(1)
			if value, ok := optional[float32](
				values, prefix+"rope.scaling.attn_factor", gguf.ValueTypeFloat32,
			); ok {
				rawAttentionFactor = value
			}
			denominator := float32(1)
			if spec.RopeYaRNLogMultiplier != 0 {
				denominator += 0.1 * spec.RopeYaRNLogMultiplier *
					float32(math.Log(float64(spec.RopeScalingFactor)))
			}
			spec.YaRNAttentionFactor = rawAttentionFactor / denominator
		}
		spec.ExpertCount, _ = optional[uint32](values, prefix+"expert_count", gguf.ValueTypeUint32)
		if spec.ExpertCount > 0 {
			if spec.ExpertUsedCount, err = required[uint32](
				values, prefix+"expert_used_count", gguf.ValueTypeUint32,
			); err != nil {
				return Spec{}, err
			}
			spec.ExpertFeedForward = spec.FeedForwardLength
			spec.ExpertWeightsNorm = true
			spec.ExpertWeightsScale = 1
			if value, ok := optional[float32](
				values, prefix+"expert_weights_scale", gguf.ValueTypeFloat32,
			); ok && value != 0 {
				spec.ExpertWeightsScale = value
			}
		}
	}
	if architecture == "qwen" {
		if spec.FeedForwardLength == 0 || spec.FeedForwardLength%2 != 0 {
			return Spec{}, errors.New("Qwen feed-forward length must be positive and even")
		}
		spec.FeedForwardLength /= 2
		spec.RopeDimensionCount = spec.KeyLength
		if value, ok := optional[uint32](values, prefix+"rope.dimension_count", gguf.ValueTypeUint32); ok {
			spec.RopeDimensionCount = value
		}
	}
	if architecture == "chatglm" {
		spec.RopeDimensionCount = spec.KeyLength
		if value, ok := optional[uint32](values, prefix+"rope.dimension_count", gguf.ValueTypeUint32); ok {
			spec.RopeDimensionCount = value
		}
	}
	if architecture == "hunyuan-dense" || architecture == "hunyuan_vl" {
		spec.RopeDimensionCount = spec.KeyLength
		if value, ok := optional[uint32](values, prefix+"rope.dimension_count", gguf.ValueTypeUint32); ok {
			spec.RopeDimensionCount = value
		}
		if sections, ok, sectionsErr := optionalArray[int32](values, prefix+"rope.dimension_sections", gguf.ValueTypeInt32); sectionsErr != nil {
			return Spec{}, sectionsErr
		} else if ok {
			if len(sections) != 4 {
				return Spec{}, fmt.Errorf("metadata %q has %d values, need 4", prefix+"rope.dimension_sections", len(sections))
			}
			copy(spec.RopeSections[:], sections)
		}
		if alpha, ok := optional[float32](values, prefix+"rope.scaling.alpha", gguf.ValueTypeFloat32); ok && alpha != 0 {
			if alpha < 0 || spec.KeyLength <= 2 || math.IsNaN(float64(alpha)) || math.IsInf(float64(alpha), 0) {
				return Spec{}, errors.New("Hunyuan-Dense XDRoPE alpha is invalid")
			}
			exponent := float64(spec.KeyLength) / float64(spec.KeyLength-2)
			spec.RopeFrequencyBase *= float32(math.Pow(float64(alpha), exponent))
		}
	}
	if architecture == "paddleocr" || architecture == "qwen2vl" || architecture == "qwen3vl" || architecture == "qwen3vlmoe" {
		spec.RopeDimensionCount = spec.KeyLength
		sections, sectionsErr := requiredArray[int32](
			values, prefix+"rope.dimension_sections", gguf.ValueTypeInt32,
		)
		if sectionsErr != nil {
			return Spec{}, sectionsErr
		}
		if len(sections) != len(spec.RopeSections) {
			return Spec{}, fmt.Errorf(
				"metadata %q has %d values, need %d",
				prefix+"rope.dimension_sections", len(sections), len(spec.RopeSections),
			)
		}
		copy(spec.RopeSections[:], sections)
	}
	if architecture == "qwen3vl" || architecture == "qwen3vlmoe" {
		if value, ok := optional[uint32](values, prefix+"n_deepstack_layers", gguf.ValueTypeUint32); ok {
			spec.DeepstackLayerCount = value
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
				spec.SlidingLayers = slices.Clone(layers)
			}
			spec.RopeFrequencySWA = spec.RopeFrequencyBase
			if value, ok := optional[float32](values, prefix+"rope.freq_base_swa", gguf.ValueTypeFloat32); ok {
				spec.RopeFrequencySWA = value
			}
		}
	}
	if architecture == "plamo3" {
		spec.RopeDimensionCount = spec.KeyLength
		if value, ok := optional[uint32](values, prefix+"attention.sliding_window", gguf.ValueTypeUint32); ok {
			spec.SlidingWindow = value
		}
		if spec.SlidingWindow > 0 {
			spec.SlidingPattern = 8
			if value, ok := optional[uint32](values, prefix+"attention.sliding_window_pattern", gguf.ValueTypeUint32); ok {
				spec.SlidingPattern = value
			} else if layers, ok, arrayErr := optionalArray[bool](values, prefix+"attention.sliding_window_pattern", gguf.ValueTypeBool); arrayErr != nil {
				return Spec{}, arrayErr
			} else if ok {
				if len(layers) != int(spec.BlockCount) {
					return Spec{}, fmt.Errorf("metadata %q has %d values, need %d", prefix+"attention.sliding_window_pattern", len(layers), spec.BlockCount)
				}
				spec.SlidingLayers = slices.Clone(layers)
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
		); ok {
			spec.AttentionClamp = clamp
		}
	}
	if architecture == "t5" || architecture == "t5encoder" {
		if spec.RelativeBuckets, err = required[uint32](
			values,
			prefix+"attention.relative_buckets_count",
			gguf.ValueTypeUint32,
		); err != nil {
			return Spec{}, err
		}
		if architecture == "t5" {
			spec.DecoderBlockCount = spec.BlockCount
			if value, ok := optional[uint32](values, prefix+"decoder_block_count", gguf.ValueTypeUint32); ok {
				spec.DecoderBlockCount = value
			}
			spec.DecoderStartTokenID, _ = optional[uint32](values, prefix+"decoder_start_token_id", gguf.ValueTypeUint32)
		}
	}
	if architecture == "qwen3next" || architecture == "qwen35" || architecture == "qwen35moe" {
		if architecture == "qwen3next" {
			spec.RopeDimensionCount = spec.KeyLength
			if value, ok := optional[uint32](
				values, prefix+"rope.dimension_count", gguf.ValueTypeUint32,
			); ok {
				spec.RopeDimensionCount = value
			}
		} else {
			if spec.RopeDimensionCount, err = required[uint32](
				values, prefix+"rope.dimension_count", gguf.ValueTypeUint32,
			); err != nil {
				return Spec{}, err
			}
			sections, sectionsErr := requiredArray[int32](
				values, prefix+"rope.dimension_sections", gguf.ValueTypeInt32,
			)
			if sectionsErr != nil {
				return Spec{}, sectionsErr
			}
			if len(sections) != len(spec.RopeSections) {
				return Spec{}, fmt.Errorf(
					"metadata %q has %d values, need %d",
					prefix+"rope.dimension_sections", len(sections), len(spec.RopeSections),
				)
			}
			copy(spec.RopeSections[:], sections)
		}
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
			if len(recurrent) != int(spec.BlockCount) && len(recurrent) != int(declaredBlockCount) {
				return Spec{}, fmt.Errorf(
					"metadata %q has %d values, need %d or %d",
					prefix+"attention.recurrent_layers",
					len(recurrent),
					spec.BlockCount,
					declaredBlockCount,
				)
			}
			spec.RecurrentLayers = slices.Clone(recurrent[:spec.BlockCount])
		}
	}
	if architecture == "kimi-linear" {
		if spec.KVLoRARank, err = required[uint32](
			values, prefix+"attention.kv_lora_rank", gguf.ValueTypeUint32,
		); err != nil {
			return Spec{}, err
		}
		spec.QLoRARank, _ = optional[uint32](values, prefix+"attention.q_lora_rank", gguf.ValueTypeUint32)
		if spec.RopeDimensionCount, err = required[uint32](
			values, prefix+"rope.dimension_count", gguf.ValueTypeUint32,
		); err != nil {
			return Spec{}, err
		}
		if spec.SSMConvKernel, err = required[uint32](
			values, prefix+"ssm.conv_kernel", gguf.ValueTypeUint32,
		); err != nil {
			return Spec{}, err
		}
		if spec.KDAHeadDim, err = required[uint32](
			values, prefix+"kda.head_dim", gguf.ValueTypeUint32,
		); err != nil {
			return Spec{}, err
		}
		spec.SSMInnerSize = spec.HeadCount * spec.KDAHeadDim
		spec.SSMStateSize = spec.KDAHeadDim
		spec.SSMTimeStepRank = spec.HeadCount
		spec.SSMGroupCount = spec.HeadCount
	}
	if architecture == "rwkv6" || architecture == "rwkv6qwen2" {
		for key, destination := range map[string]*uint32{
			"wkv.head_size":        &spec.WKVHeadSize,
			"time_mix_extra_dim":   &spec.TimeMixExtraDim,
			"time_decay_extra_dim": &spec.TimeDecayExtraDim,
		} {
			*destination, err = required[uint32](values, prefix+key, gguf.ValueTypeUint32)
			if err != nil {
				return Spec{}, err
			}
		}
		spec.RescaleEvery, _ = optional[uint32](values, prefix+"rescale_every_n_layers", gguf.ValueTypeUint32)
		spec.TokenShiftCount = 1
		if architecture == "rwkv6" {
			spec.TokenShiftCount = 2
		}
		if value, ok := optional[uint32](values, prefix+"token_shift_count", gguf.ValueTypeUint32); ok {
			spec.TokenShiftCount = value
		}
	}
	if architecture == "rwkv7" || architecture == "arwkv7" {
		for key, destination := range map[string]*uint32{
			"wkv.head_size":                          &spec.WKVHeadSize,
			"attention.decay_lora_rank":              &spec.DecayLoRARank,
			"attention.iclr_lora_rank":               &spec.ICLRLoRARank,
			"attention.value_residual_mix_lora_rank": &spec.ValueMixLoRARank,
		} {
			*destination, err = required[uint32](values, prefix+key, gguf.ValueTypeUint32)
			if err != nil {
				return Spec{}, err
			}
		}
		spec.GateLoRARank, _ = optional[uint32](values, prefix+"attention.gate_lora_rank", gguf.ValueTypeUint32)
		spec.TokenShiftCount = 1
		if architecture == "rwkv7" {
			spec.TokenShiftCount = 2
		}
		if value, ok := optional[uint32](values, prefix+"token_shift_count", gguf.ValueTypeUint32); ok {
			spec.TokenShiftCount = value
		}
	}
	if architecture == "mamba" || architecture == "mamba2" || architecture == "jamba" || architecture == "granitehybrid" || architecture == "plamo2" || architecture == "nemotron_h" || architecture == "nemotron_h_moe" || architecture == "falcon-h1" {
		for key, destination := range map[string]*uint32{
			"ssm.conv_kernel":    &spec.SSMConvKernel,
			"ssm.inner_size":     &spec.SSMInnerSize,
			"ssm.state_size":     &spec.SSMStateSize,
			"ssm.time_step_rank": &spec.SSMTimeStepRank,
		} {
			*destination, err = required[uint32](values, prefix+key, gguf.ValueTypeUint32)
			if err != nil {
				return Spec{}, err
			}
		}
		if architecture == "mamba" || architecture == "jamba" {
			spec.SSMGroupCount = 1
			spec.SSMDtBCNorm, _ = optional[bool](values, prefix+"ssm.dt_b_c_rms", gguf.ValueTypeBool)
		} else if spec.SSMGroupCount, err = required[uint32](values, prefix+"ssm.group_count", gguf.ValueTypeUint32); err != nil {
			return Spec{}, err
		}
	}
	if architecture == "falcon-h1" {
		spec.RopeDimensionCount = spec.KeyLength
		if value, ok := optional[uint32](values, prefix+"rope.dimension_count", gguf.ValueTypeUint32); ok {
			spec.RopeDimensionCount = value
		}
	}
	if architecture == "plamo2" {
		spec.AttentionScale = float32(1 / math.Sqrt(float64(spec.ValueLength)))
	}
	if architecture == "granitehybrid" {
		spec.ExpertCount, _ = optional[uint32](values, prefix+"expert_count", gguf.ValueTypeUint32)
		if spec.ExpertCount > 0 {
			if spec.ExpertUsedCount, err = required[uint32](values, prefix+"expert_used_count", gguf.ValueTypeUint32); err != nil {
				return Spec{}, err
			}
			spec.ExpertFeedForward = spec.FeedForwardLength
			spec.ExpertWeightsScale = 1
			spec.ExpertWeightsNorm = true
			spec.SharedExpertFF, _ = optional[uint32](values, prefix+"expert_shared_feed_forward_length", gguf.ValueTypeUint32)
		}
	}
	if architecture == "nemotron_h_moe" {
		for key, destination := range map[string]*uint32{
			"expert_count":                      &spec.ExpertCount,
			"expert_used_count":                 &spec.ExpertUsedCount,
			"expert_feed_forward_length":        &spec.ExpertFeedForward,
			"expert_shared_feed_forward_length": &spec.SharedExpertFF,
		} {
			*destination, err = required[uint32](values, prefix+key, gguf.ValueTypeUint32)
			if err != nil {
				return Spec{}, err
			}
		}
		spec.SharedExpertCount, _ = optional[uint32](values, prefix+"expert_shared_count", gguf.ValueTypeUint32)
		spec.ExpertWeightsNorm, _ = optional[bool](values, prefix+"expert_weights_norm", gguf.ValueTypeBool)
		spec.ExpertWeightsScale = 1
		if value, ok := optional[float32](values, prefix+"expert_weights_scale", gguf.ValueTypeFloat32); ok {
			spec.ExpertWeightsScale = value
		}
		spec.MoELatentSize, _ = optional[uint32](values, prefix+"moe_latent_size", gguf.ValueTypeUint32)
		spec.ExpertGatingFunc = 2
	}
	return spec, nil
}

func (m specMetadata) readExpertMetadata(spec Spec, state specReadState) (Spec, error) {
	values, architecture, prefix, profile := m.values, m.architecture, m.prefix, m.profile
	isLlamaMoE := state.llamaMoE
	var err error
	if isLlamaMoE || architecture == "llama4" || architecture == "gpt-oss" || architecture == "arctic" || architecture == "bailingmoe" || architecture == "bailingmoe2" || architecture == "cohere2moe" || architecture == "deepseek" || architecture == "deepseek2-ocr" || architecture == "dbrx" || architecture == "dots1" || architecture == "ernie4_5-moe" || architecture == "glm4moe" || architecture == "granitemoe" || (architecture == "granite" && spec.ExpertCount > 0) || architecture == "grovemoe" || architecture == "grok" || architecture == "hunyuan-moe" || architecture == "hy_v3" || architecture == "jamba" || architecture == "kimi-linear" || architecture == "llada-moe" || architecture == "mellum" || architecture == "mimo2" || architecture == "step35" || architecture == "minimax-m2" || architecture == "nomic-bert-moe" || architecture == "qwen3moe" || architecture == "qwen3vlmoe" || architecture == "qwen3next" || architecture == "qwen35moe" || architecture == "qwen2moe" || architecture == "olmoe" || architecture == "phimoe" || architecture == "exaone-moe" || architecture == "rnd1" || architecture == "afmoe" || architecture == "laguna" || architecture == "lfm2moe" || architecture == "smallthinker" {
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
		if architecture == "nomic-bert-moe" {
			spec.ExpertFeedForward = spec.FeedForwardLength
		}
		if architecture == "jamba" {
			spec.ExpertFeedForward = spec.FeedForwardLength
			spec.ExpertWeightsNorm = false
		}
		if architecture == "kimi-linear" {
			if spec.ExpertFeedForward, err = required[uint32](
				values, prefix+"expert_feed_forward_length", gguf.ValueTypeUint32,
			); err != nil {
				return Spec{}, err
			}
			spec.SharedExpertCount, _ = optional[uint32](values, prefix+"expert_shared_count", gguf.ValueTypeUint32)
			spec.SharedExpertFF = spec.ExpertFeedForward * spec.SharedExpertCount
			spec.LeadingDenseBlocks, _ = optional[uint32](values, prefix+"leading_dense_block_count", gguf.ValueTypeUint32)
			if spec.ExpertGatingFunc, err = required[uint32](
				values, prefix+"expert_gating_func", gguf.ValueTypeUint32,
			); err != nil {
				return Spec{}, err
			}
			spec.ExpertWeightsNorm = true
		}
		if architecture == "llama4" {
			if spec.ExpertFeedForward, err = required[uint32](
				values, prefix+"expert_feed_forward_length", gguf.ValueTypeUint32,
			); err != nil {
				return Spec{}, err
			}
			if spec.MoELayerStep, err = required[uint32](
				values, prefix+"interleave_moe_layer_step", gguf.ValueTypeUint32,
			); err != nil {
				return Spec{}, err
			}
			spec.SharedExpertFF = spec.ExpertFeedForward
			spec.ExpertGatingFunc = 2
		}
		spec.ExpertWeightsScale = 1
		if architecture == "gpt-oss" {
			spec.ExpertGatingFunc = 3
			spec.ExpertWeightsNorm = false
		}
		if value, ok := optional[float32](
			values, prefix+"expert_weights_scale", gguf.ValueTypeFloat32,
		); ok {
			spec.ExpertWeightsScale = value
		}
		if architecture == "qwen3next" || architecture == "qwen35moe" {
			spec.SharedExpertFF = spec.FeedForwardLength
			if value, ok := optional[uint32](
				values, prefix+"expert_shared_feed_forward_length", gguf.ValueTypeUint32,
			); ok {
				spec.SharedExpertFF = value
			}
		}
	}
	if profile.Has(ArchitectureDeepSeek2Layout) {
		if architecture == "deepseek32" {
			if spec.ExpertCount, err = required[uint32](values, prefix+"expert_count", gguf.ValueTypeUint32); err != nil {
				return Spec{}, err
			}
		} else {
			spec.ExpertCount, _ = optional[uint32](values, prefix+"expert_count", gguf.ValueTypeUint32)
		}
		if spec.ExpertFeedForward, err = required[uint32](values, prefix+"expert_feed_forward_length", gguf.ValueTypeUint32); err != nil {
			return Spec{}, err
		}
		if spec.ExpertCount > 0 {
			if spec.ExpertUsedCount, err = required[uint32](values, prefix+"expert_used_count", gguf.ValueTypeUint32); err != nil {
				return Spec{}, err
			}
		}
		spec.ExpertWeightsScale = 1
		if value, ok := optional[float32](values, prefix+"expert_weights_scale", gguf.ValueTypeFloat32); ok {
			spec.ExpertWeightsScale = value
		}
		spec.ExpertWeightsNorm, _ = optional[bool](values, prefix+"expert_weights_norm", gguf.ValueTypeBool)
		spec.ExpertGatingFunc = 1
		if architecture == "glm-dsa" {
			spec.ExpertGatingFunc = 2
		}
		if architecture == "deepseek32" {
			if spec.ExpertGatingFunc, err = required[uint32](values, prefix+"expert_gating_func", gguf.ValueTypeUint32); err != nil {
				return Spec{}, err
			}
		} else if value, ok := optional[uint32](values, prefix+"expert_gating_func", gguf.ValueTypeUint32); ok && value != 0 {
			spec.ExpertGatingFunc = value
		} else if (spec.BlockCount == 47 || spec.BlockCount == 48) && spec.VocabularySize == 154880 {
			spec.ExpertGatingFunc = 2
		}
	}
	if architecture == "mimo2" {
		spec.ExpertGatingFunc = 2
		spec.ExpertWeightsNorm = true
	}
	if architecture == "gemma4" {
		if count, ok := optional[uint32](values, prefix+"expert_count", gguf.ValueTypeUint32); ok && count > 0 {
			spec.ExpertCount = count
			if spec.ExpertUsedCount, err = required[uint32](
				values, prefix+"expert_used_count", gguf.ValueTypeUint32,
			); err != nil {
				return Spec{}, err
			}
			if spec.ExpertFeedForward, err = required[uint32](
				values, prefix+"expert_feed_forward_length", gguf.ValueTypeUint32,
			); err != nil {
				return Spec{}, err
			}
			spec.ExpertWeightsScale = 1
			spec.ExpertWeightsNorm = true
		}
	}
	if architecture == "step35" {
		if spec.ExpertFeedForward, err = required[uint32](
			values, prefix+"expert_feed_forward_length", gguf.ValueTypeUint32,
		); err != nil {
			return Spec{}, err
		}
		spec.SharedExpertFF, _ = optional[uint32](
			values, prefix+"expert_shared_feed_forward_length", gguf.ValueTypeUint32,
		)
		spec.LeadingDenseBlocks, _ = optional[uint32](
			values, prefix+"leading_dense_block_count", gguf.ValueTypeUint32,
		)
		spec.MoELayerStep = 1
		if value, ok := optional[uint32](values, prefix+"moe_every_n_layers", gguf.ValueTypeUint32); ok {
			spec.MoELayerStep = value
		}
		spec.ExpertGatingFunc = 2
		if value, ok := optional[uint32](values, prefix+"expert_gating_func", gguf.ValueTypeUint32); ok && value != 0 {
			spec.ExpertGatingFunc = value
		}
		spec.ExpertWeightsNorm, _ = optional[bool](values, prefix+"expert_weights_norm", gguf.ValueTypeBool)
	}
	if architecture == "glm4moe" {
		spec.LeadingDenseBlocks, _ = optional[uint32](values, prefix+"leading_dense_block_count", gguf.ValueTypeUint32)
		if spec.SharedExpertCount, err = required[uint32](values, prefix+"expert_shared_count", gguf.ValueTypeUint32); err != nil {
			return Spec{}, err
		}
		if spec.ExpertFeedForward > 0 && spec.SharedExpertCount > math.MaxUint32/spec.ExpertFeedForward {
			return Spec{}, errors.New("GLM4-MoE shared expert width overflows")
		}
		spec.SharedExpertFF = spec.ExpertFeedForward * spec.SharedExpertCount
		spec.ExpertGatingFunc = 2
		if value, ok := optional[uint32](values, prefix+"expert_gating_func", gguf.ValueTypeUint32); ok {
			spec.ExpertGatingFunc = value
		}
		if value, ok := optional[bool](values, prefix+"expert_weights_norm", gguf.ValueTypeBool); ok {
			spec.ExpertWeightsNorm = value
		}
	}
	if architecture == "grovemoe" {
		spec.ExpertChunkFeedForward = spec.KeyLength
		if value, ok := optional[uint32](values, prefix+"expert_chunk_feed_forward_length", gguf.ValueTypeUint32); ok {
			spec.ExpertChunkFeedForward = value
		}
		if spec.ExpertGroupScale, err = required[float32](values, prefix+"expert_group_scale", gguf.ValueTypeFloat32); err != nil {
			return Spec{}, err
		}
		if spec.ExpertsPerGroup, err = required[uint32](values, prefix+"experts_per_group", gguf.ValueTypeUint32); err != nil {
			return Spec{}, err
		}
	}
	if architecture == "cohere2moe" {
		if nextN, ok := optional[uint32](values, prefix+"nextn_predict_layers", gguf.ValueTypeUint32); ok && nextN > 0 {
			if nextN != 1 || nextN >= spec.BlockCount {
				return Spec{}, errors.New("Cohere2-MoE NextN/MTP layer count is invalid")
			}
			spec.NextNPredictLayers = nextN
			spec.BlockCount -= nextN
		}
		if spec.LeadingDenseBlocks, err = required[uint32](values, prefix+"leading_dense_block_count", gguf.ValueTypeUint32); err != nil {
			return Spec{}, err
		}
		if spec.ExpertFeedForward, err = required[uint32](values, prefix+"expert_feed_forward_length", gguf.ValueTypeUint32); err != nil {
			return Spec{}, err
		}
		spec.ExpertGatingFunc = 2
		if value, ok := optional[uint32](values, prefix+"expert_gating_func", gguf.ValueTypeUint32); ok {
			spec.ExpertGatingFunc = value
		}
		if value, ok := optional[bool](values, prefix+"expert_weights_norm", gguf.ValueTypeBool); ok {
			spec.ExpertWeightsNorm = value
		}
		if value, ok := optional[uint32](values, prefix+"expert_shared_count", gguf.ValueTypeUint32); ok {
			spec.SharedExpertCount = value
		}
		if spec.SharedExpertCount > 0 {
			spec.SharedExpertFF = spec.ExpertFeedForward * spec.SharedExpertCount
			if value, ok := optional[uint32](values, prefix+"expert_shared_feed_forward_length", gguf.ValueTypeUint32); ok {
				spec.SharedExpertFF = value
			}
		}
	}
	if architecture == "hy_v3" {
		if nextN, ok := optional[uint32](values, prefix+"nextn_predict_layers", gguf.ValueTypeUint32); ok && nextN > 0 {
			if nextN >= spec.BlockCount {
				return Spec{}, errors.New("HY-V3 NextN/MTP layer count is invalid")
			}
			spec.NextNPredictLayers = nextN
			spec.BlockCount -= nextN
		}
		if spec.ExpertFeedForward, err = required[uint32](values, prefix+"expert_feed_forward_length", gguf.ValueTypeUint32); err != nil {
			return Spec{}, err
		}
		spec.SharedExpertFF = spec.ExpertFeedForward
		if value, ok := optional[uint32](values, prefix+"expert_shared_feed_forward_length", gguf.ValueTypeUint32); ok {
			spec.SharedExpertFF = value
		}
		spec.ExpertGatingFunc = 2
		if value, ok := optional[uint32](values, prefix+"expert_gating_func", gguf.ValueTypeUint32); ok {
			spec.ExpertGatingFunc = value
		}
		if value, ok := optional[bool](values, prefix+"expert_weights_norm", gguf.ValueTypeBool); ok {
			spec.ExpertWeightsNorm = value
		}
		spec.RopeDimensionCount = spec.KeyLength
	}
	if architecture == "deepseek2-ocr" {
		spec.LeadingDenseBlocks, _ = optional[uint32](values, prefix+"leading_dense_block_count", gguf.ValueTypeUint32)
		if spec.ExpertFeedForward, err = required[uint32](values, prefix+"expert_feed_forward_length", gguf.ValueTypeUint32); err != nil {
			return Spec{}, err
		}
		if spec.SharedExpertCount, err = required[uint32](values, prefix+"expert_shared_count", gguf.ValueTypeUint32); err != nil {
			return Spec{}, err
		}
		if spec.SharedExpertCount > math.MaxUint32/spec.ExpertFeedForward {
			return Spec{}, errors.New("DeepSeek2-OCR shared expert width overflows")
		}
		spec.SharedExpertFF = spec.ExpertFeedForward * spec.SharedExpertCount
		spec.ExpertGatingFunc = 1
		if value, ok := optional[uint32](values, prefix+"expert_gating_func", gguf.ValueTypeUint32); ok {
			spec.ExpertGatingFunc = value
		}
		if value, ok := optional[bool](values, prefix+"expert_weights_norm", gguf.ValueTypeBool); ok {
			spec.ExpertWeightsNorm = value
		}
		spec.RopeDimensionCount = spec.KeyLength
	}
	if architecture == "neo-bert" {
		spec.RopeDimensionCount = spec.KeyLength
	}
	if architecture == "jina-bert-v3" {
		spec.RopeDimensionCount = spec.KeyLength
		if value, ok := optional[uint32](values, prefix+"rope.dimension_count", gguf.ValueTypeUint32); ok {
			spec.RopeDimensionCount = value
		}
		spec.ExpertCount, _ = optional[uint32](values, prefix+"expert_count", gguf.ValueTypeUint32)
		spec.ExpertUsedCount, _ = optional[uint32](values, prefix+"expert_used_count", gguf.ValueTypeUint32)
		spec.MoELayerStep, _ = optional[uint32](values, prefix+"moe_every_n_layers", gguf.ValueTypeUint32)
		if spec.ExpertCount > 0 {
			spec.ExpertFeedForward = spec.FeedForwardLength
			spec.ExpertWeightsScale = 1
			if value, ok := optional[float32](values, prefix+"expert_weights_scale", gguf.ValueTypeFloat32); ok {
				spec.ExpertWeightsScale = value
			}
		}
	}
	if architecture == "nomic-bert" {
		spec.RopeDimensionCount = spec.KeyLength
		if value, ok := optional[uint32](values, prefix+"rope.dimension_count", gguf.ValueTypeUint32); ok {
			spec.RopeDimensionCount = value
		}
		if cadence, ok := optional[uint32](values, prefix+"moe_every_n_layers", gguf.ValueTypeUint32); ok && cadence > 0 {
			return Spec{}, errors.New("NomicBERT MoE cadence requires nomic-bert-moe architecture")
		}
	}
	if architecture == "nomic-bert-moe" {
		spec.RopeDimensionCount = spec.KeyLength
		if value, ok := optional[uint32](values, prefix+"rope.dimension_count", gguf.ValueTypeUint32); ok {
			spec.RopeDimensionCount = value
		}
		if spec.MoELayerStep, err = required[uint32](values, prefix+"moe_every_n_layers", gguf.ValueTypeUint32); err != nil {
			return Spec{}, err
		}
	}
	if architecture == "ernie4_5-moe" {
		if spec.ExpertFeedForward, err = required[uint32](values, prefix+"expert_feed_forward_length", gguf.ValueTypeUint32); err != nil {
			return Spec{}, err
		}
		if spec.MoELayerStep, err = required[uint32](values, prefix+"interleave_moe_layer_step", gguf.ValueTypeUint32); err != nil {
			return Spec{}, err
		}
		spec.LeadingDenseBlocks, _ = optional[uint32](values, prefix+"leading_dense_block_count", gguf.ValueTypeUint32)
		spec.SharedExpertFF, _ = optional[uint32](values, prefix+"expert_shared_feed_forward_length", gguf.ValueTypeUint32)
		spec.ExpertWeightsNorm = true
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
	if architecture == "granitemoe" || (architecture == "granite" && spec.ExpertCount > 0) {
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
	return spec, nil
}

func (m specMetadata) readRuntimeMetadata(spec Spec, _ specReadState) (Spec, error) {
	values, architecture, prefix, profile := m.values, m.architecture, m.prefix, m.profile
	var err error
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
	if profile.Attention == AttentionMLA || profile.Attention == AttentionDSA {
		if architecture == "minicpm3" {
			if spec.QLoRARank, err = required[uint32](
				values, prefix+"attention.q_lora_rank", gguf.ValueTypeUint32,
			); err != nil {
				return Spec{}, err
			}
		}
		if profile.Has(ArchitectureDeepSeek2Layout) {
			lite := spec.BlockCount == 26 || spec.BlockCount == 27 || (spec.BlockCount == 48 && spec.VocabularySize == 128256)
			if !lite {
				if spec.QLoRARank, err = required[uint32](values, prefix+"attention.q_lora_rank", gguf.ValueTypeUint32); err != nil {
					return Spec{}, err
				}
			}
		}
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
		if profile.Has(ArchitectureDeepSeek2Layout) {
			spec.LeadingDenseBlocks, _ = optional[uint32](values, prefix+"leading_dense_block_count", gguf.ValueTypeUint32)
			if spec.SharedExpertCount, err = required[uint32](values, prefix+"expert_shared_count", gguf.ValueTypeUint32); err != nil {
				return Spec{}, err
			}
			if spec.ExpertFeedForward > 0 && spec.SharedExpertCount > math.MaxUint32/spec.ExpertFeedForward {
				return Spec{}, errors.New("DeepSeek2 shared expert width overflows")
			}
			spec.SharedExpertFF = spec.ExpertFeedForward * spec.SharedExpertCount
			if value, ok := optional[float32](values, prefix+"rope.scaling.yarn_log_multiplier", gguf.ValueTypeFloat32); ok {
				spec.RopeYaRNLogMultiplier = value / 0.1
			}
			if spec.RopeScalingType == "yarn" && spec.RopeScalingFactor > 0 {
				rawAttentionFactor := float32(1)
				if value, ok := optional[float32](values, prefix+"rope.scaling.attn_factor", gguf.ValueTypeFloat32); ok {
					rawAttentionFactor = value
				}
				spec.YaRNAttentionFactor = rawAttentionFactor /
					(1 + 0.1*float32(math.Log(float64(spec.RopeScalingFactor))))
			}
			spec.AttentionTempScale, _ = optional[float32](values, prefix+"attention.temperature_scale", gguf.ValueTypeFloat32)
			spec.AttentionTempFloor, _ = optional[uint32](values, prefix+"attention.temperature_length", gguf.ValueTypeUint32)
		}
	}
	if profile.Attention == AttentionDSA {
		sections, hasSections, sectionsErr := optionalArray[int32](values, prefix+"rope.dimension_sections", gguf.ValueTypeInt32)
		if sectionsErr != nil {
			return Spec{}, sectionsErr
		}
		if hasSections && len(sections) != len(spec.RopeSections) {
			return Spec{}, fmt.Errorf("metadata %q has %d values, need %d", prefix+"rope.dimension_sections", len(sections), len(spec.RopeSections))
		}
		if hasSections {
			copy(spec.RopeSections[:], sections)
		}
		for key, destination := range map[string]*uint32{
			"attention.indexer.head_count": &spec.IndexerHeadCount,
			"attention.indexer.key_length": &spec.IndexerKeyLength,
			"attention.indexer.top_k":      &spec.IndexerTopK,
		} {
			*destination, err = required[uint32](values, prefix+key, gguf.ValueTypeUint32)
			if err != nil {
				return Spec{}, err
			}
		}
		spec.IndexerFullLayers = make([]bool, spec.BlockCount)
		if architecture == "deepseek32" || spec.ContextLength < deepSeekDenseIndexerContext {
			for index := range spec.IndexerFullLayers {
				spec.IndexerFullLayers[index] = true
			}
		} else {
			for index := range spec.IndexerFullLayers {
				spec.IndexerFullLayers[index] = index < deepSeekInitialFullIndexers ||
					(index-deepSeekInitialFullIndexers)%deepSeekFullIndexerPeriod == 0
			}
		}
		indexerTypesKey := prefix + "attention.indexer.types"
		if value, present := values[indexerTypesKey]; present && value.Type == gguf.ValueTypeUint32 {
			typeValue, valid := value.Data.(uint32)
			if !valid || typeValue > 1 {
				return Spec{}, fmt.Errorf("metadata %q must be 0 or 1", indexerTypesKey)
			}
			for index := range spec.IndexerFullLayers {
				spec.IndexerFullLayers[index] = typeValue == 1
			}
		} else if types, ok, arrayErr := optionalArray[uint32](values, indexerTypesKey, gguf.ValueTypeUint32); arrayErr != nil {
			return Spec{}, arrayErr
		} else if ok {
			if len(types) != int(spec.BlockCount) {
				return Spec{}, fmt.Errorf("metadata %q has %d values, need %d", indexerTypesKey, len(types), spec.BlockCount)
			}
			for index, typeValue := range types {
				if typeValue > 1 {
					return Spec{}, fmt.Errorf("metadata %q value %d is not 0 or 1", indexerTypesKey, typeValue)
				}
				spec.IndexerFullLayers[index] = typeValue == 1
			}
		}
	}
	if architecture == "deepseek4" {
		if spec.QLoRARank, err = required[uint32](values, prefix+"attention.q_lora_rank", gguf.ValueTypeUint32); err != nil {
			return Spec{}, err
		}
		if spec.RopeDimensionCount, err = required[uint32](values, prefix+"rope.dimension_count", gguf.ValueTypeUint32); err != nil {
			return Spec{}, err
		}
		for key, destination := range map[string]*uint32{
			"attention.sliding_window":             &spec.SlidingWindow,
			"attention.indexer.head_count":         &spec.IndexerHeadCount,
			"attention.indexer.key_length":         &spec.IndexerKeyLength,
			"attention.indexer.top_k":              &spec.IndexerTopK,
			"attention.output_group_count":         &spec.AttentionOutputGroups,
			"attention.output_lora_rank":           &spec.AttentionOutputRank,
			"hyper_connection.count":               &spec.HyperConnectionCount,
			"hyper_connection.sinkhorn_iterations": &spec.HyperSinkhornIters,
			"hash_layer_count":                     &spec.HashLayerCount,
			"expert_count":                         &spec.ExpertCount,
			"expert_used_count":                    &spec.ExpertUsedCount,
			"expert_feed_forward_length":           &spec.ExpertFeedForward,
			"expert_shared_count":                  &spec.SharedExpertCount,
			"expert_gating_func":                   &spec.ExpertGatingFunc,
		} {
			*destination, err = required[uint32](values, prefix+key, gguf.ValueTypeUint32)
			if err != nil {
				return Spec{}, err
			}
		}
		for key, destination := range map[string]*float32{
			"attention.compress_rope_freq_base": &spec.CompressRopeBase,
			"hyper_connection.epsilon":          &spec.HyperConnectionEps,
			"expert_weights_scale":              &spec.ExpertWeightsScale,
		} {
			*destination, err = required[float32](values, prefix+key, gguf.ValueTypeFloat32)
			if err != nil {
				return Spec{}, err
			}
		}
		if spec.ExpertWeightsNorm, err = required[bool](values, prefix+"expert_weights_norm", gguf.ValueTypeBool); err != nil {
			return Spec{}, err
		}
		if spec.CompressRatios, err = requiredArray[uint32](values, prefix+"attention.compress_ratios", gguf.ValueTypeUint32); err != nil {
			return Spec{}, err
		}
		if len(spec.CompressRatios) < int(spec.BlockCount) {
			return Spec{}, errors.New("DeepSeek 4 compression schedule is shorter than its block count")
		}
		spec.CompressRatios = spec.CompressRatios[:spec.BlockCount]
		if spec.LayerSwiGLUClamp, err = optionalLayerFloat32(values, prefix+"swiglu_clamp_exp", spec.BlockCount); err != nil {
			return Spec{}, err
		}
		if len(spec.LayerSwiGLUClamp) == 0 {
			return Spec{}, errors.New("DeepSeek 4 expert SwiGLU clamp is missing")
		}
		if spec.LayerSharedSwiGLUClamp, err = optionalLayerFloat32(values, prefix+"swiglu_clamp_shexp", spec.BlockCount); err != nil {
			return Spec{}, err
		}
		if len(spec.LayerSharedSwiGLUClamp) == 0 {
			spec.LayerSharedSwiGLUClamp = slices.Clone(spec.LayerSwiGLUClamp)
		}
		if spec.SharedExpertCount > math.MaxUint32/spec.ExpertFeedForward {
			return Spec{}, errors.New("DeepSeek 4 shared expert width overflows")
		}
		spec.SharedExpertFF = spec.SharedExpertCount * spec.ExpertFeedForward
	}
	if architecture == "gemma2" || architecture == "gemma3" || architecture == "gemma3n" || architecture == "gemma4" || architecture == "gemma4-assistant" ||
		architecture == "olmo2" || architecture == "cohere2" || architecture == "cohere2moe" {
		spec.RopeFrequencySWA = spec.RopeFrequencyBase
		if architecture == "gemma3" || architecture == "gemma3n" || architecture == "gemma4" || architecture == "gemma4-assistant" {
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
			if architecture == "gemma3n" {
				spec.SlidingPattern = 5
			}
			if architecture == "gemma4" || architecture == "gemma4-assistant" {
				if spec.SlidingLayers, err = requiredLayerBoolCompatible(
					values, prefix+"attention.sliding_window_pattern", spec.BlockCount,
				); err != nil {
					return Spec{}, err
				}
			}
			if value, ok := optional[uint32](
				values,
				prefix+"attention.sliding_window_pattern",
				gguf.ValueTypeUint32,
			); ok {
				spec.SlidingPattern = value
			}
			patternValue := values[prefix+"attention.sliding_window_pattern"]
			if (architecture == "cohere2" || architecture == "cohere2moe") &&
				patternValue.Type == gguf.ValueTypeArray {
				if layers, ok, layersErr := optionalArray[bool](values, prefix+"attention.sliding_window_pattern", gguf.ValueTypeBool); layersErr != nil {
					return Spec{}, layersErr
				} else if ok {
					if len(layers) != int(spec.BlockCount) {
						return Spec{}, fmt.Errorf("metadata %q has %d values, need %d", prefix+"attention.sliding_window_pattern", len(layers), spec.BlockCount)
					}
					spec.SlidingLayers = slices.Clone(layers)
				}
			}
			if architecture == "cohere2" && len(spec.SlidingLayers) == 0 {
				spec.NoRopeLayerStep = spec.SlidingPattern
			}
		}
	}
	if architecture == "llama4" {
		spec.RopeDimensionCount = spec.KeyLength
		if value, ok := optional[uint32](values, prefix+"rope.dimension_count", gguf.ValueTypeUint32); ok {
			spec.RopeDimensionCount = value
		}
		spec.RopeFrequencySWA = spec.RopeFrequencyBase
		if value, ok := optional[float32](values, prefix+"rope.freq_base_swa", gguf.ValueTypeFloat32); ok {
			spec.RopeFrequencySWA = value
		}
		window, hasWindow := optional[uint32](values, prefix+"attention.sliding_window", gguf.ValueTypeUint32)
		if hasWindow && window == 0 {
			spec.NoRopeLayerStep = 0
		} else {
			spec.SlidingWindow = 8192
			spec.SlidingPattern = 4
			if value, ok := optional[uint32](values, prefix+"attention.sliding_window_pattern", gguf.ValueTypeUint32); ok {
				spec.SlidingPattern = value
			}
			spec.NoRopeLayerStep = spec.SlidingPattern
			spec.AttentionTempFloor = 8192
			spec.AttentionTempScale = 0.1
			spec.AttentionTempOffset = 1
		}
	}
	if architecture == "gpt-oss" {
		spec.RopeDimensionCount = spec.KeyLength
		if spec.SlidingWindow, err = required[uint32](values, prefix+"attention.sliding_window", gguf.ValueTypeUint32); err != nil {
			return Spec{}, err
		}
		spec.SlidingPattern = 2
		if value, ok := optional[uint32](values, prefix+"attention.sliding_window_pattern", gguf.ValueTypeUint32); ok {
			spec.SlidingPattern = value
		}
		spec.RopeFrequencySWA = spec.RopeFrequencyBase
		if value, ok := optional[float32](values, prefix+"rope.freq_base_swa", gguf.ValueTypeFloat32); ok {
			spec.RopeFrequencySWA = value
		}
	}
	if architecture == "gemma4" {
		if spec.RopeDimensionCount, err = required[uint32](
			values, prefix+"rope.dimension_count", gguf.ValueTypeUint32,
		); err != nil {
			return Spec{}, err
		}
		if spec.RopeDimensionSWA, err = required[uint32](
			values, prefix+"rope.dimension_count_swa", gguf.ValueTypeUint32,
		); err != nil {
			return Spec{}, err
		}
		if spec.EmbeddingPerLayer, err = required[uint32](
			values, prefix+"embedding_length_per_layer_input", gguf.ValueTypeUint32,
		); err != nil {
			return Spec{}, err
		}
		spec.SharedKVLayers, _ = optional[uint32](
			values, prefix+"attention.shared_kv_layers", gguf.ValueTypeUint32,
		)
		spec.AttentionScale = 1
	}
	if architecture == "gemma4-assistant" {
		spec.RopeDimensionCount = spec.KeyLength
		spec.RopeDimensionSWA = spec.KeyLengthSWA
		spec.AttentionScale = 1
	}
	if spec.RopeScalingType == "longrope" &&
		(architecture == "llama" || architecture == "llama-embed" ||
			architecture == "minicpm" || architecture == "mistral3") {
		spec.RopeDimensionCount = spec.KeyLength
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
	if spec.RopeScalingType == "yarn" &&
		(architecture == "llama" || architecture == "llama-embed" || architecture == "minicpm") {
		rawAttentionFactor := float32(1)
		if value, ok := optional[float32](
			values, prefix+"rope.scaling.attn_factor", gguf.ValueTypeFloat32,
		); ok {
			rawAttentionFactor = value
		}
		spec.YaRNAttentionFactor = rawAttentionFactor
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
	if architecture == "bert" || architecture == "jina-bert-v2" || architecture == "jina-bert-v3" || architecture == "nomic-bert" || architecture == "nomic-bert-moe" {
		if spec.TokenTypeCount, err = required[uint32](values, "tokenizer.ggml.token_type_count", gguf.ValueTypeUint32); err != nil {
			return Spec{}, err
		}
	}
	if err := spec.validate(); err != nil {
		return Spec{}, err
	}
	return spec, nil
}

func (s Spec) IsRecurrentLayer(block uint32) bool {
	return s.Profile().Cadence.recurrent(s, block)
}

func (s Spec) IsInterleavedMoELayer(block uint32) bool {
	return s.Profile().Cadence.moe(s, block)
}

func (s Spec) IsSlidingLayer(block uint32) bool {
	return s.Profile().Cadence.sliding(s, block)
}

func layerValue[T any](values []T, layer uint32, fallback T) T {
	if layer < uint32(len(values)) {
		return values[layer]
	}
	return fallback
}

// LayerHeadCount: per-layer query heads; scalar fallback.
func (s Spec) LayerHeadCount(block uint32) uint32 {
	return layerValue(s.LayerHeadCounts, block, s.HeadCount)
}

// LayerKVHeadCount: per-layer KV heads; scalar fallback.
func (s Spec) LayerKVHeadCount(block uint32) uint32 {
	return layerValue(s.LayerKVHeadCounts, block, s.HeadCountKV)
}

func (s Spec) LayerFeedForwardLength(block uint32) uint32 {
	return layerValue(s.LayerFeedForward, block, s.FeedForwardLength)
}

func (s Spec) LayerKeyLength(block uint32) uint32 {
	if s.KeyLengthSWA > 0 && s.IsSlidingLayer(block) {
		return s.KeyLengthSWA
	}
	return s.KeyLength
}

func (s Spec) LayerValueLength(block uint32) uint32 {
	if s.ValueLengthSWA > 0 && s.IsSlidingLayer(block) {
		return s.ValueLengthSWA
	}
	return s.ValueLength
}

func (s Spec) LayerHasKV(block uint32) bool {
	return !s.Profile().Has(ArchitectureSharedKV) || block < s.BlockCount-s.SharedKVLayers
}

func (s Spec) LayerSharedKVSource(block uint32) uint32 {
	start := s.BlockCount - s.SharedKVLayers
	if s.IsSlidingLayer(block) {
		return start - 2
	}
	return start - 1
}

func (s Spec) LayerRopeDimensionCount(block uint32) uint32 {
	if s.KeyLengthSWA > 0 && s.RopeDimensionSWA > 0 && s.IsSlidingLayer(block) {
		return s.RopeDimensionSWA
	}
	if s.Profile().Rotary.FactorPairs && !s.IsSlidingLayer(block) {
		return s.RopeDimensionCount / 2
	}
	return s.RopeDimensionCount
}

func (s Spec) LayerExpertSwiGLUClamp(block uint32) float32 {
	return layerValue(s.LayerSwiGLUClamp, block, 0)
}

func (s Spec) LayerSharedSwiGLUClampLimit(block uint32) float32 {
	return layerValue(s.LayerSharedSwiGLUClamp, block, 0)
}

func (s Spec) UsesRoPE(block uint32) bool {
	if s.Architecture == "cohere2moe" || s.Architecture == "cohere2" && len(s.SlidingLayers) != 0 {
		return !s.RopeDisabled && block < s.BlockCount &&
			(s.Architecture == "cohere2moe" && block < s.LeadingDenseBlocks || s.IsSlidingLayer(block))
	}
	if s.Architecture == "smallthinker" {
		return !s.RopeDisabled && block < s.BlockCount &&
			(s.SlidingWindow == 0 || s.NoRopeLayerStep == 0 || block%s.NoRopeLayerStep != 0)
	}
	if s.Architecture == "exaone-moe" {
		return s.IsSlidingLayer(block)
	}
	blockCount := s.BlockCount
	if draft := s.Profile().DraftPlan(s.NextNPredictLayers); draft.AppendedBlocks {
		blockCount += draft.Heads
	}
	return !s.RopeDisabled &&
		(blockCount == 0 || block < blockCount) &&
		(s.NoRopeLayerStep == 0 || (block+1)%s.NoRopeLayerStep != 0)
}

func (s Spec) InputEmbeddingScale() float32 {
	if s.EmbeddingScale > 0 {
		return s.EmbeddingScale
	}
	if s.Architecture == "afmoe" || s.Profile().Has(ArchitectureGemma) {
		return float32(math.Sqrt(float64(s.EmbeddingLength)))
	}
	return 1
}

func (s Spec) OutputLogitMultiplier() float32 {
	if s.LogitScale > 0 {
		if s.Architecture == "cohere2" || s.Architecture == "cohere2moe" || s.Architecture == "command-r" || s.Architecture == "grok" || s.Architecture == "talkie" {
			return s.LogitScale
		}
		return 1 / s.LogitScale
	}
	return 1
}

func (s Spec) IsEncoderOnly() bool {
	return s.Profile().Has(ArchitectureEncoderOnly)
}

func (s Spec) UsesLayerNorm() bool {
	return s.NormPlan().Operation == NormalizationLayer
}

func (s Spec) RequiresLayerNormBias() bool {
	return s.NormPlan().Bias
}

func (s Spec) UsesUnweightedLayerNorm() bool {
	return s.NormPlan().Operation == NormalizationUnweightedLayer
}

func (s Spec) UsesUnweightedRMSNorm() bool {
	return s.NormPlan().Operation == NormalizationUnweightedRMS
}

func (s Spec) UsesWeightOnlyLayerNorm() bool {
	return s.NormPlan().Operation == NormalizationWeightOnlyLayer
}

// NormPlan: compiles normalization behavior from profile and metadata.
func (s Spec) NormPlan() NormalizationPlan {
	profile := s.Profile()
	operation := profile.Normalization
	if operation == NormalizationWeightOnlyLayer && s.Architecture == "cohere2moe" && s.LayerNormEpsilon <= 0 {
		operation = NormalizationRMS
	}
	post := profile.Has(ArchitecturePostNorm) || s.Architecture == "olmo2"
	pre := !profile.Has(ArchitecturePostOnlyNorm) && s.Architecture != "olmo2"
	return NormalizationPlan{
		Operation: operation, PreAttention: pre, PreFeedForward: pre,
		PostAttention: post, PostFeedForward: post,
		Bias: s.Architecture == "phimoe" ||
			(operation == NormalizationLayer && s.Architecture != "dbrx" && s.Architecture != "mpt"),
		PostNormLayout: profile.PostNormLayout, FeedForwardLayout: profile.FFNNormLayout,
	}
}

func (s Spec) validate() error {
	if err := s.validateBaseMetadata(); err != nil {
		return err
	}
	if err := s.validateRecurrentFamilies(); err != nil {
		return err
	}
	if err := s.validateEncoderFamilies(); err != nil {
		return err
	}
	if err := s.validateHybridMoEFamilies(); err != nil {
		return err
	}
	if err := s.validateMLAFamilies(); err != nil {
		return err
	}
	if err := s.validateAttentionFamilies(); err != nil {
		return err
	}
	if s.RopeScalingType == "linear" && s.RopeScalingFactor <= 0 {
		return errors.New("linear RoPE scaling factor must be positive")
	}
	return s.validateNumericPolicies()
}

func firstPositive(values []uint32) uint32 {
	for _, value := range values {
		if value > 0 {
			return value
		}
	}
	return 0
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
	return slices.Clone(items), nil
}

func optionalLayerFloat32(
	values map[string]gguf.Value,
	key string,
	count uint32,
) ([]float32, error) {
	if _, ok := values[key]; !ok {
		return nil, nil
	}
	return requiredLayerFloat32(values, key, count)
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
	return slices.Clone(items), nil
}

func requiredLayerUint32Compatible(
	values map[string]gguf.Value,
	key string,
	count uint32,
) ([]uint32, error) {
	value, ok := values[key]
	if !ok {
		return nil, fmt.Errorf("required metadata %q is missing", key)
	}
	if value.Type == gguf.ValueTypeUint32 {
		return requiredLayerUint32(values, key, count)
	}
	if value.Type == gguf.ValueTypeInt32 {
		scalar, valid := value.Data.(int32)
		if !valid || scalar < 0 {
			return nil, fmt.Errorf("metadata %q has an invalid scalar value", key)
		}
		result := make([]uint32, count)
		for index := range result {
			result[index] = uint32(scalar)
		}
		return result, nil
	}
	if value.Type != gguf.ValueTypeArray ||
		(value.ArrayType != gguf.ValueTypeUint32 && value.ArrayType != gguf.ValueTypeInt32) {
		return nil, fmt.Errorf("metadata %q must be a uint32 or integer array", key)
	}
	result := make([]uint32, count)
	switch value.ArrayType {
	case gguf.ValueTypeUint32:
		items, valid := value.Data.([]uint32)
		if !valid || len(items) != int(count) {
			return nil, fmt.Errorf("metadata %q has invalid layer values", key)
		}
		copy(result, items)
	case gguf.ValueTypeInt32:
		items, valid := value.Data.([]int32)
		if !valid || len(items) != int(count) {
			return nil, fmt.Errorf("metadata %q has invalid layer values", key)
		}
		for index, item := range items {
			if item < 0 {
				return nil, fmt.Errorf("metadata %q has a negative layer value", key)
			}
			result[index] = uint32(item)
		}
	}
	return result, nil
}

func requiredLayerBoolCompatible(
	values map[string]gguf.Value,
	key string,
	count uint32,
) ([]bool, error) {
	value, ok := values[key]
	if !ok {
		return nil, fmt.Errorf("required metadata %q is missing", key)
	}
	result := make([]bool, count)
	if value.Type == gguf.ValueTypeBool {
		scalar, valid := value.Data.(bool)
		if !valid {
			return nil, fmt.Errorf("metadata %q has an invalid Go representation", key)
		}
		for index := range result {
			result[index] = scalar
		}
		return result, nil
	}
	if value.Type == gguf.ValueTypeUint32 {
		scalar, valid := value.Data.(uint32)
		if !valid {
			return nil, fmt.Errorf("metadata %q has an invalid Go representation", key)
		}
		for index := range result {
			result[index] = scalar != 0
		}
		return result, nil
	}
	if value.Type == gguf.ValueTypeInt32 {
		scalar, valid := value.Data.(int32)
		if !valid || scalar < 0 {
			return nil, fmt.Errorf("metadata %q has an invalid scalar value", key)
		}
		for index := range result {
			result[index] = scalar != 0
		}
		return result, nil
	}
	if value.Type != gguf.ValueTypeArray {
		return nil, fmt.Errorf("metadata %q must be a bool, uint32, or compatible array", key)
	}
	switch value.ArrayType {
	case gguf.ValueTypeBool:
		items, valid := value.Data.([]bool)
		if !valid || len(items) != int(count) {
			return nil, fmt.Errorf("metadata %q has invalid layer values", key)
		}
		copy(result, items)
	case gguf.ValueTypeUint32:
		items, valid := value.Data.([]uint32)
		if !valid || len(items) != int(count) {
			return nil, fmt.Errorf("metadata %q has invalid layer values", key)
		}
		for index, item := range items {
			result[index] = item != 0
		}
	case gguf.ValueTypeInt32:
		items, valid := value.Data.([]int32)
		if !valid || len(items) != int(count) {
			return nil, fmt.Errorf("metadata %q has invalid layer values", key)
		}
		for index, item := range items {
			if item < 0 {
				return nil, fmt.Errorf("metadata %q has a negative layer value", key)
			}
			result[index] = item != 0
		}
	default:
		return nil, fmt.Errorf("metadata %q must use bool or integer layer values", key)
	}
	return result, nil
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
