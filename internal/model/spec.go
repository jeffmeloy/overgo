package model

import (
	"errors"
	"fmt"
	"math"
	"slices"

	"overgo/internal/gguf"
)

// UnsupportedArchitectureError: valid, unsupported GGUF architecture.
type UnsupportedArchitectureError struct {
	Architecture string
}

// LayerHasFullIndexer: DSA full-indexer predicate.
func (s Spec) LayerHasFullIndexer(layer uint32) bool {
	if s.Profile().Cadence.FullIndexerEveryLayer && layer < s.BlockCount+s.NextNPredictLayers {
		return true
	}
	return s.Profile().Attention == AttentionSparseLatent && layerValue(s.IndexerFullLayers, layer, false)
}

func (e *UnsupportedArchitectureError) Error() string {
	return fmt.Sprintf("model architecture %q is not supported", e.Architecture)
}

// ReadSpec: validates model metadata.
func ReadSpec(file *gguf.File) (Spec, error) {
	return readSpec(file, nil)
}

// ReadSpecWithProfile: validates metadata against a resolved policy.
func ReadSpecWithProfile(file *gguf.File, profile ArchitectureProfile) (Spec, error) {
	return readSpec(file, &profile)
}

// BindSpecProfile validates persisted metadata against an exact policy.
func BindSpecProfile(spec Spec, profile ArchitectureProfile) (Spec, error) {
	if err := ValidateArchitectureProfile(profile); err != nil {
		return Spec{}, err
	}
	if profile.Name == "" || spec.Architecture != profile.Name {
		return Spec{}, errors.New("model profile does not match persisted metadata")
	}
	spec = spec.withProfile(profile)
	if err := spec.validate(); err != nil {
		return Spec{}, err
	}
	return spec, nil
}

func readSpec(file *gguf.File, resolved *ArchitectureProfile) (Spec, error) {
	metadata, err := newSpecMetadataWithProfile(file, resolved)
	if err != nil {
		return Spec{}, err
	}
	architecture, profile := metadata.architecture, metadata.profile
	spec := Spec{CommonSpec: CommonSpec{Architecture: architecture}, AttentionSpec: AttentionSpec{NonCausalAttention: profile.Has(ArchitectureNonCausal),
		RopeDisabled: profile.Has(ArchitectureRoPEDisabled)}, profile: &profile,
	}
	state, err := metadata.readBase(&spec)
	if err != nil {
		return Spec{}, err
	}
	if err := metadata.readAttentionShape(&spec, state); err != nil {
		return Spec{}, err
	}
	profile.MetadataDefaults.readPosition(&spec)
	if !profile.Metadata.DraftBeforeShape {
		if err := metadata.readDraftLayers(&spec); err != nil {
			return Spec{}, err
		}
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
	values, prefix := m.values, m.prefix
	validation := m.profile.Validation
	declaredBlockCount := state.declaredBlockCount
	var err error
	if validation.Attention == AttentionValidationRefact {
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
	if validation.Attention == AttentionValidationCohere2MoE {
		if value, ok := optional[float32](values, prefix+"attention.layer_norm_rms_epsilon", gguf.ValueTypeFloat32); ok {
			spec.RMSNormEpsilon = value
			spec.profile.Normalization = NormalizationRMS
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
	if m.profile.readsMetadata(MetadataReadJaisScale) {
		spec.AttentionScale = 1 / float32(spec.KeyLength)
	}
	if value, ok := optional[float32](values, prefix+"attention.scale", gguf.ValueTypeFloat32); ok {
		spec.AttentionScale = value
	}
	m.profile.MetadataDefaults.read(values, prefix, &spec)
	if validation.Attention == AttentionValidationTalkie {
		if spec.LogitScale, err = required[float32](values, prefix+"logit_scale", gguf.ValueTypeFloat32); err != nil {
			return Spec{}, err
		}
	}
	if validation.Recurrent == RecurrentValidationDFlash {
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
	if validation.Hybrid == HybridValidationRopeScaling || validation.MLA == MLAValidationMiniCPM3 {
		spec.EmbeddingScale = 12
		spec.ResidualScale = float32(1.4 / math.Sqrt(float64(spec.BlockCount)))
		spec.LogitScale = 256 / float32(spec.EmbeddingLength)
		if validation.MLA == MLAValidationMiniCPM3 {
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
	if validation.hybridOneOf(
		HybridValidationGranite, HybridValidationGraniteMoE, HybridValidationGraniteHybrid,
	) {
		if validation.Hybrid == HybridValidationGraniteHybrid {
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
		if mapping, ok, mappingErr := optionalArray[int32](
			values,
			prefix+"deepstack_mapping",
			gguf.ValueTypeInt32,
		); mappingErr != nil {
			return Spec{}, mappingErr
		} else if ok && len(mapping) > 0 {
			if validation.Hybrid != HybridValidationGranite {
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
	if validation.attentionOneOf(AttentionValidationCohere2, AttentionValidationCohere2MoE) {
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
	if validation.Attention == AttentionValidationStableLM {
		if spec.RopeDimensionCount, err = required[uint32](
			values,
			prefix+"rope.dimension_count",
			gguf.ValueTypeUint32,
		); err != nil {
			return Spec{}, err
		}
	}
	if m.profile.readsMetadata(MetadataReadGPTJRotary) ||
		validation.Attention == AttentionValidationPhi2 {
		if spec.RopeDimensionCount, err = required[uint32](
			values,
			prefix+"rope.dimension_count",
			gguf.ValueTypeUint32,
		); err != nil {
			return Spec{}, err
		}
	}
	if validation.Attention == AttentionValidationPhi3 {
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
		spec.RopeAttentionFactor = optionalOr(
			values, prefix+"rope.scaling.attn_factor", gguf.ValueTypeFloat32, float32(1),
		)
	}
	if validation.Attention == AttentionValidationPanguEmbedded {
		spec.RopeDimensionCount = optionalOr(values, prefix+"rope.dimension_count", gguf.ValueTypeUint32, spec.KeyLength)
		spec.OriginalContextLength = optionalOr(
			values, prefix+"rope.scaling.original_context_length", gguf.ValueTypeUint32, spec.ContextLength,
		)
		spec.RopeAttentionFactor = optionalOr(
			values, prefix+"rope.scaling.attn_factor", gguf.ValueTypeFloat32, float32(1),
		)
	}
	if validation.Attention == AttentionValidationModernBERT {
		spec.RopeDimensionCount = optionalOr(values, prefix+"rope.dimension_count", gguf.ValueTypeUint32, spec.KeyLength)
		spec.RopeFrequencySWA = optionalOr(values, prefix+"rope.freq_base_swa", gguf.ValueTypeFloat32, float32(10000))
		if value, ok := optional[uint32](values, prefix+"attention.sliding_window", gguf.ValueTypeUint32); ok {
			spec.SlidingWindow = value
		}
		if spec.SlidingWindow > 0 {
			spec.SlidingPattern = optionalOr(
				values, prefix+"attention.sliding_window_pattern", gguf.ValueTypeUint32, uint32(3),
			)
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
	if validation.Attention == AttentionValidationGemmaEmbedding {
		spec.RopeDimensionCount = optionalOr(values, prefix+"rope.dimension_count", gguf.ValueTypeUint32, spec.KeyLength)
		spec.RopeFrequencySWA = optionalOr(values, prefix+"rope.freq_base_swa", gguf.ValueTypeFloat32, float32(10000))
		if spec.SlidingWindow, err = required[uint32](values, prefix+"attention.sliding_window", gguf.ValueTypeUint32); err != nil {
			return Spec{}, err
		}
		spec.SlidingPattern = optionalOr(
			values, prefix+"attention.sliding_window_pattern", gguf.ValueTypeUint32, uint32(6),
		)
		spec.Dense2FeatureIn, _ = optional[uint32](values, prefix+"dense_2_feat_in", gguf.ValueTypeUint32)
		spec.Dense2FeatureOut, _ = optional[uint32](values, prefix+"dense_2_feat_out", gguf.ValueTypeUint32)
		spec.Dense3FeatureIn, _ = optional[uint32](values, prefix+"dense_3_feat_in", gguf.ValueTypeUint32)
		spec.Dense3FeatureOut, _ = optional[uint32](values, prefix+"dense_3_feat_out", gguf.ValueTypeUint32)
	}
	if validation.Attention == AttentionValidationDeci {
		spec.RopeDimensionCount = optionalOr(values, prefix+"rope.dimension_count", gguf.ValueTypeUint32, spec.KeyLength)
		spec.OriginalContextLength = optionalOr(
			values, prefix+"rope.scaling.original_context_length", gguf.ValueTypeUint32, spec.ContextLength,
		)
		spec.RopeAttentionFactor = optionalOr(
			values, prefix+"rope.scaling.attn_factor", gguf.ValueTypeFloat32, float32(1),
		)
	}
	if validation.Attention == AttentionValidationApertus {
		spec.RopeDimensionCount = optionalOr(values, prefix+"rope.dimension_count", gguf.ValueTypeUint32, spec.KeyLength)
		spec.OriginalContextLength = optionalOr(
			values, prefix+"rope.scaling.original_context_length", gguf.ValueTypeUint32, spec.ContextLength,
		)
		spec.RopeAttentionFactor = optionalOr(
			values, prefix+"rope.scaling.attn_factor", gguf.ValueTypeFloat32, float32(1),
		)
		for _, field := range []metadataField[[]float32]{
			metadataDestination("xielu.alpha_n", &spec.XIELUAlphaN),
			metadataDestination("xielu.alpha_p", &spec.XIELUAlphaP),
			metadataDestination("xielu.beta", &spec.XIELUBeta),
			metadataDestination("xielu.eps", &spec.XIELUEpsilon),
		} {
			*field.destination, err = requiredLayerFloat32(
				values, prefix+field.key, spec.BlockCount,
			)
			if err != nil {
				return Spec{}, err
			}
		}
	}
	if validation.Attention == AttentionValidationGPTNeoX {
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
	if validation.attentionOneOf(AttentionValidationGLM4, AttentionValidationGLM4MoE) {
		spec.RopeDimensionCount = optionalOr(values, prefix+"rope.dimension_count", gguf.ValueTypeUint32, spec.KeyLength)
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
	if validation.Hybrid == HybridValidationMiMo2 {
		spec.RopeDimensionCount = optionalOr(values, prefix+"rope.dimension_count", gguf.ValueTypeUint32, spec.KeyLength)
		if spec.SlidingWindow, err = required[uint32](values, prefix+"attention.sliding_window", gguf.ValueTypeUint32); err != nil {
			return Spec{}, err
		}
		spec.RopeFrequencySWA = optionalOr(
			values, prefix+"rope.freq_base_swa", gguf.ValueTypeFloat32, spec.RopeFrequencyBase,
		)
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
	if validation.Hybrid == HybridValidationStep35 {
		spec.RopeDimensionCount = optionalOr(values, prefix+"rope.dimension_count", gguf.ValueTypeUint32, spec.KeyLength)
		if spec.SlidingWindow, err = required[uint32](values, prefix+"attention.sliding_window", gguf.ValueTypeUint32); err != nil {
			return Spec{}, err
		}
		spec.RopeFrequencySWA = optionalOr(
			values, prefix+"rope.freq_base_swa", gguf.ValueTypeFloat32, spec.RopeFrequencyBase,
		)
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
	if validation.Attention == AttentionValidationEXAOne4 {
		spec.RopeDimensionCount = optionalOr(values, prefix+"rope.dimension_count", gguf.ValueTypeUint32, spec.KeyLength)
		spec.RopeFrequencySWA = optionalOr(
			values, prefix+"rope.freq_base_swa", gguf.ValueTypeFloat32, spec.RopeFrequencyBase,
		)
		if value, ok := optional[uint32](
			values, prefix+"attention.sliding_window", gguf.ValueTypeUint32,
		); ok {
			spec.SlidingWindow = value
		}
		if spec.SlidingWindow > 0 {
			if value, ok := optional[uint32](values, prefix+"attention.sliding_window_pattern", gguf.ValueTypeUint32); ok {
				spec.SlidingPattern = value
			}
			spec.NoRopeLayerStep = spec.SlidingPattern
		}
	}
	if validation.Hybrid == HybridValidationEXAOneMoE {
		spec.RopeDimensionCount = optionalOr(values, prefix+"rope.dimension_count", gguf.ValueTypeUint32, spec.KeyLength)
		spec.RopeFrequencySWA = optionalOr(
			values, prefix+"rope.freq_base_swa", gguf.ValueTypeFloat32, spec.RopeFrequencyBase,
		)
		if spec.SlidingWindow, err = required[uint32](values, prefix+"attention.sliding_window", gguf.ValueTypeUint32); err != nil {
			return Spec{}, err
		}
		spec.SlidingPattern = m.profile.MetadataDefaults.SlidingPattern
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
	if validation.Hybrid == HybridValidationBailingMoE2 {
		spec.RopeDimensionCount = optionalOr(values, prefix+"rope.dimension_count", gguf.ValueTypeUint32, spec.KeyLength)
	}
	if validation.Attention == AttentionValidationFalcon {
		spec.RopeDimensionCount, _ = optional[uint32](
			values,
			prefix+"rope.dimension_count",
			gguf.ValueTypeUint32,
		)
	}
	if m.profile.readsMetadata(MetadataReadCommandRLogits) {
		spec.LogitScale, _ = optional[float32](
			values,
			prefix+"logit_scale",
			gguf.ValueTypeFloat32,
		)
	}
	if m.profile.readsMetadata(MetadataReadBaichuanBlocks) &&
		spec.BlockCount != 32 && spec.BlockCount != 40 {
		return Spec{}, errors.New("Baichuan block count must select the 32-layer RoPE or 40-layer ALiBi variant")
	}
	if validation.MLA == MLAValidationMistral3 {
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
		if spec.RopeScalingType == ropeScalingYaRN {
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
	if validation.Attention == AttentionValidationQwen {
		if spec.FeedForwardLength == 0 || spec.FeedForwardLength%2 != 0 {
			return Spec{}, errors.New("Qwen feed-forward length must be positive and even")
		}
		spec.FeedForwardLength /= 2
		spec.RopeDimensionCount = spec.KeyLength
		if value, ok := optional[uint32](values, prefix+"rope.dimension_count", gguf.ValueTypeUint32); ok {
			spec.RopeDimensionCount = value
		}
	}
	if validation.Attention == AttentionValidationChatGLM {
		spec.RopeDimensionCount = spec.KeyLength
		if value, ok := optional[uint32](values, prefix+"rope.dimension_count", gguf.ValueTypeUint32); ok {
			spec.RopeDimensionCount = value
		}
	}
	if validation.Attention == AttentionValidationHunyuan {
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
	if m.profile.readsMetadata(MetadataReadVisualSections) {
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
	if m.profile.readsMetadata(MetadataReadQwen3VLDeepstack) {
		if value, ok := optional[uint32](values, prefix+"n_deepstack_layers", gguf.ValueTypeUint32); ok {
			spec.DeepstackLayerCount = value
		}
	}
	if m.profile.readsMetadata(MetadataReadSmolLM3NoRoPE) {
		spec.NoRopeLayerStep = 4
	}
	if validation.Hybrid == HybridValidationSmallThinker {
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
	if validation.Hybrid == HybridValidationMellum {
		spec.RopeDimensionCount = optionalOr(values, prefix+"rope.dimension_count", gguf.ValueTypeUint32, spec.KeyLength)
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
			spec.RopeFrequencySWA = optionalOr(
				values, prefix+"rope.freq_base_swa", gguf.ValueTypeFloat32, spec.RopeFrequencyBase,
			)
		}
	}
	if validation.Attention == AttentionValidationPLaMo3 {
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
			spec.RopeFrequencySWA = optionalOr(
				values, prefix+"rope.freq_base_swa", gguf.ValueTypeFloat32, spec.RopeFrequencyBase,
			)
		}
	}
	if validation.Hybrid == HybridValidationAFMoE {
		spec.NoRopeLayerStep = 4
	}
	if m.profile.readsMetadata(MetadataReadOLMoClamp) {
		if clamp, ok := optional[float32](
			values,
			prefix+"attention.clamp_kqv",
			gguf.ValueTypeFloat32,
		); ok {
			spec.AttentionClamp = clamp
		}
	}
	if m.profile.Forward.Session == ForwardSessionEncoderDecoder ||
		m.profile.Forward.Operation == ForwardOperationEncoder {
		if spec.RelativeBuckets, err = required[uint32](
			values,
			prefix+"attention.relative_buckets_count",
			gguf.ValueTypeUint32,
		); err != nil {
			return Spec{}, err
		}
		if m.profile.Forward.Session == ForwardSessionEncoderDecoder {
			spec.DecoderBlockCount = optionalOr(
				values, prefix+"decoder_block_count", gguf.ValueTypeUint32, spec.BlockCount,
			)
			spec.DecoderStartTokenID, _ = optional[uint32](values, prefix+"decoder_start_token_id", gguf.ValueTypeUint32)
		}
	}
	if validation.hybridOneOf(
		HybridValidationQwen3Next, HybridValidationQwen35, HybridValidationQwen35MoE,
	) {
		if validation.Hybrid == HybridValidationQwen3Next {
			spec.RopeDimensionCount = optionalOr(
				values, prefix+"rope.dimension_count", gguf.ValueTypeUint32, spec.KeyLength,
			)
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
		if err = readRequiredMetadataFields(
			values, prefix, gguf.ValueTypeUint32,
			metadataDestination("ssm.conv_kernel", &spec.SSMConvKernel),
			metadataDestination("ssm.inner_size", &spec.SSMInnerSize),
			metadataDestination("ssm.state_size", &spec.SSMStateSize),
			metadataDestination("ssm.time_step_rank", &spec.SSMTimeStepRank),
			metadataDestination("ssm.group_count", &spec.SSMGroupCount),
		); err != nil {
			return Spec{}, err
		}
		spec.FullAttentionInterval = optionalOr(
			values, prefix+"full_attention_interval", gguf.ValueTypeUint32, uint32(4),
		)
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
	if validation.MLA == MLAValidationKimiLinear {
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
	if validation.recurrentOneOf(RecurrentValidationRWKV6, RecurrentValidationRWKV6Qwen2) {
		if err = readRequiredMetadataFields(
			values, prefix, gguf.ValueTypeUint32,
			metadataDestination("wkv.head_size", &spec.WKVHeadSize),
			metadataDestination("time_mix_extra_dim", &spec.TimeMixExtraDim),
			metadataDestination("time_decay_extra_dim", &spec.TimeDecayExtraDim),
		); err != nil {
			return Spec{}, err
		}
		spec.RescaleEvery, _ = optional[uint32](values, prefix+"rescale_every_n_layers", gguf.ValueTypeUint32)
		spec.TokenShiftCount = 1
		if validation.Recurrent == RecurrentValidationRWKV6 {
			spec.TokenShiftCount = 2
		}
		if value, ok := optional[uint32](values, prefix+"token_shift_count", gguf.ValueTypeUint32); ok {
			spec.TokenShiftCount = value
		}
	}
	if validation.recurrentOneOf(RecurrentValidationRWKV7, RecurrentValidationARWKV7) {
		if err = readRequiredMetadataFields(
			values, prefix, gguf.ValueTypeUint32,
			metadataDestination("wkv.head_size", &spec.WKVHeadSize),
			metadataDestination("attention.decay_lora_rank", &spec.DecayLoRARank),
			metadataDestination("attention.iclr_lora_rank", &spec.ICLRLoRARank),
			metadataDestination("attention.value_residual_mix_lora_rank", &spec.ValueMixLoRARank),
		); err != nil {
			return Spec{}, err
		}
		spec.GateLoRARank, _ = optional[uint32](values, prefix+"attention.gate_lora_rank", gguf.ValueTypeUint32)
		spec.TokenShiftCount = 1
		if validation.Recurrent == RecurrentValidationRWKV7 {
			spec.TokenShiftCount = 2
		}
		if value, ok := optional[uint32](values, prefix+"token_shift_count", gguf.ValueTypeUint32); ok {
			spec.TokenShiftCount = value
		}
	}
	if validation.recurrentOneOf(
		RecurrentValidationMamba, RecurrentValidationMamba2, RecurrentValidationJamba,
		RecurrentValidationGraniteHybrid, RecurrentValidationPLaMo2,
		RecurrentValidationNemotronH, RecurrentValidationNemotronHMoE,
		RecurrentValidationFalconH1,
	) {
		if err = readRequiredMetadataFields(
			values, prefix, gguf.ValueTypeUint32,
			metadataDestination("ssm.conv_kernel", &spec.SSMConvKernel),
			metadataDestination("ssm.inner_size", &spec.SSMInnerSize),
			metadataDestination("ssm.state_size", &spec.SSMStateSize),
			metadataDestination("ssm.time_step_rank", &spec.SSMTimeStepRank),
		); err != nil {
			return Spec{}, err
		}
		if validation.recurrentOneOf(RecurrentValidationMamba, RecurrentValidationJamba) {
			spec.SSMGroupCount = 1
			spec.SSMDtBCNorm, _ = optional[bool](values, prefix+"ssm.dt_b_c_rms", gguf.ValueTypeBool)
		} else if spec.SSMGroupCount, err = required[uint32](values, prefix+"ssm.group_count", gguf.ValueTypeUint32); err != nil {
			return Spec{}, err
		}
	}
	if validation.Recurrent == RecurrentValidationFalconH1 {
		spec.RopeDimensionCount = spec.KeyLength
		if value, ok := optional[uint32](values, prefix+"rope.dimension_count", gguf.ValueTypeUint32); ok {
			spec.RopeDimensionCount = value
		}
	}
	if validation.Recurrent == RecurrentValidationPLaMo2 {
		spec.AttentionScale = float32(1 / math.Sqrt(float64(spec.ValueLength)))
	}
	if validation.Recurrent == RecurrentValidationGraniteHybrid {
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
	if validation.Recurrent == RecurrentValidationNemotronHMoE {
		if err = readRequiredMetadataFields(
			values, prefix, gguf.ValueTypeUint32,
			metadataDestination("expert_count", &spec.ExpertCount),
			metadataDestination("expert_used_count", &spec.ExpertUsedCount),
			metadataDestination("expert_feed_forward_length", &spec.ExpertFeedForward),
			metadataDestination("expert_shared_feed_forward_length", &spec.SharedExpertFF),
		); err != nil {
			return Spec{}, err
		}
		spec.SharedExpertCount, _ = optional[uint32](values, prefix+"expert_shared_count", gguf.ValueTypeUint32)
		spec.ExpertWeightsNorm, _ = optional[bool](values, prefix+"expert_weights_norm", gguf.ValueTypeBool)
		spec.ExpertWeightsScale = 1
		if value, ok := optional[float32](values, prefix+"expert_weights_scale", gguf.ValueTypeFloat32); ok {
			spec.ExpertWeightsScale = value
		}
		spec.MoELatentSize, _ = optional[uint32](values, prefix+"moe_latent_size", gguf.ValueTypeUint32)
		spec.ExpertGatingFunc = expertGatingSigmoid
	}
	return spec, nil
}

func (m specMetadata) readExpertMetadata(spec Spec, state specReadState) (Spec, error) {
	values, prefix, profile := m.values, m.prefix, m.profile
	validation := profile.Validation
	var err error
	if validation.ExpertMetadata == ExpertMetadataRequired || state.declaredExperts {
		if err = readRequiredMetadataFields(
			values, prefix, gguf.ValueTypeUint32,
			metadataDestination("expert_count", &spec.ExpertCount),
			metadataDestination("expert_used_count", &spec.ExpertUsedCount),
		); err != nil {
			return Spec{}, err
		}
		if spec.ExpertUsedCount == 0 {
			return Spec{}, errors.New("expert used count is zero")
		}
		spec.ExpertFeedForward = optionalOr(
			values, prefix+"expert_feed_forward_length", gguf.ValueTypeUint32,
			spec.FeedForwardLength/spec.ExpertUsedCount,
		)
		if validation.Hybrid == HybridValidationArctic {
			spec.ExpertFeedForward = spec.FeedForwardLength
		}
		if validation.Encoder == EncoderValidationNomicBERTMoE {
			spec.ExpertFeedForward = spec.FeedForwardLength
		}
		if validation.Recurrent == RecurrentValidationJamba {
			spec.ExpertFeedForward = spec.FeedForwardLength
			spec.ExpertWeightsNorm = false
		}
		if validation.MLA == MLAValidationKimiLinear {
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
		if validation.Hybrid == HybridValidationLlama4 {
			if err = readRequiredMetadataFields(
				values, prefix, gguf.ValueTypeUint32,
				metadataDestination("expert_feed_forward_length", &spec.ExpertFeedForward),
				metadataDestination("interleave_moe_layer_step", &spec.MoELayerStep),
			); err != nil {
				return Spec{}, err
			}
			spec.SharedExpertFF = spec.ExpertFeedForward
			spec.ExpertGatingFunc = expertGatingSigmoid
		}
		spec.ExpertWeightsScale = optionalOr(
			values, prefix+"expert_weights_scale", gguf.ValueTypeFloat32, float32(1),
		)
		if validation.Hybrid == HybridValidationGPTOSS {
			spec.ExpertGatingFunc = expertGatingSelectedSoftmax
			spec.ExpertWeightsNorm = false
		}
		if validation.hybridOneOf(HybridValidationQwen3Next, HybridValidationQwen35MoE) {
			spec.SharedExpertFF = optionalOr(
				values, prefix+"expert_shared_feed_forward_length", gguf.ValueTypeUint32, spec.FeedForwardLength,
			)
		}
	}
	if profile.Has(ArchitectureLatentKVLayout) {
		if validation.MLA == MLAValidationDeepSeek32 {
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
		spec.ExpertWeightsScale = optionalOr(
			values, prefix+"expert_weights_scale", gguf.ValueTypeFloat32, float32(1),
		)
		spec.ExpertWeightsNorm, _ = optional[bool](values, prefix+"expert_weights_norm", gguf.ValueTypeBool)
		spec.ExpertGatingFunc = expertGatingSoftmax
		switch profile.Experts.Routing {
		case expertRouteSigmoid:
			spec.ExpertGatingFunc = expertGatingSigmoid
		case expertRouteSelectedSoftmax:
			spec.ExpertGatingFunc = expertGatingSelectedSoftmax
		}
		if profile.readsMetadata(MetadataReadGLMDSAGating) {
			spec.ExpertGatingFunc = expertGatingSigmoid
		}
		if validation.MLA == MLAValidationDeepSeek32 {
			if spec.ExpertGatingFunc, err = required[uint32](values, prefix+"expert_gating_func", gguf.ValueTypeUint32); err != nil {
				return Spec{}, err
			}
		} else if value, ok := optional[uint32](values, prefix+"expert_gating_func", gguf.ValueTypeUint32); ok && value != 0 {
			spec.ExpertGatingFunc = value
		}
	}
	if validation.Hybrid == HybridValidationMiMo2 {
		spec.ExpertGatingFunc = expertGatingSigmoid
		spec.ExpertWeightsNorm = true
	}
	if validation.Attention == AttentionValidationGemma4 {
		if count, ok := optional[uint32](values, prefix+"expert_count", gguf.ValueTypeUint32); ok && count > 0 {
			spec.ExpertCount = count
			if err = readRequiredMetadataFields(
				values, prefix, gguf.ValueTypeUint32,
				metadataDestination("expert_used_count", &spec.ExpertUsedCount),
				metadataDestination("expert_feed_forward_length", &spec.ExpertFeedForward),
			); err != nil {
				return Spec{}, err
			}
			spec.ExpertWeightsScale = 1
			spec.ExpertWeightsNorm = true
		}
	}
	if validation.Hybrid == HybridValidationStep35 {
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
		spec.MoELayerStep = optionalOr(
			values, prefix+"moe_every_n_layers", gguf.ValueTypeUint32, uint32(1),
		)
		spec.ExpertGatingFunc = expertGatingSigmoid
		if value, ok := optional[uint32](values, prefix+"expert_gating_func", gguf.ValueTypeUint32); ok && value != 0 {
			spec.ExpertGatingFunc = value
		}
		spec.ExpertWeightsNorm, _ = optional[bool](values, prefix+"expert_weights_norm", gguf.ValueTypeBool)
	}
	if validation.Attention == AttentionValidationGLM4MoE {
		spec.LeadingDenseBlocks, _ = optional[uint32](values, prefix+"leading_dense_block_count", gguf.ValueTypeUint32)
		if spec.SharedExpertCount, err = required[uint32](values, prefix+"expert_shared_count", gguf.ValueTypeUint32); err != nil {
			return Spec{}, err
		}
		if spec.ExpertFeedForward > 0 && spec.SharedExpertCount > math.MaxUint32/spec.ExpertFeedForward {
			return Spec{}, errors.New("GLM4-MoE shared expert width overflows")
		}
		spec.SharedExpertFF = spec.ExpertFeedForward * spec.SharedExpertCount
		spec.ExpertGatingFunc = expertGatingSigmoid
		if value, ok := optional[uint32](values, prefix+"expert_gating_func", gguf.ValueTypeUint32); ok {
			spec.ExpertGatingFunc = value
		}
		if value, ok := optional[bool](values, prefix+"expert_weights_norm", gguf.ValueTypeBool); ok {
			spec.ExpertWeightsNorm = value
		}
	}
	if validation.Hybrid == HybridValidationGroveMoE {
		spec.ExpertChunkFeedForward = optionalOr(
			values, prefix+"expert_chunk_feed_forward_length", gguf.ValueTypeUint32, spec.KeyLength,
		)
		if spec.ExpertGroupScale, err = required[float32](values, prefix+"expert_group_scale", gguf.ValueTypeFloat32); err != nil {
			return Spec{}, err
		}
		if spec.ExpertsPerGroup, err = required[uint32](values, prefix+"experts_per_group", gguf.ValueTypeUint32); err != nil {
			return Spec{}, err
		}
	}
	if validation.Attention == AttentionValidationCohere2MoE {
		if spec.LeadingDenseBlocks, err = required[uint32](values, prefix+"leading_dense_block_count", gguf.ValueTypeUint32); err != nil {
			return Spec{}, err
		}
		if spec.ExpertFeedForward, err = required[uint32](values, prefix+"expert_feed_forward_length", gguf.ValueTypeUint32); err != nil {
			return Spec{}, err
		}
		spec.ExpertGatingFunc = expertGatingSigmoid
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
			spec.SharedExpertFF = optionalOr(
				values, prefix+"expert_shared_feed_forward_length", gguf.ValueTypeUint32,
				spec.ExpertFeedForward*spec.SharedExpertCount,
			)
		}
	}
	if validation.Hybrid == HybridValidationHYV3 {
		if spec.ExpertFeedForward, err = required[uint32](values, prefix+"expert_feed_forward_length", gguf.ValueTypeUint32); err != nil {
			return Spec{}, err
		}
		spec.SharedExpertFF = optionalOr(
			values, prefix+"expert_shared_feed_forward_length", gguf.ValueTypeUint32, spec.ExpertFeedForward,
		)
		spec.ExpertGatingFunc = expertGatingSigmoid
		if value, ok := optional[uint32](values, prefix+"expert_gating_func", gguf.ValueTypeUint32); ok {
			spec.ExpertGatingFunc = value
		}
		if value, ok := optional[bool](values, prefix+"expert_weights_norm", gguf.ValueTypeBool); ok {
			spec.ExpertWeightsNorm = value
		}
		spec.RopeDimensionCount = spec.KeyLength
	}
	if validation.Hybrid == HybridValidationDeepSeek2OCR {
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
		spec.ExpertGatingFunc = expertGatingSoftmax
		if value, ok := optional[uint32](values, prefix+"expert_gating_func", gguf.ValueTypeUint32); ok {
			spec.ExpertGatingFunc = value
		}
		if value, ok := optional[bool](values, prefix+"expert_weights_norm", gguf.ValueTypeBool); ok {
			spec.ExpertWeightsNorm = value
		}
		spec.RopeDimensionCount = spec.KeyLength
	}
	if validation.Encoder == EncoderValidationNeoBERT {
		spec.RopeDimensionCount = spec.KeyLength
	}
	if validation.Encoder == EncoderValidationJinaV3 {
		spec.RopeDimensionCount = optionalOr(values, prefix+"rope.dimension_count", gguf.ValueTypeUint32, spec.KeyLength)
		spec.ExpertCount, _ = optional[uint32](values, prefix+"expert_count", gguf.ValueTypeUint32)
		spec.ExpertUsedCount, _ = optional[uint32](values, prefix+"expert_used_count", gguf.ValueTypeUint32)
		spec.MoELayerStep, _ = optional[uint32](values, prefix+"moe_every_n_layers", gguf.ValueTypeUint32)
		if spec.ExpertCount > 0 {
			spec.ExpertFeedForward = spec.FeedForwardLength
			spec.ExpertWeightsScale = optionalOr(
				values, prefix+"expert_weights_scale", gguf.ValueTypeFloat32, float32(1),
			)
		}
	}
	if validation.Encoder == EncoderValidationNomicBERT {
		spec.RopeDimensionCount = optionalOr(values, prefix+"rope.dimension_count", gguf.ValueTypeUint32, spec.KeyLength)
		if cadence, ok := optional[uint32](values, prefix+"moe_every_n_layers", gguf.ValueTypeUint32); ok && cadence > 0 {
			return Spec{}, errors.New("NomicBERT MoE cadence requires nomic-bert-moe architecture")
		}
	}
	if validation.Encoder == EncoderValidationNomicBERTMoE {
		spec.RopeDimensionCount = optionalOr(values, prefix+"rope.dimension_count", gguf.ValueTypeUint32, spec.KeyLength)
		if spec.MoELayerStep, err = required[uint32](values, prefix+"moe_every_n_layers", gguf.ValueTypeUint32); err != nil {
			return Spec{}, err
		}
	}
	if validation.Attention == AttentionValidationErnie45MoE {
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
	if validation.Hybrid == HybridValidationMellum {
		if spec.ExpertFeedForward, err = required[uint32](values, prefix+"expert_feed_forward_length", gguf.ValueTypeUint32); err != nil {
			return Spec{}, err
		}
		spec.ExpertWeightsNorm = true
	}
	if validation.Hybrid == HybridValidationHunyuanMoE {
		if spec.ExpertFeedForward, err = required[uint32](values, prefix+"expert_feed_forward_length", gguf.ValueTypeUint32); err != nil {
			return Spec{}, err
		}
		spec.SharedExpertFF = optionalOr(
			values, prefix+"expert_shared_feed_forward_length", gguf.ValueTypeUint32, spec.FeedForwardLength,
		)
		spec.ExpertWeightsNorm = true
		spec.RopeDimensionCount = spec.KeyLength
	}
	if validation.Hybrid == HybridValidationGrok {
		if _, ok := optional[uint32](
			values, prefix+"expert_feed_forward_length", gguf.ValueTypeUint32,
		); !ok {
			spec.ExpertFeedForward = spec.FeedForwardLength
		}
		spec.ExpertWeightsNorm = true
	}
	if validation.Hybrid == HybridValidationDBRX {
		spec.ExpertFeedForward = spec.FeedForwardLength
		spec.ExpertWeightsNorm = true
	}
	if validation.Hybrid == HybridValidationGraniteMoE ||
		validation.Hybrid == HybridValidationGranite && spec.ExpertCount > 0 {
		spec.ExpertFeedForward = spec.FeedForwardLength
		spec.ExpertWeightsNorm = true
		spec.SharedExpertFF, _ = optional[uint32](
			values, prefix+"expert_shared_feed_forward_length", gguf.ValueTypeUint32,
		)
	}
	if validation.Hybrid == HybridValidationSmallThinker {
		spec.ExpertFeedForward = spec.FeedForwardLength
		spec.ExpertWeightsNorm = true
		if spec.ExpertGatingFunc, err = required[uint32](
			values, prefix+"expert_gating_func", gguf.ValueTypeUint32,
		); err != nil {
			return Spec{}, err
		}
	}
	if validation.Hybrid == HybridValidationDOTS1 {
		if err = readRequiredMetadataFields(
			values, prefix, gguf.ValueTypeUint32,
			metadataDestination("expert_feed_forward_length", &spec.ExpertFeedForward),
			metadataDestination("expert_shared_count", &spec.SharedExpertCount),
			metadataDestination("expert_gating_func", &spec.ExpertGatingFunc),
		); err != nil {
			return Spec{}, err
		}
		spec.SharedExpertFF = spec.ExpertFeedForward * spec.SharedExpertCount
		spec.ExpertWeightsNorm, _ = optional[bool](
			values, prefix+"expert_weights_norm", gguf.ValueTypeBool,
		)
		spec.LeadingDenseBlocks, _ = optional[uint32](
			values, prefix+"leading_dense_block_count", gguf.ValueTypeUint32,
		)
	}
	if validation.Hybrid == HybridValidationMiniMaxM2 {
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
		if err = readRequiredMetadataFields(
			values, prefix, gguf.ValueTypeUint32,
			metadataDestination("expert_gating_func", &spec.ExpertGatingFunc),
			metadataDestination("rope.dimension_count", &spec.RopeDimensionCount),
		); err != nil {
			return Spec{}, err
		}
	}
	if validation.Hybrid == HybridValidationBailingMoE {
		if err = readRequiredMetadataFields(
			values, prefix, gguf.ValueTypeUint32,
			metadataDestination("expert_feed_forward_length", &spec.ExpertFeedForward),
			metadataDestination("expert_shared_count", &spec.SharedExpertCount),
		); err != nil {
			return Spec{}, err
		}
		spec.SharedExpertFF = spec.ExpertFeedForward * spec.SharedExpertCount
		spec.ExpertWeightsNorm, _ = optional[bool](values, prefix+"expert_weights_norm", gguf.ValueTypeBool)
		spec.LeadingDenseBlocks, _ = optional[uint32](values, prefix+"leading_dense_block_count", gguf.ValueTypeUint32)
	}
	if validation.Hybrid == HybridValidationDeepSeek {
		if err = readRequiredMetadataFields(
			values, prefix, gguf.ValueTypeUint32,
			metadataDestination("expert_feed_forward_length", &spec.ExpertFeedForward),
			metadataDestination("expert_shared_count", &spec.SharedExpertCount),
		); err != nil {
			return Spec{}, err
		}
		spec.SharedExpertFF = spec.ExpertFeedForward * spec.SharedExpertCount
		spec.LeadingDenseBlocks, _ = optional[uint32](values, prefix+"leading_dense_block_count", gguf.ValueTypeUint32)
	}
	if validation.Hybrid == HybridValidationLFM2MoE {
		if err = readRequiredMetadataFields(
			values, prefix, gguf.ValueTypeUint32,
			metadataDestination("expert_feed_forward_length", &spec.ExpertFeedForward),
			metadataDestination("expert_gating_func", &spec.ExpertGatingFunc),
		); err != nil {
			return Spec{}, err
		}
		spec.LeadingDenseBlocks, _ = optional[uint32](values, prefix+"leading_dense_block_count", gguf.ValueTypeUint32)
	}
	if validation.Hybrid == HybridValidationBailingMoE2 {
		if err = readRequiredMetadataFields(
			values, prefix, gguf.ValueTypeUint32,
			metadataDestination("expert_feed_forward_length", &spec.ExpertFeedForward),
			metadataDestination("expert_shared_count", &spec.SharedExpertCount),
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
	if validation.Hybrid == HybridValidationQwen2MoE {
		if spec.ExpertFeedForward == 0 {
			spec.ExpertFeedForward = spec.FeedForwardLength
		}
		spec.SharedExpertCount = 1
		spec.SharedExpertFF = spec.FeedForwardLength
		if value, ok := optional[uint32](values, prefix+"expert_shared_feed_forward_length", gguf.ValueTypeUint32); ok {
			spec.SharedExpertFF = value
		}
	}
	if state.declaredExperts || validation.hybridOneOf(HybridValidationOLMoE, HybridValidationPhiMoE) {
		spec.ExpertFeedForward = spec.FeedForwardLength
	}
	if validation.Hybrid == HybridValidationAFMoE {
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
		spec.ExpertGatingFunc = expertGatingSigmoid
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
	if validation.Hybrid == HybridValidationEXAOneMoE {
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
	if validation.Hybrid == HybridValidationLaguna {
		if err = readRequiredMetadataFields(
			values, prefix, gguf.ValueTypeUint32,
			metadataDestination("expert_feed_forward_length", &spec.ExpertFeedForward),
			metadataDestination("leading_dense_block_count", &spec.LeadingDenseBlocks),
			metadataDestination("expert_shared_feed_forward_length", &spec.SharedExpertFF),
		); err != nil {
			return Spec{}, err
		}
		spec.ExpertGatingFunc = expertGatingSigmoid
		spec.ExpertGatingFunc, _ = optional[uint32](values, prefix+"expert_gating_func", gguf.ValueTypeUint32)
		if spec.ExpertGatingFunc == expertGatingUnset {
			spec.ExpertGatingFunc = expertGatingSigmoid
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
	values, prefix, profile := m.values, m.prefix, m.profile
	validation := profile.Validation
	var err error
	if validation.hybridOneOf(HybridValidationLFM2, HybridValidationLFM2MoE) {
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
	if profile.Attention == AttentionLatent || profile.Attention == AttentionSparseLatent {
		if validation.MLA == MLAValidationMiniCPM3 {
			if spec.QLoRARank, err = required[uint32](
				values, prefix+"attention.q_lora_rank", gguf.ValueTypeUint32,
			); err != nil {
				return Spec{}, err
			}
		}
		if profile.Has(ArchitectureLatentKVLayout) {
			if validation.QLoRARankOptional {
				spec.QLoRARank, _ = optional[uint32](values, prefix+"attention.q_lora_rank", gguf.ValueTypeUint32)
			} else if spec.QLoRARank, err = required[uint32](values, prefix+"attention.q_lora_rank", gguf.ValueTypeUint32); err != nil {
				return Spec{}, err
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
		if profile.Has(ArchitectureLatentKVLayout) {
			spec.LeadingDenseBlocks, _ = optional[uint32](values, prefix+"leading_dense_block_count", gguf.ValueTypeUint32)
			if spec.SharedExpertCount, err = required[uint32](values, prefix+"expert_shared_count", gguf.ValueTypeUint32); err != nil {
				return Spec{}, err
			}
			if spec.ExpertFeedForward > 0 && spec.SharedExpertCount > math.MaxUint32/spec.ExpertFeedForward {
				return Spec{}, errors.New("DeepSeek2 shared expert width overflows")
			}
			spec.SharedExpertFF = spec.ExpertFeedForward * spec.SharedExpertCount
			if value, ok := optional[float32](values, prefix+"rope.scaling.yarn_log_multiplier", gguf.ValueTypeFloat32); ok {
				spec.RopeYaRNLogMultiplier = value / yarnLogFactorStep
			}
			if spec.RopeScalingType == ropeScalingYaRN && spec.RopeScalingFactor > 0 {
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
	if profile.Attention == AttentionSparseLatent {
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
		if err = readRequiredMetadataFields(
			values, prefix, gguf.ValueTypeUint32,
			metadataDestination("attention.indexer.head_count", &spec.IndexerHeadCount),
			metadataDestination("attention.indexer.key_length", &spec.IndexerKeyLength),
			metadataDestination("attention.indexer.top_k", &spec.IndexerTopK),
		); err != nil {
			return Spec{}, err
		}
		spec.IndexerFullLayers = make([]bool, spec.BlockCount)
		for index := range spec.IndexerFullLayers {
			spec.IndexerFullLayers[index] = profile.Cadence.fullIndexer(spec.ContextLength, uint32(index))
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
	if validation.MLA == MLAValidationDeepSeek4 {
		if spec.QLoRARank, err = required[uint32](values, prefix+"attention.q_lora_rank", gguf.ValueTypeUint32); err != nil {
			return Spec{}, err
		}
		if spec.RopeDimensionCount, err = required[uint32](values, prefix+"rope.dimension_count", gguf.ValueTypeUint32); err != nil {
			return Spec{}, err
		}
		if err = readRequiredMetadataFields(
			values, prefix, gguf.ValueTypeUint32,
			metadataDestination("attention.sliding_window", &spec.SlidingWindow),
			metadataDestination("attention.indexer.head_count", &spec.IndexerHeadCount),
			metadataDestination("attention.indexer.key_length", &spec.IndexerKeyLength),
			metadataDestination("attention.indexer.top_k", &spec.IndexerTopK),
			metadataDestination("attention.output_group_count", &spec.AttentionOutputGroups),
			metadataDestination("attention.output_lora_rank", &spec.AttentionOutputRank),
			metadataDestination("hyper_connection.count", &spec.HyperConnectionCount),
			metadataDestination("hyper_connection.sinkhorn_iterations", &spec.HyperSinkhornIters),
			metadataDestination("hash_layer_count", &spec.HashLayerCount),
			metadataDestination("expert_count", &spec.ExpertCount),
			metadataDestination("expert_used_count", &spec.ExpertUsedCount),
			metadataDestination("expert_feed_forward_length", &spec.ExpertFeedForward),
			metadataDestination("expert_shared_count", &spec.SharedExpertCount),
			metadataDestination("expert_gating_func", &spec.ExpertGatingFunc),
		); err != nil {
			return Spec{}, err
		}
		if err = readRequiredMetadataFields(
			values, prefix, gguf.ValueTypeFloat32,
			metadataDestination("attention.compress_rope_freq_base", &spec.CompressRopeBase),
			metadataDestination("hyper_connection.epsilon", &spec.HyperConnectionEps),
			metadataDestination("expert_weights_scale", &spec.ExpertWeightsScale),
		); err != nil {
			return Spec{}, err
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
	if validation.attentionOneOf(
		AttentionValidationGemma2, AttentionValidationGemma3, AttentionValidationGemma3N,
		AttentionValidationGemma4, AttentionValidationGemma4Assistant,
		AttentionValidationOLMo2, AttentionValidationCohere2, AttentionValidationCohere2MoE,
	) {
		spec.RopeFrequencySWA = spec.RopeFrequencyBase
		if validation.attentionOneOf(
			AttentionValidationGemma3, AttentionValidationGemma3N,
			AttentionValidationGemma4, AttentionValidationGemma4Assistant,
		) {
			spec.RopeFrequencySWA = 10000
		}
		if value, ok := optional[float32](
			values,
			prefix+"rope.freq_base_swa",
			gguf.ValueTypeFloat32,
		); ok {
			spec.RopeFrequencySWA = value
		}
		if validation.Attention == AttentionValidationGemma2 {
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
			if validation.Attention == AttentionValidationGemma2 {
				spec.SlidingPattern = 2
			}
			if validation.Attention == AttentionValidationGemma3 {
				spec.SlidingPattern = 6
			}
			if validation.Attention == AttentionValidationGemma3N {
				spec.SlidingPattern = 5
			}
			if validation.attentionOneOf(
				AttentionValidationGemma4, AttentionValidationGemma4Assistant,
			) {
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
			if validation.attentionOneOf(AttentionValidationCohere2, AttentionValidationCohere2MoE) &&
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
			if validation.Attention == AttentionValidationCohere2 && len(spec.SlidingLayers) == 0 {
				spec.NoRopeLayerStep = spec.SlidingPattern
			}
		}
	}
	if validation.Hybrid == HybridValidationLlama4 {
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
	if validation.Hybrid == HybridValidationGPTOSS {
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
	if validation.Attention == AttentionValidationGemma4 {
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
	if validation.Attention == AttentionValidationGemma4Assistant {
		spec.RopeDimensionCount = spec.KeyLength
		spec.RopeDimensionSWA = spec.KeyLengthSWA
		spec.AttentionScale = 1
	}
	if spec.RopeScalingType == ropeScalingLongRoPE && profile.Has(ArchitectureLongRoPE) {
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
	if spec.RopeScalingType == ropeScalingYaRN &&
		(validation.Hybrid == HybridValidationLlama ||
			validation.Hybrid == HybridValidationRopeScaling && validation.MLA == MLAValidationNone) {
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
	if validation.encoderOneOf(
		EncoderValidationBERT, EncoderValidationJinaV2, EncoderValidationJinaV3,
		EncoderValidationNomicBERT, EncoderValidationNomicBERTMoE,
	) {
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
	usage := s.Profile().Rotary.Usage
	if usage == RotaryUsageSlidingMetadata && len(s.SlidingLayers) == 0 {
		usage = RotaryUsageStandard
	}
	switch usage {
	case RotaryUsageSlidingMetadata, RotaryUsageDensePrefixOrSliding:
		return !s.RopeDisabled && block < s.BlockCount &&
			(usage == RotaryUsageDensePrefixOrSliding && block < s.LeadingDenseBlocks || s.IsSlidingLayer(block))
	case RotaryUsagePeriodicZeroBased:
		return !s.RopeDisabled && block < s.BlockCount &&
			(s.SlidingWindow == 0 || s.NoRopeLayerStep == 0 || block%s.NoRopeLayerStep != 0)
	case RotaryUsageSlidingOnly:
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
	return s.Profile().Runtime.inputEmbeddingScale(s)
}

func (s Spec) OutputLogitMultiplier() float32 {
	return s.Profile().Runtime.outputLogitMultiplier(s)
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
	return profile.Runtime.normalizationPlan(s, profile)
}

func (s Spec) validate() error {
	if _, ok := s.boundProfile(); !ok {
		return fmt.Errorf("model profile for %q is not bound", s.Architecture)
	}
	if err := s.validateBaseMetadata(); err != nil {
		return err
	}
	if err := s.validateRecurrentMetadata(); err != nil {
		return err
	}
	if err := s.validateEncoderMetadata(); err != nil {
		return err
	}
	if err := s.validateHybridMetadata(); err != nil {
		return err
	}
	if err := s.validateMLAMetadata(); err != nil {
		return err
	}
	if err := s.validateAttentionMetadata(); err != nil {
		return err
	}
	if s.RopeScalingType == ropeScalingLinear && s.RopeScalingFactor <= 0 {
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
