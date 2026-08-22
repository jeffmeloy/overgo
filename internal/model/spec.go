package model

import (
	"errors"
	"fmt"
	"math"
	"slices"

	"overgo/internal/gguf"
	"overgo/internal/hostmath"
	"overgo/internal/tensor"
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
	if err := metadata.readAttentionMetadata(&spec, state); err != nil {
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
	spec, err := m.runMetadataProgram(spec, compileArchitectureCoreProgram(m.profile))
	if err != nil {
		return Spec{}, err
	}
	if validation.Attention == AttentionValidationOptionalExperts {
		spec.ExpertCount, _ = optional[uint32](
			values, prefix+"expert_count", gguf.ValueTypeUint32,
		)
		if spec.HasExperts() {
			if spec.ExpertUsedCount, err = required[uint32](
				values, prefix+"expert_used_count", gguf.ValueTypeUint32,
			); err != nil {
				return Spec{}, err
			}
			spec.ExpertFeedForward = optionalOr(
				values, prefix+"expert_feed_forward_length", gguf.ValueTypeUint32, spec.FeedForwardLength,
			)
			spec.ExpertWeightsNorm = true
			spec.ExpertWeightsScale = tensor.UnitScale
		}
	}
	if validation.Recurrent == RecurrentValidationTargetLayerBlock {
		if window, ok := optional[uint32](values, prefix+"attention.sliding_window", gguf.ValueTypeUint32); ok && window > tensor.FirstOffset {
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
	if validation.hybridOneOf(
		HybridValidationScaledDense, HybridValidationScaledExperts, HybridValidationScaledStateSpace,
	) {
		if validation.Hybrid == HybridValidationScaledStateSpace {
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
		spec.OriginalContextLength = optionalOr(
			values, prefix+"rope.scaling.original_context_length", gguf.ValueTypeUint32, spec.ContextLength,
		)
		if spec.OriginalContextLength == tensor.FirstOffset {
			spec.OriginalContextLength = spec.ContextLength
		}
		spec.RopeAttentionFactor = optionalOr(
			values, prefix+"rope.scaling.attn_factor", gguf.ValueTypeFloat32, tensor.UnitScale,
		)
		if spec.RopeAttentionFactor == tensor.FirstOffset {
			spec.RopeAttentionFactor = tensor.UnitScale
		}
		spec.RopeDisabled = !optionalOr(
			values,
			prefix+"rope.scaling.finetuned",
			gguf.ValueTypeBool,
			true,
		)
		if mapping, ok, mappingErr := optionalArray[int32](
			values,
			prefix+"deepstack_mapping",
			gguf.ValueTypeInt32,
		); mappingErr != nil {
			return Spec{}, mappingErr
		} else if ok && len(mapping) > tensor.FirstOffset {
			if validation.Hybrid != HybridValidationScaledDense {
				return Spec{}, errors.New("Granite deepstack mapping requires granite architecture")
			}
			if len(mapping) != int(spec.BlockCount) {
				return Spec{}, fmt.Errorf("Granite deepstack mapping has %d entries, need %d", len(mapping), spec.BlockCount)
			}
			unique := make(map[int32]struct{})
			for _, index := range mapping {
				if index < int32(DeepstackSourceBase) {
					return Spec{}, errors.New("Granite deepstack mapping index is invalid")
				}
				if index >= tensor.FirstOffset {
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
	if validation.Attention == AttentionValidationFullHeadSlidingRotary {
		if value, ok := optional[uint32](values, prefix+"attention.sliding_window", gguf.ValueTypeUint32); ok {
			spec.SlidingWindow = value
		}
		hiddenActivation := optionalOr(values, prefix+"hidden_activation", gguf.ValueTypeString, "geglu")
		switch hiddenActivation {
		case "gelu", "geglu":
			spec.HiddenActivation = "geglu"
		case "silu", "swish", "swiglu":
			spec.HiddenActivation = "swiglu"
		case "reglu":
			spec.HiddenActivation = "reglu"
		default:
			return Spec{}, fmt.Errorf("ModernBERT hidden activation %q is unsupported", hiddenActivation)
		}
	}
	if validation.Attention == AttentionValidationFullScaledRotaryXIELU {
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
	if validation.Hybrid == HybridValidationSigmoidExperts {
		if spec.SlidingWindow, err = required[uint32](values, prefix+"attention.sliding_window", gguf.ValueTypeUint32); err != nil {
			return Spec{}, err
		}
		patternKey := prefix + "attention.sliding_window_pattern"
		pattern, exists := values[patternKey]
		if !exists {
			return Spec{}, fmt.Errorf("required metadata %q is missing", patternKey)
		}
		if pattern.Type == gguf.ValueTypeUint32 {
			value, valid := pattern.Data.(uint32)
			if !valid {
				return Spec{}, fmt.Errorf("required metadata %q is missing", patternKey)
			}
			spec.SlidingPattern = value
		} else {
			if pattern.Type != gguf.ValueTypeArray {
				return Spec{}, fmt.Errorf("required metadata %q is missing", patternKey)
			}
			layers, layerErr := requiredLayerBoolCompatible(values, patternKey, declaredBlockCount)
			if layerErr != nil {
				return Spec{}, layerErr
			}
			spec.SlidingLayers = slices.Clone(layers[:spec.BlockCount])
		}
		if value, ok := optional[float32](values, prefix+"attention.value_scale", gguf.ValueTypeFloat32); ok && value != tensor.UnitScale {
			spec.AttentionValueScale = value
		}
	}
	if validation.Hybrid == HybridValidationCompressedHyperDraft {
		if spec.SlidingWindow, err = required[uint32](values, prefix+"attention.sliding_window", gguf.ValueTypeUint32); err != nil {
			return Spec{}, err
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
	if validation.Attention == AttentionValidationSharedKVAttention {
		if value, ok := optional[uint32](
			values, prefix+"attention.sliding_window", gguf.ValueTypeUint32,
		); ok {
			spec.SlidingWindow = value
		}
		if spec.SlidingWindow > tensor.FirstOffset {
			if value, ok := optional[uint32](values, prefix+"attention.sliding_window_pattern", gguf.ValueTypeUint32); ok {
				spec.SlidingPattern = value
			}
			spec.NoRopeLayerStep = spec.SlidingPattern
		}
	}
	if validation.Hybrid == HybridValidationSlidingSharedExperts {
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
	if m.profile.readsMetadata(MetadataReadBaichuanBlocks) &&
		spec.BlockCount != validation.RequiredBlockCount && spec.BlockCount != validation.AlternateBlockCount {
		return Spec{}, fmt.Errorf(
			"metadata block count must select profile variant %d or %d",
			validation.RequiredBlockCount, validation.AlternateBlockCount,
		)
	}
	if validation.MLA == MLAValidationOptionalExpertsLatent {
		spec.OriginalContextLength = optionalOr(
			values, prefix+"rope.scaling.original_context_length", gguf.ValueTypeUint32, spec.ContextLength,
		)
		spec.AttentionTempScale, _ = optional[float32](
			values, prefix+"attention.temperature_scale", gguf.ValueTypeFloat32,
		)
		if spec.AttentionTempScale != tensor.FirstOffset {
			spec.AttentionTempFloor = spec.OriginalContextLength
		}
		if spec.RopeScalingType == ropeScalingYaRN {
			spec.RopeYaRNLogMultiplier, _ = optional[float32](
				values, prefix+"rope.scaling.yarn_log_multiplier", gguf.ValueTypeFloat32,
			)
			rawAttentionFactor := optionalOr(
				values, prefix+"rope.scaling.attn_factor", gguf.ValueTypeFloat32, tensor.UnitScale,
			)
			denominator := tensor.UnitScale
			if spec.RopeYaRNLogMultiplier != tensor.FirstOffset {
				denominator += yarnLogFactorStep * spec.RopeYaRNLogMultiplier *
					float32(math.Log(float64(spec.RopeScalingFactor)))
			}
			spec.YaRNAttentionFactor = rawAttentionFactor / denominator
		}
		spec.ExpertCount, _ = optional[uint32](values, prefix+"expert_count", gguf.ValueTypeUint32)
		if spec.HasExperts() {
			if spec.ExpertUsedCount, err = required[uint32](
				values, prefix+"expert_used_count", gguf.ValueTypeUint32,
			); err != nil {
				return Spec{}, err
			}
			spec.ExpertFeedForward = spec.FeedForwardLength
			spec.ExpertWeightsNorm = true
			spec.ExpertWeightsScale = tensor.UnitScale
			if value, ok := optional[float32](
				values, prefix+"expert_weights_scale", gguf.ValueTypeFloat32,
			); ok && value != tensor.FirstOffset {
				spec.ExpertWeightsScale = value
			}
		}
	}
	if validation.Attention == AttentionValidationLayerwiseQKNorm {
		if sections, ok, sectionsErr := optionalArray[int32](values, prefix+"rope.dimension_sections", gguf.ValueTypeInt32); sectionsErr != nil {
			return Spec{}, sectionsErr
		} else if ok {
			if len(sections) != tensor.MaxDimensions {
				return Spec{}, fmt.Errorf("metadata %q has %d values, need %d", prefix+"rope.dimension_sections", len(sections), tensor.MaxDimensions)
			}
			copy(spec.RopeSections[:], sections)
		}
		if alpha, ok := optional[float32](values, prefix+"rope.scaling.alpha", gguf.ValueTypeFloat32); ok && alpha != tensor.FirstOffset {
			if alpha < tensor.FirstOffset || spec.KeyLength <= tensor.PairedExtent || !finite(alpha) {
				return Spec{}, errors.New("Hunyuan-Dense XDRoPE alpha is invalid")
			}
			exponent := float64(spec.KeyLength) / float64(spec.KeyLength-tensor.PairedExtent)
			spec.RopeFrequencyBase *= float32(math.Pow(float64(alpha), exponent))
		}
	}
	if validation.Hybrid == HybridValidationDualExpertProduct {
		spec.NoRopeLayerStep = spec.BlockCount
		if value, ok := optional[uint32](
			values, prefix+"attention.sliding_window", gguf.ValueTypeUint32,
		); ok && value > tensor.FirstOffset {
			spec.SlidingWindow = value
			spec.NoRopeLayerStep = spec.SlidingPattern
		}
	}
	if validation.Hybrid == HybridValidationRequiredExpertFeedForward ||
		validation.Attention == AttentionValidationPerLayerSlidingAttention {
		if value, ok := optional[uint32](values, prefix+"attention.sliding_window", gguf.ValueTypeUint32); ok {
			spec.SlidingWindow = value
		}
		if spec.SlidingWindow > tensor.FirstOffset {
			_, scalar := optional[uint32](values, prefix+"attention.sliding_window_pattern", gguf.ValueTypeUint32)
			if layers, ok, arrayErr := optionalArray[bool](values, prefix+"attention.sliding_window_pattern", gguf.ValueTypeBool); !scalar && arrayErr != nil {
				return Spec{}, arrayErr
			} else if !scalar && ok {
				if len(layers) != int(spec.BlockCount) {
					return Spec{}, fmt.Errorf("metadata %q has %d values, need %d", prefix+"attention.sliding_window_pattern", len(layers), spec.BlockCount)
				}
				spec.SlidingLayers = slices.Clone(layers)
			}
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
			spec.DecoderBlockCount = m.profile.MetadataDefaults.uint(
				values, prefix, "decoder_block_count", spec.BlockCount,
			)
			spec.DecoderStartTokenID, _ = optional[uint32](values, prefix+"decoder_start_token_id", gguf.ValueTypeUint32)
		}
	}
	if validation.hybridOneOf(
		HybridValidationAlternatingGatedDelta, HybridValidationAlternatingGatedDeltaHybrid, HybridValidationAlternatingGatedDeltaExperts,
	) {
		spec.FullAttentionInterval = m.profile.MetadataDefaults.uint(
			values, prefix, "full_attention_interval", m.profile.MetadataDefaults.FullAttentionInterval,
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
	if validation.Recurrent == RecurrentValidationUngroupedScheduledStateSpace {
		spec.AttentionScale = hostmath.InvSqrt32(uint64(spec.ValueLength))
	}
	if validation.Recurrent == RecurrentValidationGroupedStateSpaceOptionalExperts {
		spec.ExpertCount, _ = optional[uint32](values, prefix+"expert_count", gguf.ValueTypeUint32)
		if spec.HasExperts() {
			if spec.ExpertUsedCount, err = required[uint32](values, prefix+"expert_used_count", gguf.ValueTypeUint32); err != nil {
				return Spec{}, err
			}
			spec.ExpertFeedForward = spec.FeedForwardLength
			spec.ExpertWeightsScale = tensor.UnitScale
			spec.ExpertWeightsNorm = true
			spec.SharedExpertFF, _ = optional[uint32](values, prefix+"expert_shared_feed_forward_length", gguf.ValueTypeUint32)
		}
	}
	if validation.Recurrent == RecurrentValidationScheduledStateSpaceExperts {
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
		spec.ExpertWeightsScale = optionalOr(
			values, prefix+"expert_weights_scale", gguf.ValueTypeFloat32, tensor.UnitScale,
		)
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
		if spec.ExpertUsedCount == tensor.FirstOffset {
			return Spec{}, errors.New("expert used count is zero")
		}
		if value, ok := optional[uint32](
			values, prefix+"expert_feed_forward_length", gguf.ValueTypeUint32,
		); ok {
			spec.ExpertFeedForward = value
		} else if spec.ExpertFeedForward, err = derivedExpertFeedForward(spec); err != nil {
			return Spec{}, err
		}
		if validation.Hybrid == HybridValidationRoutedExperts {
			spec.ExpertFeedForward = spec.FeedForwardLength
		}
		if validation.Encoder == EncoderValidationRotaryPeriodicExperts {
			spec.ExpertFeedForward = spec.FeedForwardLength
		}
		if validation.Recurrent == RecurrentValidationStateSpaceAttentionExperts {
			spec.ExpertFeedForward = spec.FeedForwardLength
			spec.ExpertWeightsNorm = false
		}
		if validation.MLA == MLAValidationHybridLinearAttention {
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
		if validation.Hybrid == HybridValidationChunkedExperts {
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
			values, prefix+"expert_weights_scale", gguf.ValueTypeFloat32, tensor.UnitScale,
		)
		if validation.Hybrid == HybridValidationSelectedSoftmaxExperts {
			spec.ExpertGatingFunc = expertGatingSelectedSoftmax
			spec.ExpertWeightsNorm = false
		}
		if validation.hybridOneOf(HybridValidationAlternatingGatedDelta, HybridValidationAlternatingGatedDeltaExperts) {
			if spec.SharedExpertFF, err = profile.MetadataDefaults.readSharedExpertFeedForward(values, prefix, spec); err != nil {
				return Spec{}, err
			}
		}
	}
	if profile.Has(ArchitectureLatentKVLayout) {
		if validation.MLA == MLAValidationSparseLatentIndexer {
			if spec.ExpertCount, err = required[uint32](values, prefix+"expert_count", gguf.ValueTypeUint32); err != nil {
				return Spec{}, err
			}
		} else {
			spec.ExpertCount, _ = optional[uint32](values, prefix+"expert_count", gguf.ValueTypeUint32)
		}
		if spec.ExpertFeedForward, err = required[uint32](values, prefix+"expert_feed_forward_length", gguf.ValueTypeUint32); err != nil {
			return Spec{}, err
		}
		if spec.HasExperts() {
			if spec.ExpertUsedCount, err = required[uint32](values, prefix+"expert_used_count", gguf.ValueTypeUint32); err != nil {
				return Spec{}, err
			}
		}
		spec.ExpertWeightsScale = optionalOr(
			values, prefix+"expert_weights_scale", gguf.ValueTypeFloat32, tensor.UnitScale,
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
		if validation.MLA == MLAValidationSparseLatentIndexer {
			if spec.ExpertGatingFunc, err = required[uint32](values, prefix+"expert_gating_func", gguf.ValueTypeUint32); err != nil {
				return Spec{}, err
			}
		} else if value, ok := optional[uint32](values, prefix+"expert_gating_func", gguf.ValueTypeUint32); ok && value != tensor.FirstOffset {
			spec.ExpertGatingFunc = value
		}
	}
	if spec, err = m.runMetadataProgram(spec, compileExpertProgram(profile)); err != nil {
		return Spec{}, err
	}
	if validation.Attention == AttentionValidationPerLayerDualRotaryAttention {
		if count, ok := optional[uint32](values, prefix+"expert_count", gguf.ValueTypeUint32); ok && count > tensor.FirstOffset {
			spec.ExpertCount = count
			if err = readRequiredMetadataFields(
				values, prefix, gguf.ValueTypeUint32,
				metadataDestination("expert_used_count", &spec.ExpertUsedCount),
				metadataDestination("expert_feed_forward_length", &spec.ExpertFeedForward),
			); err != nil {
				return Spec{}, err
			}
			spec.ExpertWeightsScale = tensor.UnitScale
			spec.ExpertWeightsNorm = true
		}
	}
	if validation.Hybrid == HybridValidationCompressedHyperDraft {
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
		spec.MoELayerStep = profile.MetadataDefaults.uint(
			values, prefix, "moe_every_n_layers", profile.MetadataDefaults.MoELayerStep,
		)
		spec.ExpertGatingFunc = expertGatingSigmoid
		if value, ok := optional[uint32](values, prefix+"expert_gating_func", gguf.ValueTypeUint32); ok && value != tensor.FirstOffset {
			spec.ExpertGatingFunc = value
		}
		spec.ExpertWeightsNorm, _ = optional[bool](values, prefix+"expert_weights_norm", gguf.ValueTypeBool)
	}
	if validation.Attention == AttentionValidationOptionalRopeSectionsExperts {
		spec.LeadingDenseBlocks, _ = optional[uint32](values, prefix+"leading_dense_block_count", gguf.ValueTypeUint32)
		if spec.SharedExpertCount, err = required[uint32](values, prefix+"expert_shared_count", gguf.ValueTypeUint32); err != nil {
			return Spec{}, err
		}
		if spec.ExpertFeedForward > tensor.FirstOffset && spec.SharedExpertCount > math.MaxUint32/spec.ExpertFeedForward {
			return Spec{}, errors.New("GLM4-MoE shared expert width overflows")
		}
		spec.SharedExpertFF = spec.ExpertFeedForward * spec.SharedExpertCount
		spec.ExpertGatingFunc = optionalOr(
			values, prefix+"expert_gating_func", gguf.ValueTypeUint32, expertGatingSigmoid,
		)
		if value, ok := optional[bool](values, prefix+"expert_weights_norm", gguf.ValueTypeBool); ok {
			spec.ExpertWeightsNorm = value
		}
	}
	if validation.Hybrid == HybridValidationSparseSharedExperts {
		spec.ExpertChunkFeedForward = profile.MetadataDefaults.uint(
			values, prefix, "expert_chunk_feed_forward_length", spec.KeyLength,
		)
		if spec.ExpertGroupScale, err = required[float32](values, prefix+"expert_group_scale", gguf.ValueTypeFloat32); err != nil {
			return Spec{}, err
		}
		if spec.ExpertsPerGroup, err = required[uint32](values, prefix+"experts_per_group", gguf.ValueTypeUint32); err != nil {
			return Spec{}, err
		}
	}
	if validation.Attention == AttentionValidationRequiredSlidingRotaryExperts {
		if err = readRequiredMetadataFields(
			values, prefix, gguf.ValueTypeUint32,
			metadataDestination("leading_dense_block_count", &spec.LeadingDenseBlocks),
			metadataDestination("expert_feed_forward_length", &spec.ExpertFeedForward),
		); err != nil {
			return Spec{}, err
		}
		spec.ExpertGatingFunc = optionalOr(
			values, prefix+"expert_gating_func", gguf.ValueTypeUint32, expertGatingSigmoid,
		)
		if value, ok := optional[bool](values, prefix+"expert_weights_norm", gguf.ValueTypeBool); ok {
			spec.ExpertWeightsNorm = value
		}
		if value, ok := optional[uint32](values, prefix+"expert_shared_count", gguf.ValueTypeUint32); ok {
			spec.SharedExpertCount = value
		}
		if spec.HasSharedExperts() {
			if spec.SharedExpertFF, err = profile.MetadataDefaults.readSharedExpertFeedForward(values, prefix, spec); err != nil {
				return Spec{}, err
			}
		}
	}
	if validation.Hybrid == HybridValidationMultiHeadDraft {
		if spec.ExpertFeedForward, err = required[uint32](values, prefix+"expert_feed_forward_length", gguf.ValueTypeUint32); err != nil {
			return Spec{}, err
		}
		if spec.SharedExpertFF, err = profile.MetadataDefaults.readSharedExpertFeedForward(values, prefix, spec); err != nil {
			return Spec{}, err
		}
		spec.ExpertGatingFunc = optionalOr(
			values, prefix+"expert_gating_func", gguf.ValueTypeUint32, expertGatingSigmoid,
		)
		if value, ok := optional[bool](values, prefix+"expert_weights_norm", gguf.ValueTypeBool); ok {
			spec.ExpertWeightsNorm = value
		}
	}
	if validation.Hybrid == HybridValidationFullRotaryVision {
		spec.LeadingDenseBlocks, _ = optional[uint32](values, prefix+"leading_dense_block_count", gguf.ValueTypeUint32)
		if err = readRequiredMetadataFields(
			values, prefix, gguf.ValueTypeUint32,
			metadataDestination("expert_feed_forward_length", &spec.ExpertFeedForward),
			metadataDestination("expert_shared_count", &spec.SharedExpertCount),
		); err != nil {
			return Spec{}, err
		}
		if spec.SharedExpertCount > math.MaxUint32/spec.ExpertFeedForward {
			return Spec{}, errors.New("DeepSeek2-OCR shared expert width overflows")
		}
		spec.SharedExpertFF = spec.ExpertFeedForward * spec.SharedExpertCount
		spec.ExpertGatingFunc = optionalOr(
			values, prefix+"expert_gating_func", gguf.ValueTypeUint32, expertGatingSoftmax,
		)
		if value, ok := optional[bool](values, prefix+"expert_weights_norm", gguf.ValueTypeBool); ok {
			spec.ExpertWeightsNorm = value
		}
	}
	if validation.Encoder == EncoderValidationRotaryOptionalExperts {
		spec.ExpertCount, _ = optional[uint32](values, prefix+"expert_count", gguf.ValueTypeUint32)
		spec.ExpertUsedCount, _ = optional[uint32](values, prefix+"expert_used_count", gguf.ValueTypeUint32)
		spec.MoELayerStep, _ = optional[uint32](values, prefix+"moe_every_n_layers", gguf.ValueTypeUint32)
		if spec.HasExperts() {
			spec.ExpertFeedForward = spec.FeedForwardLength
			spec.ExpertWeightsScale = optionalOr(
				values, prefix+"expert_weights_scale", gguf.ValueTypeFloat32, tensor.UnitScale,
			)
		}
	}
	if validation.Encoder == EncoderValidationRotary {
		if cadence, ok := optional[uint32](values, prefix+"moe_every_n_layers", gguf.ValueTypeUint32); ok && cadence > tensor.FirstOffset {
			return Spec{}, errors.New("NomicBERT MoE cadence requires nomic-bert-moe architecture")
		}
	}
	if validation.Hybrid == HybridValidationSharedExpertNorm {
		if _, ok := optional[uint32](
			values, prefix+"expert_feed_forward_length", gguf.ValueTypeUint32,
		); !ok {
			spec.ExpertFeedForward = spec.FeedForwardLength
		}
		spec.ExpertWeightsNorm = true
	}
	if validation.Hybrid == HybridValidationScaledExperts ||
		validation.Hybrid == HybridValidationScaledDense && spec.HasExperts() {
		spec.ExpertFeedForward = spec.FeedForwardLength
		spec.ExpertWeightsNorm = true
		spec.SharedExpertFF, _ = optional[uint32](
			values, prefix+"expert_shared_feed_forward_length", gguf.ValueTypeUint32,
		)
	}
	if validation.Hybrid == HybridValidationScaledSigmoidExperts {
		expertFeedForward, readErr := required[uint32](
			values, prefix+"expert_feed_forward_length", gguf.ValueTypeUint32,
		)
		if readErr != nil {
			return Spec{}, readErr
		}
		if expertFeedForward != spec.FeedForwardLength {
			return Spec{}, errors.New("expert feed-forward extent differs from packed tensor extent")
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
	if state.declaredExperts || validation.hybridOneOf(HybridValidationFullHeadExperts, HybridValidationBasicScaledExperts) {
		spec.ExpertFeedForward = spec.FeedForwardLength
	}
	if validation.Hybrid == HybridValidationSlidingSigmoidExperts {
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
		if value, ok := optional[uint32](values, prefix+"expert_gating_func", gguf.ValueTypeUint32); ok && value != tensor.FirstOffset {
			spec.ExpertGatingFunc = value
		}
		spec.ExpertWeightsNorm, _ = optional[bool](values, prefix+"expert_weights_norm", gguf.ValueTypeBool)
		if value, ok := optional[uint32](values, prefix+"attention.sliding_window", gguf.ValueTypeUint32); ok {
			spec.SlidingWindow = value
		}
	}
	if validation.Hybrid == HybridValidationSlidingSharedExperts {
		if spec.ExpertFeedForward, err = required[uint32](values, prefix+"expert_feed_forward_length", gguf.ValueTypeUint32); err != nil {
			return Spec{}, err
		}
		spec.SharedExpertFF = optionalOr(
			values, prefix+"expert_shared_feed_forward_length", gguf.ValueTypeUint32, spec.ExpertFeedForward,
		)
		spec.SharedExpertCount, _ = optional[uint32](values, prefix+"expert_shared_count", gguf.ValueTypeUint32)
		spec.LeadingDenseBlocks, _ = optional[uint32](values, prefix+"leading_dense_block_count", gguf.ValueTypeUint32)
		if spec.ExpertGatingFunc, err = required[uint32](values, prefix+"expert_gating_func", gguf.ValueTypeUint32); err != nil {
			return Spec{}, err
		}
		spec.ExpertWeightsNorm, _ = optional[bool](values, prefix+"expert_weights_norm", gguf.ValueTypeBool)
	}
	if validation.Hybrid == HybridValidationPerLayerYaRNExperts {
		if err = readRequiredMetadataFields(
			values, prefix, gguf.ValueTypeUint32,
			metadataDestination("expert_feed_forward_length", &spec.ExpertFeedForward),
			metadataDestination("leading_dense_block_count", &spec.LeadingDenseBlocks),
			metadataDestination("expert_shared_feed_forward_length", &spec.SharedExpertFF),
		); err != nil {
			return Spec{}, err
		}
		spec.ExpertGatingFunc = optionalOr(
			values, prefix+"expert_gating_func", gguf.ValueTypeUint32, expertGatingSigmoid,
		)
		if spec.ExpertGatingFunc == expertGatingUnset {
			spec.ExpertGatingFunc = expertGatingSigmoid
		}
		spec.ExpertWeightsNorm, _ = optional[bool](values, prefix+"expert_weights_norm", gguf.ValueTypeBool)
		sharedCount := optionalOr(
			values, prefix+"expert_shared_count", gguf.ValueTypeUint32, uint32(tensor.SingletonExtent),
		)
		if sharedCount != tensor.SingletonExtent {
			return Spec{}, errors.New("Laguna requires exactly one shared expert")
		}
		if value, ok := optional[uint32](values, prefix+"attention.sliding_window", gguf.ValueTypeUint32); ok {
			spec.SlidingWindow = value
		}
		if spec.SlidingWindow > tensor.FirstOffset {
			spec.RopeDimensionSWA = optionalOr(
				values, prefix+"rope.dimension_count_swa", gguf.ValueTypeUint32, spec.KeyLength,
			)
		}
	}
	return spec, nil
}

func (m specMetadata) readRuntimeMetadata(spec Spec, _ specReadState) (Spec, error) {
	values, prefix, profile := m.values, m.prefix, m.profile
	validation := profile.Validation
	spec, err := m.runMetadataProgram(spec, compileRuntimeProgram(profile))
	if err != nil {
		return Spec{}, err
	}
	if profile.Attention == AttentionLatent || profile.Attention == AttentionSparseLatent {
		if validation.MLA == MLAValidationScaledLatent {
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
		if err = readRequiredMetadataFields(
			values, prefix, gguf.ValueTypeUint32,
			metadataDestination("attention.kv_lora_rank", &spec.KVLoRARank),
			metadataDestination("rope.dimension_count", &spec.RopeDimensionCount),
		); err != nil {
			return Spec{}, err
		}
		if profile.Has(ArchitectureLatentKVLayout) {
			spec.LeadingDenseBlocks, _ = optional[uint32](values, prefix+"leading_dense_block_count", gguf.ValueTypeUint32)
			if spec.SharedExpertCount, err = required[uint32](values, prefix+"expert_shared_count", gguf.ValueTypeUint32); err != nil {
				return Spec{}, err
			}
			if spec.ExpertFeedForward > tensor.FirstOffset && spec.SharedExpertCount > math.MaxUint32/spec.ExpertFeedForward {
				return Spec{}, errors.New("DeepSeek2 shared expert width overflows")
			}
			spec.SharedExpertFF = spec.ExpertFeedForward * spec.SharedExpertCount
			if value, ok := optional[float32](values, prefix+"rope.scaling.yarn_log_multiplier", gguf.ValueTypeFloat32); ok {
				spec.RopeYaRNLogMultiplier = value / yarnLogFactorStep
			}
			if spec.RopeScalingType == ropeScalingYaRN && positiveFinite(spec.RopeScalingFactor) {
				rawAttentionFactor := optionalOr(
					values, prefix+"rope.scaling.attn_factor", gguf.ValueTypeFloat32, tensor.UnitScale,
				)
				spec.YaRNAttentionFactor = rawAttentionFactor /
					(tensor.UnitScale + yarnLogFactorStep*float32(math.Log(float64(spec.RopeScalingFactor))))
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
			if !valid || typeValue > tensor.SingletonExtent {
				return Spec{}, fmt.Errorf("metadata %q must be 0 or 1", indexerTypesKey)
			}
			for index := range spec.IndexerFullLayers {
				spec.IndexerFullLayers[index] = typeValue == tensor.SingletonExtent
			}
		} else if types, ok, arrayErr := optionalArray[uint32](values, indexerTypesKey, gguf.ValueTypeUint32); arrayErr != nil {
			return Spec{}, arrayErr
		} else if ok {
			if len(types) != int(spec.BlockCount) {
				return Spec{}, fmt.Errorf("metadata %q has %d values, need %d", indexerTypesKey, len(types), spec.BlockCount)
			}
			for index, typeValue := range types {
				if typeValue > tensor.SingletonExtent {
					return Spec{}, fmt.Errorf("metadata %q value %d is not 0 or 1", indexerTypesKey, typeValue)
				}
				spec.IndexerFullLayers[index] = typeValue == tensor.SingletonExtent
			}
		}
	}
	if validation.MLA == MLAValidationCompressedHyper {
		if err = readRequiredMetadataFields(
			values, prefix, gguf.ValueTypeUint32,
			metadataDestination("attention.q_lora_rank", &spec.QLoRARank),
			metadataDestination("rope.dimension_count", &spec.RopeDimensionCount),
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
		if len(spec.LayerSwiGLUClamp) == tensor.FirstOffset {
			return Spec{}, errors.New("DeepSeek 4 expert SwiGLU clamp is missing")
		}
		if spec.LayerSharedSwiGLUClamp, err = optionalLayerFloat32(values, prefix+"swiglu_clamp_shexp", spec.BlockCount); err != nil {
			return Spec{}, err
		}
		if len(spec.LayerSharedSwiGLUClamp) == tensor.FirstOffset {
			spec.LayerSharedSwiGLUClamp = slices.Clone(spec.LayerSwiGLUClamp)
		}
		if spec.SharedExpertCount > math.MaxUint32/spec.ExpertFeedForward {
			return Spec{}, errors.New("DeepSeek 4 shared expert width overflows")
		}
		spec.SharedExpertFF = spec.SharedExpertCount * spec.ExpertFeedForward
	}
	if validation.attentionOneOf(
		AttentionValidationRequiredSlidingFrequency, AttentionValidationScaledSlidingAttention, AttentionValidationSharedKVAlternatingState,
		AttentionValidationPerLayerDualRotaryAttention, AttentionValidationTargetHiddenDualRotaryAttention,
		AttentionValidationOptionalSlidingFrequency, AttentionValidationRequiredSlidingRotary, AttentionValidationRequiredSlidingRotaryExperts,
	) {
		if value, ok := optional[uint32](
			values,
			prefix+"attention.sliding_window",
			gguf.ValueTypeUint32,
		); ok {
			spec.SlidingWindow = value
		}
		if spec.SlidingWindow > tensor.FirstOffset {
			if validation.attentionOneOf(
				AttentionValidationPerLayerDualRotaryAttention, AttentionValidationTargetHiddenDualRotaryAttention,
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
			if validation.attentionOneOf(AttentionValidationRequiredSlidingRotary, AttentionValidationRequiredSlidingRotaryExperts) &&
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
			if validation.Attention == AttentionValidationRequiredSlidingRotary && len(spec.SlidingLayers) == tensor.FirstOffset {
				spec.NoRopeLayerStep = spec.SlidingPattern
			}
		}
	}
	if validation.Attention == AttentionValidationPerLayerDualRotaryAttention {
		if err = readRequiredMetadataFields(
			values, prefix, gguf.ValueTypeUint32,
			metadataDestination("rope.dimension_count", &spec.RopeDimensionCount),
			metadataDestination("rope.dimension_count_swa", &spec.RopeDimensionSWA),
			metadataDestination("embedding_length_per_layer_input", &spec.EmbeddingPerLayer),
		); err != nil {
			return Spec{}, err
		}
		spec.SharedKVLayers, _ = optional[uint32](
			values, prefix+"attention.shared_kv_layers", gguf.ValueTypeUint32,
		)
		spec.AttentionScale = tensor.UnitScale
	}
	if validation.Attention == AttentionValidationTargetHiddenDualRotaryAttention {
		spec.RopeDimensionSWA = spec.KeyLengthSWA
		spec.AttentionScale = tensor.UnitScale
	}
	if spec.RopeScalingType == ropeScalingLongRoPE && profile.Has(ArchitectureLongRoPE) {
		spec.RopeDimensionCount = spec.KeyLength
		spec.OriginalContextLength = optionalOr(
			values, prefix+"rope.scaling.original_context_length", gguf.ValueTypeUint32, spec.ContextLength,
		)
		spec.RopeAttentionFactor = optionalOr(
			values, prefix+"rope.scaling.attn_factor", gguf.ValueTypeFloat32, tensor.UnitScale,
		)
	}
	if spec.RopeScalingType == ropeScalingYaRN &&
		(validation.Hybrid == HybridValidationOptionalExperts ||
			validation.Hybrid == HybridValidationExtendedRotary && validation.MLA == MLAValidationNone) {
		spec.YaRNAttentionFactor = optionalOr(
			values, prefix+"rope.scaling.attn_factor", gguf.ValueTypeFloat32, tensor.UnitScale,
		)
	}
	spec.VocabularySize, _ = optional[uint32](values, prefix+"vocab_size", gguf.ValueTypeUint32)
	if tokens, ok := values["tokenizer.ggml.tokens"]; ok && spec.VocabularySize == tensor.FirstOffset {
		if tokens.Type != gguf.ValueTypeArray || tokens.ArrayType != gguf.ValueTypeString {
			return Spec{}, errors.New(`metadata "tokenizer.ggml.tokens" must be a string array`)
		}
		if tokens.Count() > int(math.MaxUint32) {
			return Spec{}, errors.New("tokenizer vocabulary exceeds uint32")
		}
		spec.VocabularySize = uint32(tokens.Count())
	}
	if validation.encoderOneOf(
		EncoderValidationTokenTypesMatchingHeads, EncoderValidationTokenTypesALiBi, EncoderValidationRotaryOptionalExperts,
		EncoderValidationRotary, EncoderValidationRotaryPeriodicExperts,
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

func (s Spec) IsPeriodicRescaleLayer(block uint32) bool {
	return s.Profile().Cadence.periodicRescale(s, block)
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

func (s Spec) LayerHasAttention(block uint32) bool {
	return s.LayerHeadCount(block) != tensor.FirstOffset
}

func (s Spec) LayerHasKVHeads(block uint32) bool {
	return s.LayerKVHeadCount(block) != tensor.FirstOffset
}

func (s Spec) LayerHasFeedForward(block uint32) bool {
	return s.LayerFeedForwardLength(block) != tensor.FirstOffset
}

func (s Spec) HasExperts() bool {
	return s.ExpertCount > tensor.FirstOffset
}

func (s Spec) HasSharedExperts() bool {
	return s.SharedExpertCount > tensor.FirstOffset
}

func (s Spec) LayerKeyLength(block uint32) uint32 {
	if s.KeyLengthSWA > tensor.FirstOffset && s.IsSlidingLayer(block) {
		return s.KeyLengthSWA
	}
	return s.KeyLength
}

func (s Spec) LayerValueLength(block uint32) uint32 {
	if s.ValueLengthSWA > tensor.FirstOffset && s.IsSlidingLayer(block) {
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
		return start - tensor.PairedExtent
	}
	return start - tensor.SingletonExtent
}

func (s Spec) LayerRopeDimensionCount(block uint32) uint32 {
	if s.KeyLengthSWA > tensor.FirstOffset && s.RopeDimensionSWA > tensor.FirstOffset && s.IsSlidingLayer(block) {
		return s.RopeDimensionSWA
	}
	if s.Profile().Rotary.FactorPairs && !s.IsSlidingLayer(block) {
		return s.RopeDimensionCount / tensor.PairedExtent
	}
	return s.RopeDimensionCount
}

func (s Spec) LayerExpertSwiGLUClamp(block uint32) float32 {
	return layerValue(s.LayerSwiGLUClamp, block, float32(tensor.FirstOffset))
}

func (s Spec) LayerSharedSwiGLUClampLimit(block uint32) float32 {
	return layerValue(s.LayerSharedSwiGLUClamp, block, float32(tensor.FirstOffset))
}

func (s Spec) UsesRoPE(block uint32) bool {
	usage := s.Profile().Rotary.Usage
	if usage == RotaryUsageSlidingMetadata && len(s.SlidingLayers) == tensor.FirstOffset {
		usage = RotaryUsageStandard
	}
	switch usage {
	case RotaryUsageSlidingMetadata, RotaryUsageDensePrefixOrSliding:
		return !s.RopeDisabled && block < s.BlockCount &&
			(usage == RotaryUsageDensePrefixOrSliding && block < s.LeadingDenseBlocks || s.IsSlidingLayer(block))
	case RotaryUsagePeriodicZeroBased:
		return !s.RopeDisabled && block < s.BlockCount &&
			(s.SlidingWindow == tensor.FirstOffset || s.NoRopeLayerStep == tensor.FirstOffset || block%s.NoRopeLayerStep != tensor.FirstOffset)
	case RotaryUsageSlidingOnly:
		return s.IsSlidingLayer(block)
	}
	blockCount := s.BlockCount
	if draft := s.Profile().DraftPlan(s.NextNPredictLayers); draft.AppendedBlocks {
		blockCount += draft.Heads
	}
	return !s.RopeDisabled &&
		(blockCount == tensor.FirstOffset || block < blockCount) &&
		(s.NoRopeLayerStep == tensor.FirstOffset || (block+tensor.SingletonExtent)%s.NoRopeLayerStep != tensor.FirstOffset)
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
	if s.RopeScalingType == ropeScalingLinear && !positiveFinite(s.RopeScalingFactor) {
		return errors.New("linear RoPE scaling factor must be positive")
	}
	return s.validateNumericPolicies()
}

func firstPositive(values []uint32) uint32 {
	for _, value := range values {
		if value > tensor.FirstOffset {
			return value
		}
	}
	return tensor.FirstOffset
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
		return slices.Repeat([]float32{scalar}, int(count)), nil
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
		return slices.Repeat([]uint32{scalar}, int(count)), nil
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
		if !valid || scalar < tensor.FirstOffset {
			return nil, fmt.Errorf("metadata %q has an invalid scalar value", key)
		}
		return slices.Repeat([]uint32{uint32(scalar)}, int(count)), nil
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
			if item < tensor.FirstOffset {
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
	if value.Type == gguf.ValueTypeBool {
		scalar, valid := value.Data.(bool)
		if !valid {
			return nil, fmt.Errorf("metadata %q has an invalid Go representation", key)
		}
		return slices.Repeat([]bool{scalar}, int(count)), nil
	}
	if value.Type == gguf.ValueTypeUint32 {
		scalar, valid := value.Data.(uint32)
		if !valid {
			return nil, fmt.Errorf("metadata %q has an invalid Go representation", key)
		}
		return slices.Repeat([]bool{scalar != tensor.FirstOffset}, int(count)), nil
	}
	if value.Type == gguf.ValueTypeInt32 {
		scalar, valid := value.Data.(int32)
		if !valid || scalar < tensor.FirstOffset {
			return nil, fmt.Errorf("metadata %q has an invalid scalar value", key)
		}
		return slices.Repeat([]bool{scalar != tensor.FirstOffset}, int(count)), nil
	}
	if value.Type != gguf.ValueTypeArray {
		return nil, fmt.Errorf("metadata %q must be a bool, uint32, or compatible array", key)
	}
	result := make([]bool, count)
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
			result[index] = item != tensor.FirstOffset
		}
	case gguf.ValueTypeInt32:
		items, valid := value.Data.([]int32)
		if !valid || len(items) != int(count) {
			return nil, fmt.Errorf("metadata %q has invalid layer values", key)
		}
		for index, item := range items {
			if item < tensor.FirstOffset {
				return nil, fmt.Errorf("metadata %q has a negative layer value", key)
			}
			result[index] = item != tensor.FirstOffset
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
