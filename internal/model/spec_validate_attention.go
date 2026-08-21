package model

import (
	"errors"
	"fmt"

	"overgo/internal/tensor"
)

func (s Spec) validateAttentionMetadata() error {
	validation := s.Profile().Validation
	defaults := s.Profile().MetadataDefaults
	attention := validation.Attention
	if attention == AttentionValidationQKNormEpsilon && !positiveFinite(s.QKNormEpsilon) {
		return errors.New("Chameleon Q/K LayerNorm epsilon must be positive")
	}
	if validation.MultiAxisRoPE {
		var sectionPairs int32
		for _, section := range s.RopeSections {
			if section < tensor.FirstOffset {
				return fmt.Errorf("%s MRoPE section count is negative", s.Architecture)
			}
			sectionPairs += section
		}
		if !validRotaryDimension(s.RopeDimensionCount, s.KeyLength) ||
			s.RopeDimensionCount != s.KeyLength || s.KeyLength != s.ValueLength || sectionPairs == tensor.FirstOffset ||
			sectionPairs > int32(s.RopeDimensionCount/tensor.PairedExtent) {
			return fmt.Errorf("%s MRoPE metadata is invalid", s.Architecture)
		}
		if validation.BoundDeepstack && s.DeepstackLayerCount > s.BlockCount {
			return errors.New("Qwen3-VL deepstack layer count exceeds block count")
		}
	}
	if attention == AttentionValidationMatchingAttentionKVHeads && s.HeadCountKV != s.HeadCount {
		return errors.New("Jais2 requires matching attention and KV head counts")
	}
	if attention == AttentionValidationPerLayerAttentionAndFeedForward {
		if len(s.LayerHeadCounts) != int(s.BlockCount) ||
			len(s.LayerKVHeadCounts) != int(s.BlockCount) ||
			len(s.LayerFeedForward) != int(s.BlockCount) {
			return errors.New("OpenELM per-layer metadata is invalid")
		}
		for block := uint32(tensor.FirstOffset); block < s.BlockCount; block++ {
			heads := s.LayerHeadCount(block)
			kvHeads := s.LayerKVHeadCount(block)
			if heads == tensor.FirstOffset || kvHeads == tensor.FirstOffset || heads%kvHeads != tensor.FirstOffset || !s.LayerHasFeedForward(block) {
				return fmt.Errorf("OpenELM layer %d dimensions are invalid", block)
			}
		}
	}
	if attention == AttentionValidationPerLayerSlidingAttention {
		if len(s.LayerHeadCounts) != int(s.BlockCount) ||
			len(s.LayerKVHeadCounts) != int(s.BlockCount) ||
			len(s.LayerFeedForward) != int(s.BlockCount) {
			return errors.New("PLaMo 3 per-layer metadata is invalid")
		}
		for block := uint32(tensor.FirstOffset); block < s.BlockCount; block++ {
			heads := s.LayerHeadCount(block)
			kvHeads := s.LayerKVHeadCount(block)
			if heads == tensor.FirstOffset || kvHeads == tensor.FirstOffset || heads%kvHeads != tensor.FirstOffset || !s.LayerHasFeedForward(block) {
				return fmt.Errorf("PLaMo 3 layer %d dimensions are invalid", block)
			}
		}
		if s.RopeDimensionCount != s.KeyLength || s.KeyLength%tensor.PairedExtent != tensor.FirstOffset ||
			(s.SlidingWindow > tensor.FirstOffset && (!positiveFinite(s.RopeFrequencySWA) ||
				(len(s.SlidingLayers) == tensor.FirstOffset && s.SlidingPattern < tensor.PairedExtent))) {
			return errors.New("PLaMo 3 rotary or sliding-attention metadata is invalid")
		}
	}
	if attention == AttentionValidationSparseLayerAttention {
		if len(s.LayerHeadCounts) != int(s.BlockCount) ||
			len(s.LayerKVHeadCounts) != int(s.BlockCount) ||
			len(s.LayerFeedForward) != int(s.BlockCount) {
			return errors.New("Deci per-layer metadata is invalid")
		}
		var fullAttention bool
		for block := uint32(tensor.FirstOffset); block < s.BlockCount; block++ {
			heads := s.LayerHeadCount(block)
			kvHeads := s.LayerKVHeadCount(block)
			if heads == tensor.FirstOffset && kvHeads != tensor.FirstOffset || kvHeads > tensor.FirstOffset &&
				(heads == tensor.FirstOffset || heads%kvHeads != tensor.FirstOffset) {
				return fmt.Errorf("Deci layer %d attention head metadata is invalid", block)
			}
			fullAttention = fullAttention || kvHeads > tensor.FirstOffset
		}
		if !fullAttention {
			return errors.New("Deci requires at least one full-attention layer")
		}
		if s.RopeDimensionCount == tensor.FirstOffset || s.RopeDimensionCount > s.KeyLength ||
			s.RopeDimensionCount%tensor.PairedExtent != tensor.FirstOffset || s.KeyLength != s.ValueLength {
			return errors.New("Deci rotary/head dimensions are invalid")
		}
		if s.RopeScalingType == ropeScalingLongRoPE &&
			(s.OriginalContextLength == tensor.FirstOffset || !positiveFinite(s.RopeAttentionFactor)) {
			return errors.New("Deci LongRoPE metadata is invalid")
		}
	}
	if attention == AttentionValidationScaledSlidingAttention {
		switch {
		case !positiveFinite(s.RopeFrequencySWA):
			return errors.New("Gemma 3 sliding RoPE frequency base must be positive")
		case !positiveFinite(s.RopeScalingFactor):
			return errors.New("Gemma 3 RoPE scaling factor must be positive")
		case s.SlidingWindow > tensor.FirstOffset && s.SlidingPattern < tensor.PairedExtent:
			return errors.New("Gemma 3 sliding attention pattern must be at least 2")
		}
	}
	if attention == AttentionValidationSharedKVAlternatingState {
		switch {
		case s.BlockCount != validation.RequiredBlockCount &&
			s.BlockCount != validation.AlternateBlockCount:
			return fmt.Errorf(
				"shared-KV block count must be %d or %d",
				validation.RequiredBlockCount, validation.AlternateBlockCount,
			)
		case s.KVFromStart != defaults.SharedKVStartLayer || s.SharedKVLayers != s.BlockCount-s.KVFromStart:
			return errors.New("shared-KV boundary is invalid")
		case s.AltUpCount != defaults.AlternateStateCount || s.AltUpActive != defaults.AlternateStateActive ||
			s.LaurelRank != defaults.LowRankResidualWidth ||
			s.EmbeddingPerLayer != defaults.PerLayerEmbeddingWidth:
			return errors.New("alternate-state dimensions are invalid")
		case s.SparseLayerCount != defaults.SparseLayerCount ||
			s.SparsityStdMultiplier != defaults.SparsityStdMultiplier:
			return errors.New("shared-KV sparsity parameters are invalid")
		case s.KeyLength == tensor.FirstOffset || s.KeyLength != s.ValueLength || s.HeadCount == tensor.FirstOffset ||
			s.HeadCountKV == tensor.FirstOffset || s.HeadCount%s.HeadCountKV != tensor.FirstOffset:
			return errors.New("shared-KV attention dimensions are invalid")
		case s.RopeDimensionCount != s.KeyLength || s.KeyLength%tensor.PairedExtent != tensor.FirstOffset ||
			!positiveFinite(s.RopeFrequencySWA) || s.SlidingWindow == tensor.FirstOffset ||
			s.SlidingPattern != validation.SlidingPeriod:
			return errors.New("shared-KV rotary/sliding metadata is invalid")
		case !positiveFinite(s.FinalLogitSoftcap):
			return errors.New("Gemma 3n final logit softcap must be positive")
		}
	}
	if attention == AttentionValidationPerLayerDualRotaryAttention {
		switch {
		case s.KeyLength == tensor.FirstOffset || s.ValueLength == tensor.FirstOffset || s.KeyLength != s.ValueLength ||
			s.KeyLengthSWA == tensor.FirstOffset || s.ValueLengthSWA == tensor.FirstOffset || s.KeyLengthSWA != s.ValueLengthSWA:
			return errors.New("Gemma 4 attention head dimensions are invalid")
		case s.RopeDimensionCount == tensor.FirstOffset || s.RopeDimensionCount > s.KeyLength || s.RopeDimensionCount%tensor.PairedExtent != tensor.FirstOffset ||
			s.RopeDimensionSWA == tensor.FirstOffset || s.RopeDimensionSWA > s.KeyLengthSWA || s.RopeDimensionSWA%tensor.PairedExtent != tensor.FirstOffset:
			return errors.New("Gemma 4 rotary dimensions are invalid")
		case !positiveFinite(s.RopeFrequencySWA) || s.SlidingWindow == tensor.FirstOffset || len(s.SlidingLayers) != int(s.BlockCount):
			return errors.New("Gemma 4 sliding-attention metadata is invalid")
		case s.SharedKVLayers > tensor.FirstOffset && (s.SharedKVLayers >= s.BlockCount || s.BlockCount-s.SharedKVLayers < tensor.PairedExtent):
			return errors.New("Gemma 4 shared-KV layer count is invalid")
		case len(s.LayerFeedForward) != int(s.BlockCount) || len(s.LayerKVHeadCounts) != int(s.BlockCount):
			return errors.New("Gemma 4 layer metadata is invalid")
		case s.HasExperts() && (!validMoESelection(s.ExpertUsedCount, s.ExpertCount) || s.ExpertFeedForward == tensor.FirstOffset):
			return errors.New("Gemma 4 expert metadata is invalid")
		}
		for block := uint32(tensor.FirstOffset); block < s.BlockCount; block++ {
			if !s.LayerHasFeedForward(block) || !s.LayerHasKVHeads(block) ||
				s.HeadCount%s.LayerKVHeadCount(block) != tensor.FirstOffset {
				return errors.New("Gemma 4 per-layer dimensions are invalid")
			}
		}
	}
	if attention == AttentionValidationTargetHiddenDualRotaryAttention {
		switch {
		case s.TargetHiddenSize == tensor.FirstOffset || s.TargetHiddenSize == s.EmbeddingLength:
			return errors.New("Gemma 4 assistant target hidden size is invalid")
		case s.KeyLength == tensor.FirstOffset || s.ValueLength == tensor.FirstOffset || s.KeyLength != s.ValueLength ||
			s.KeyLengthSWA == tensor.FirstOffset || s.ValueLengthSWA == tensor.FirstOffset || s.KeyLengthSWA != s.ValueLengthSWA:
			return errors.New("Gemma 4 assistant attention head dimensions are invalid")
		case s.HeadCount == tensor.FirstOffset || s.HeadCountKV == tensor.FirstOffset || s.HeadCount%s.HeadCountKV != tensor.FirstOffset:
			return errors.New("Gemma 4 assistant attention head counts are invalid")
		case s.RopeDimensionCount == tensor.FirstOffset || s.RopeDimensionCount > s.KeyLength || s.RopeDimensionCount%tensor.PairedExtent != tensor.FirstOffset ||
			s.RopeDimensionSWA == tensor.FirstOffset || s.RopeDimensionSWA > s.KeyLengthSWA || s.RopeDimensionSWA%tensor.PairedExtent != tensor.FirstOffset:
			return errors.New("Gemma 4 assistant rotary dimensions are invalid")
		case !positiveFinite(s.RopeFrequencySWA) || s.SlidingWindow == tensor.FirstOffset || len(s.SlidingLayers) != int(s.BlockCount):
			return errors.New("Gemma 4 assistant sliding-attention metadata is invalid")
		}
	}
	if attention == AttentionValidationRequiredSlidingFrequency {
		switch {
		case !positiveFinite(s.RopeFrequencySWA):
			return errors.New("Gemma 2 sliding RoPE frequency base must be positive")
		case s.SlidingWindow > tensor.FirstOffset && s.SlidingPattern < tensor.PairedExtent:
			return errors.New("Gemma 2 sliding attention pattern must be at least 2")
		}
	}
	if attention == AttentionValidationOptionalSlidingFrequency {
		switch {
		case s.SlidingWindow > tensor.FirstOffset && !positiveFinite(s.RopeFrequencySWA):
			return errors.New("OLMo2 sliding RoPE frequency base must be positive")
		case s.SlidingWindow > tensor.FirstOffset && s.SlidingPattern < tensor.PairedExtent:
			return errors.New("OLMo2 sliding attention pattern must be at least 2")
		}
	}
	if attention == AttentionValidationRequiredSlidingRotary || attention == AttentionValidationRequiredSlidingRotaryExperts {
		switch {
		case !validRotaryDimension(s.RopeDimensionCount, s.KeyLength):
			return errors.New("Cohere2 rotary dimension count is invalid")
		case !positiveFinite(s.RopeFrequencySWA):
			return errors.New("Cohere2 sliding RoPE frequency base must be positive")
		case s.SlidingWindow == tensor.FirstOffset:
			return errors.New("Cohere2 sliding attention window is zero")
		case s.SlidingPattern < tensor.PairedExtent:
			return errors.New("Cohere2 sliding attention pattern must be at least 2")
		}
	}
	if attention == AttentionValidationRequiredSlidingRotaryExperts &&
		(s.LeadingDenseBlocks >= s.BlockCount || !validExpertDimensions(s) ||
			s.ExpertGatingFunc != expertGatingSigmoid || s.HasSharedExperts() && s.SharedExpertFF == tensor.FirstOffset) {
		return errors.New("Cohere2-MoE expert metadata is invalid")
	}
	if attention == AttentionValidationPeriodicExperts &&
		(!validExpertDimensions(s) || s.MoELayerStep == tensor.FirstOffset || s.LeadingDenseBlocks >= s.BlockCount) {
		return errors.New("ERNIE 4.5 MoE expert metadata is invalid")
	}
	if attention == AttentionValidationPartialRotaryRequired &&
		!validRotaryDimension(s.RopeDimensionCount, s.KeyLength) {
		return errors.New("StableLM rotary dimension count is invalid")
	}
	if attention == AttentionValidationPartialRotaryFixed &&
		!validRotaryDimension(s.RopeDimensionCount, s.KeyLength) {
		return errors.New("Phi-2 rotary dimension count is invalid")
	}
	if attention == AttentionValidationScaledPartialRotary &&
		(!validRotaryDimension(s.RopeDimensionCount, s.KeyLength) ||
			s.OriginalContextLength == tensor.FirstOffset || !positiveFinite(s.RopeAttentionFactor)) {
		return errors.New("Phi-3 RoPE metadata is invalid")
	}
	if attention == AttentionValidationFullScaledRotary &&
		(s.RopeDimensionCount != s.KeyLength || !validRotaryDimension(s.RopeDimensionCount, s.KeyLength) ||
			s.KeyLength != s.ValueLength || s.OriginalContextLength == tensor.FirstOffset ||
			!positiveFinite(s.RopeAttentionFactor)) {
		return errors.New("Pangu Embedded RoPE metadata is invalid")
	}
	if attention == AttentionValidationFullHeadSlidingRotary {
		switch {
		case s.HeadCountKV != s.HeadCount || s.KeyLength != s.ValueLength:
			return errors.New("ModernBERT requires full-head matching key/value attention")
		case !validRotaryDimension(s.RopeDimensionCount, s.KeyLength):
			return errors.New("ModernBERT rotary dimension count is invalid")
		case s.SlidingWindow > tensor.FirstOffset &&
			(s.SlidingPattern < tensor.PairedExtent || !positiveFinite(s.RopeFrequencySWA)):
			return errors.New("ModernBERT sliding attention metadata is invalid")
		}
	}
	if attention == AttentionValidationSlidingRotaryEmbeddingProjection {
		switch {
		case s.KeyLength != s.ValueLength:
			return errors.New("Gemma embedding requires matching key/value head widths")
		case !validRotaryDimension(s.RopeDimensionCount, s.KeyLength):
			return errors.New("Gemma embedding rotary dimension count is invalid")
		case s.SlidingWindow == tensor.FirstOffset || s.SlidingPattern < tensor.PairedExtent || !positiveFinite(s.RopeFrequencySWA):
			return errors.New("Gemma embedding sliding attention metadata is invalid")
		case s.Dense2FeatureIn > tensor.FirstOffset && s.Dense2FeatureIn != s.EmbeddingLength:
			return errors.New("Gemma embedding dense-2 input width must match embedding length")
		case s.Dense2FeatureOut > tensor.FirstOffset && s.Dense3FeatureIn > tensor.FirstOffset && s.Dense2FeatureOut != s.Dense3FeatureIn:
			return errors.New("embedding projection widths do not compose")
		case s.Dense3FeatureOut > tensor.FirstOffset && s.Dense3FeatureOut != s.EmbeddingLength:
			return errors.New("Gemma embedding dense-3 output width must match embedding length")
		}
	}
	if attention == AttentionValidationFullRotary &&
		(s.KeyLength != s.ValueLength || s.RopeDimensionCount != s.KeyLength ||
			!validRotaryDimension(s.RopeDimensionCount, s.KeyLength)) {
		return errors.New("Talkie attention metadata is invalid")
	}
	if attention == AttentionValidationFullScaledRotaryXIELU {
		if s.RopeDimensionCount != s.KeyLength || !validRotaryDimension(s.RopeDimensionCount, s.KeyLength) ||
			s.OriginalContextLength == tensor.FirstOffset || !positiveFinite(s.RopeAttentionFactor) {
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
				if !finite(value) {
					return fmt.Errorf("Apertus xIELU %s values must be finite", name)
				}
			}
		}
	}
	if attention == AttentionValidationOptionalRotaryBase && s.RopeDimensionCount > tensor.FirstOffset &&
		!validRotaryDimension(s.RopeDimensionCount, s.KeyLength) {
		return errors.New("GPT-NeoX rotary dimension count is invalid")
	}
	if attention == AttentionValidationHalvedFeedForward &&
		(s.HeadCountKV != s.HeadCount || !validRotaryDimension(s.RopeDimensionCount, s.KeyLength)) {
		return errors.New("Qwen attention metadata is invalid")
	}
	if attention == AttentionValidationMultiAxisAttention &&
		(!validRotaryDimension(s.RopeDimensionCount, s.KeyLength) || s.KeyLength != s.ValueLength) {
		return errors.New("ChatGLM attention metadata is invalid")
	}
	if attention == AttentionValidationVisualExpertAttention &&
		(s.HeadCountKV != s.HeadCount || s.KeyLength != s.ValueLength ||
			uint64(s.KeyLength)*uint64(s.HeadCount) != uint64(s.EmbeddingLength) ||
			s.RopeDimensionCount != s.KeyLength) {
		return errors.New("CogVLM attention metadata is invalid")
	}
	if attention == AttentionValidationLayerwiseQKNorm &&
		(s.RopeDimensionCount != s.KeyLength || s.KeyLength != s.ValueLength ||
			!validRotaryDimension(s.RopeDimensionCount, s.KeyLength) || !positiveFinite(s.RopeFrequencyBase)) {
		return fmt.Errorf("%s attention metadata is invalid", s.Architecture)
	}
	if attention == AttentionValidationLayerwiseQKNorm {
		for _, section := range s.RopeSections {
			if section < tensor.FirstOffset {
				return errors.New("Hunyuan MRoPE section count is negative")
			}
		}
	}
	if (attention == AttentionValidationOptionalRopeSections || attention == AttentionValidationOptionalRopeSectionsExperts) &&
		!validRotaryDimension(s.RopeDimensionCount, s.KeyLength) {
		return errors.New("GLM4 rotary dimension count is invalid")
	}
	if attention == AttentionValidationOptionalRopeSections || attention == AttentionValidationOptionalRopeSectionsExperts {
		for _, section := range s.RopeSections {
			if section < tensor.FirstOffset {
				return errors.New("GLM4 MRoPE section count is negative")
			}
		}
	}
	if attention == AttentionValidationOptionalRopeSectionsExperts &&
		(s.LeadingDenseBlocks >= s.BlockCount || !validMoESelection(s.ExpertUsedCount, s.ExpertCount) ||
			s.ExpertFeedForward == tensor.FirstOffset ||
			s.SharedExpertCount == tensor.FirstOffset || s.SharedExpertFF == tensor.FirstOffset ||
			(s.ExpertGatingFunc != expertGatingSoftmax && s.ExpertGatingFunc != expertGatingSigmoid) ||
			!positiveFinite(s.ExpertWeightsScale)) {
		return errors.New("GLM4-MoE expert metadata is invalid")
	}
	if attention == AttentionValidationSharedKVAttention {
		switch {
		case s.RopeDimensionCount != s.KeyLength || !validRotaryDimension(s.RopeDimensionCount, s.KeyLength):
			return errors.New("EXAONE 4 rotary dimension count must equal the key length")
		case s.SlidingWindow > tensor.FirstOffset && s.SlidingPattern < tensor.PairedExtent:
			return errors.New("EXAONE 4 sliding attention pattern must be at least 2")
		case s.SlidingWindow > tensor.FirstOffset && !positiveFinite(s.RopeFrequencySWA):
			return errors.New("EXAONE 4 sliding RoPE frequency base must be positive")
		}
	}
	if attention == AttentionValidationOptionalRotaryBaseGQA && s.RopeDimensionCount > tensor.FirstOffset &&
		s.RopeDimensionCount != s.KeyLength {
		return errors.New("Falcon rotary dimension count must equal the key length")
	}
	if attention == AttentionValidationOptionalExperts && s.HasExperts() &&
		(!validExpertDimensions(s) || !positiveFinite(s.ExpertWeightsScale)) {
		return errors.New("Refact expert metadata is invalid")
	}
	return nil
}
