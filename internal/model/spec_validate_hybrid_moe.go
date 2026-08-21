package model

import (
	"errors"
	"fmt"

	"overgo/internal/tensor"
)

func (s Spec) validateHybridMetadata() error {
	hybrid := s.Profile().Validation.Hybrid
	alternatingStateSpace := hybrid == HybridValidationAlternatingGatedDelta ||
		hybrid == HybridValidationAlternatingGatedDeltaHybrid || hybrid == HybridValidationAlternatingGatedDeltaExperts
	if alternatingStateSpace {
		switch {
		case !validRotaryDimension(s.RopeDimensionCount, s.KeyLength):
			return errors.New("alternating state-space rotary dimension count is invalid")
		case s.SSMConvKernel == tensor.FirstOffset:
			return errors.New("alternating state-space convolution kernel is zero")
		case !validStateWidthGroupedSSM(s):
			return errors.New("alternating state-space dimensions are invalid")
		case s.FullAttentionInterval == tensor.FirstOffset:
			return errors.New("alternating state-space attention interval is zero")
		}
		if hybrid != HybridValidationAlternatingGatedDelta {
			var sectionPairs int64
			for _, section := range s.RopeSections {
				if section < tensor.FirstOffset {
					return errors.New("Qwen3.5 RoPE section count is negative")
				}
				sectionPairs += int64(section)
			}
			if sectionPairs == tensor.FirstOffset || sectionPairs > int64(s.RopeDimensionCount/rotaryPairAlignment) {
				return errors.New("Qwen3.5 RoPE sections exceed rotary pair count")
			}
		}
	}
	if (hybrid == HybridValidationAlternatingGatedDelta || hybrid == HybridValidationAlternatingGatedDeltaExperts) &&
		(!validExpertDimensions(s) || s.SharedExpertFF == tensor.FirstOffset || !positiveFinite(s.ExpertWeightsScale)) {
		return errors.New("Qwen3.5-MoE expert metadata is invalid")
	}
	if hybrid == HybridValidationSharedExperts &&
		(!validExpertDimensions(s) || !positiveFinite(s.ExpertWeightsScale)) {
		return errors.New("Qwen3-MoE expert metadata is invalid")
	}
	if hybrid == HybridValidationSparseSharedExperts &&
		(!validExpertDimensions(s) || s.ExpertChunkFeedForward == tensor.FirstOffset ||
			s.ExpertsPerGroup == tensor.FirstOffset || s.ExpertCount%s.ExpertsPerGroup != tensor.FirstOffset ||
			!positiveFinite(s.ExpertWeightsScale) || !finite(s.ExpertGroupScale)) {
		return errors.New("GroveMoE expert metadata is invalid")
	}
	if hybrid == HybridValidationSigmoidExperts {
		switch {
		case !validExpertDimensions(s) || !positiveFinite(s.ExpertWeightsScale):
			return errors.New("MiMo2 expert metadata is invalid")
		case !validRotaryDimension(s.RopeDimensionCount, s.KeyLength):
			return errors.New("MiMo2 rotary dimension count is invalid")
		case !validSlidingFrequency(s, true):
			return errors.New("MiMo2 sliding-attention metadata is invalid")
		case !finite(s.AttentionValueScale):
			return errors.New("MiMo2 attention value scale is invalid")
		}
		for block := uint32(tensor.FirstOffset); block < s.BlockCount; block++ {
			kvHeads := s.LayerKVHeadCount(block)
			if kvHeads == tensor.FirstOffset || s.HeadCount%kvHeads != tensor.FirstOffset {
				return errors.New("MiMo2 layer KV head count is invalid")
			}
		}
	}
	if hybrid == HybridValidationCompressedHyperDraft {
		layerCount := int(s.BlockCount + s.NextNPredictLayers)
		switch {
		case !validExpertDimensions(s) || !positiveFinite(s.ExpertWeightsScale):
			return errors.New("Step3.5 expert metadata is invalid")
		case !validExpertRouting(s):
			return errors.New("Step3.5 expert routing function is unsupported")
		case len(s.LayerHeadCounts) != layerCount || len(s.LayerKVHeadCounts) != layerCount:
			return errors.New("Step3.5 layer head metadata is invalid")
		case s.RopeDimensionCount == tensor.FirstOffset || s.RopeDimensionCount > s.KeyLength ||
			s.RopeDimensionCount%(rotaryPairAlignment*rotaryPairAlignment) != tensor.FirstOffset:
			return errors.New("Step3.5 rotary dimension count is invalid")
		case s.SlidingWindow == tensor.FirstOffset || len(s.SlidingLayers) != layerCount || !positiveFinite(s.RopeFrequencySWA):
			return errors.New("Step3.5 sliding-attention metadata is invalid")
		case len(s.LayerSwiGLUClamp) > tensor.FirstOffset && len(s.LayerSwiGLUClamp) != layerCount:
			return errors.New("Step3.5 expert clamp metadata is invalid")
		case len(s.LayerSharedSwiGLUClamp) > tensor.FirstOffset && len(s.LayerSharedSwiGLUClamp) != layerCount:
			return errors.New("Step3.5 shared-expert clamp metadata is invalid")
		}
		for block := uint32(tensor.FirstOffset); block < uint32(layerCount); block++ {
			heads := s.LayerHeadCount(block)
			kvHeads := s.LayerKVHeadCount(block)
			if heads == tensor.FirstOffset || kvHeads == tensor.FirstOffset || heads%kvHeads != tensor.FirstOffset {
				return errors.New("Step3.5 layer head count is invalid")
			}
			for _, limit := range []float32{
				s.LayerExpertSwiGLUClamp(block), s.LayerSharedSwiGLUClampLimit(block),
			} {
				if !nonNegativeFinite(limit) {
					return errors.New("Step3.5 SwiGLU clamp is invalid")
				}
			}
		}
	}
	if hybrid == HybridValidationNonCausalExperts &&
		(!validExpertDimensions(s) || !positiveFinite(s.ExpertWeightsScale)) {
		return errors.New("LLaDA-MoE expert metadata is invalid")
	}
	if hybrid == HybridValidationNormalizedSharedExperts &&
		(!validExpertDimensions(s) || s.SharedExpertFF == tensor.FirstOffset || !positiveFinite(s.ExpertWeightsScale)) {
		return errors.New("Qwen2-MoE expert metadata is invalid")
	}
	if hybrid == HybridValidationRoutedExperts &&
		(!validExpertDimensions(s) || !positiveFinite(s.ExpertWeightsScale)) {
		return errors.New("Arctic expert metadata is invalid")
	}
	if hybrid == HybridValidationProductSharedExperts {
		switch {
		case !validExpertDimensions(s) || !s.HasSharedExperts():
			return errors.New("BailingMoE expert metadata is invalid")
		case s.SharedExpertFF/s.SharedExpertCount != s.ExpertFeedForward:
			return errors.New("BailingMoE shared expert width overflows")
		case !positiveFinite(s.ExpertWeightsScale):
			return errors.New("BailingMoE expert weight scale is invalid")
		}
	}
	if hybrid == HybridValidationProductExperts {
		switch {
		case s.LeadingDenseBlocks >= s.BlockCount:
			return errors.New("DeepSeek leading dense block count leaves no MoE layers")
		case !validExpertDimensions(s) || !s.HasSharedExperts():
			return errors.New("DeepSeek expert metadata is invalid")
		case s.SharedExpertFF/s.SharedExpertCount != s.ExpertFeedForward:
			return errors.New("DeepSeek shared expert width overflows")
		case !positiveFinite(s.ExpertWeightsScale):
			return errors.New("DeepSeek expert weight scale is invalid")
		}
	}
	if (hybrid == HybridValidationScaledExperts || hybrid == HybridValidationScaledDense && s.HasExperts()) &&
		(!validExpertDimensions(s) || !positiveFinite(s.ExpertWeightsScale)) {
		return errors.New("GraniteMoE expert metadata is invalid")
	}
	if hybrid == HybridValidationModelFeedForwardExperts &&
		(!validExpertDimensions(s) || !positiveFinite(s.ExpertWeightsScale) ||
			!nonNegativeFinite(s.AttentionClamp)) {
		return errors.New("DBRX expert or attention metadata is invalid")
	}
	if hybrid == HybridValidationSharedExpertNorm {
		switch {
		case !validExpertDimensions(s):
			return errors.New("Grok expert metadata is invalid")
		case !positiveFinite(s.ExpertWeightsScale):
			return errors.New("Grok expert weight scale is invalid")
		case !validFullRotaryHead(s):
			return errors.New("Grok rotary/head dimensions are invalid")
		case !positiveFinite(s.AttentionScale) || !positiveFinite(s.AttentionSoftcap):
			return errors.New("Grok attention scaling metadata is invalid")
		case s.RopeScalingType == ropeScalingYaRN && !validYaRN(s):
			return errors.New("Grok YaRN metadata is invalid")
		}
	}
	if hybrid == HybridValidationRequiredExpertFeedForward {
		switch {
		case !validExpertDimensions(s):
			return errors.New("Mellum expert metadata is invalid")
		case !validFullRotaryHead(s):
			return errors.New("Mellum rotary/head dimensions are invalid")
		case s.SlidingWindow > tensor.FirstOffset &&
			(!positiveFinite(s.RopeFrequencySWA) ||
				(len(s.SlidingLayers) == tensor.FirstOffset && s.SlidingPattern <= tensor.SingletonExtent)):
			return errors.New("Mellum sliding-attention metadata is invalid")
		case s.RopeScalingType == ropeScalingYaRN && !validYaRN(s):
			return errors.New("Mellum YaRN metadata is invalid")
		}
	}
	if hybrid == HybridValidationSharedExpertProduct &&
		(!validExpertDimensions(s) || s.SharedExpertFF == tensor.FirstOffset ||
			!validFullRotaryHead(s)) {
		return errors.New("Hunyuan-MoE metadata is invalid")
	}
	if hybrid == HybridValidationMultiHeadDraft {
		switch {
		case !validExpertDimensions(s) || s.SharedExpertFF == tensor.FirstOffset:
			return errors.New("HY-V3 expert metadata is invalid")
		case !validExpertRouting(s):
			return errors.New("HY-V3 expert routing function is unsupported")
		case !positiveFinite(s.ExpertWeightsScale):
			return errors.New("HY-V3 expert weight scale is invalid")
		case !validFullRotaryHead(s):
			return errors.New("HY-V3 rotary/head dimensions are invalid")
		}
	}
	if hybrid == HybridValidationFullRotaryVision {
		switch {
		case s.LeadingDenseBlocks >= s.BlockCount:
			return errors.New("DeepSeek2-OCR leading dense block count leaves no MoE layers")
		case !validExpertDimensions(s) || !s.HasSharedExperts():
			return errors.New("DeepSeek2-OCR expert metadata is invalid")
		case !validExpertRouting(s):
			return errors.New("DeepSeek2-OCR expert routing function is unsupported")
		case !positiveFinite(s.ExpertWeightsScale):
			return errors.New("DeepSeek2-OCR expert weight scale is invalid")
		case s.RopeFrequencyBase != s.Profile().MetadataDefaults.RopeFrequencyBase ||
			s.RopeScalingType != "" ||
			s.AttentionScale != tensor.FirstOffset:
			return errors.New("DeepSeek2-OCR attention scaling is unsupported")
		case s.HeadCountKV != s.HeadCount || !validFullRotaryHead(s):
			return errors.New("DeepSeek2-OCR attention metadata is invalid")
		}
	}
	if hybrid == HybridValidationDualExpertProduct {
		switch {
		case !validExpertDimensions(s):
			return errors.New("SmallThinker expert metadata is invalid")
		case !validExpertRouting(s):
			return errors.New("SmallThinker expert routing function is unsupported")
		case !positiveFinite(s.ExpertWeightsScale):
			return errors.New("SmallThinker expert weight scale is invalid")
		case !validFullRotaryHead(s):
			return errors.New("SmallThinker rotary/head dimensions are invalid")
		case s.SlidingWindow > tensor.FirstOffset &&
			(s.SlidingPattern <= tensor.SingletonExtent || !positiveFinite(s.RopeFrequencySWA)):
			return errors.New("SmallThinker sliding-attention metadata is invalid")
		}
	}
	if hybrid == HybridValidationWeightedExpertProduct {
		switch {
		case s.LeadingDenseBlocks >= s.BlockCount:
			return errors.New("DOTS1 leading dense block count leaves no MoE layers")
		case !validExpertDimensions(s) || !s.HasSharedExperts():
			return errors.New("DOTS1 expert metadata is invalid")
		case s.SharedExpertFF/s.SharedExpertCount != s.ExpertFeedForward:
			return errors.New("DOTS1 shared expert width overflows")
		case !validExpertRouting(s):
			return errors.New("DOTS1 expert routing function is unsupported")
		case !positiveFinite(s.ExpertWeightsScale):
			return errors.New("DOTS1 expert weight scale is invalid")
		case s.RopeDimensionCount > tensor.FirstOffset && s.RopeDimensionCount != s.KeyLength:
			return errors.New("DOTS1 rotary dimension must equal key length")
		case s.HeadCountKV != s.HeadCount || s.KeyLength != s.ValueLength:
			return errors.New("DOTS1 requires full-head matching key/value attention")
		}
	}
	if hybrid == HybridValidationScaledSigmoidExperts {
		switch {
		case !validExpertDimensions(s):
			return errors.New("MiniMax-M2 expert metadata is invalid")
		case !validExpertRouting(s):
			return errors.New("MiniMax-M2 expert routing function is unsupported")
		case !positiveFinite(s.ExpertWeightsScale):
			return errors.New("MiniMax-M2 expert weight scale is invalid")
		case !validRotaryDimension(s.RopeDimensionCount, s.KeyLength) || s.KeyLength != s.ValueLength:
			return errors.New("MiniMax-M2 rotary/head dimensions are invalid")
		}
	}
	scaledResidual := hybrid == HybridValidationScaledDense || hybrid == HybridValidationScaledExperts ||
		hybrid == HybridValidationScaledStateSpace
	if scaledResidual &&
		s.RopeScalingType == ropeScalingLongRoPE &&
		!validLongRoPE(s) {
		return errors.New("Granite LongRoPE metadata is invalid")
	}
	if hybrid == HybridValidationScaledSharedExperts {
		switch {
		case s.LeadingDenseBlocks >= s.BlockCount:
			return errors.New("BailingMoE2 leading dense block count leaves no MoE layers")
		case !validExpertDimensions(s) || !s.HasSharedExperts():
			return errors.New("BailingMoE2 expert metadata is invalid")
		case !validExpertRouting(s):
			return errors.New("BailingMoE2 expert routing function is unsupported")
		case s.SharedExpertFF%s.SharedExpertCount != tensor.FirstOffset:
			return errors.New("BailingMoE2 shared expert width is invalid")
		case !positiveFinite(s.ExpertWeightsScale):
			return errors.New("BailingMoE2 expert weight scale is invalid")
		case !validRotaryDimension(s.RopeDimensionCount, s.KeyLength) || s.KeyLength != s.ValueLength:
			return errors.New("BailingMoE2 rotary/head dimensions are invalid")
		}
	}
	if hybrid == HybridValidationFullHeadExperts &&
		(s.HeadCountKV != s.HeadCount || !validMoESelection(s.ExpertUsedCount, s.ExpertCount) ||
			s.ExpertFeedForward == tensor.FirstOffset || !positiveFinite(s.ExpertWeightsScale)) {
		return errors.New("OLMoE expert or attention metadata is invalid")
	}
	if hybrid == HybridValidationOptionalExperts && s.HasExperts() &&
		(!validMoESelection(s.ExpertUsedCount, s.ExpertCount) ||
			s.ExpertFeedForward == tensor.FirstOffset || !positiveFinite(s.ExpertWeightsScale)) {
		return errors.New("Llama MoE expert metadata is invalid")
	}
	extendedRotary := hybrid == HybridValidationOptionalExperts || hybrid == HybridValidationExtendedRotary
	if s.RopeScalingType == ropeScalingLongRoPE && extendedRotary &&
		!validLongRoPE(s) {
		return fmt.Errorf("%s LongRoPE metadata is invalid", s.Architecture)
	}
	if s.RopeScalingType == ropeScalingYaRN && extendedRotary &&
		!validYaRN(s) {
		return fmt.Errorf("%s YaRN metadata is invalid", s.Architecture)
	}
	if hybrid == HybridValidationChunkedExperts {
		switch {
		case !validExpertDimensions(s) || s.SharedExpertFF == tensor.FirstOffset ||
			s.MoELayerStep == tensor.FirstOffset:
			return errors.New("Llama 4 expert metadata is invalid")
		case s.ExpertGatingFunc != expertGatingSigmoid || !positiveFinite(s.ExpertWeightsScale):
			return errors.New("Llama 4 expert routing metadata is invalid")
		case !validFullRotaryHead(s):
			return errors.New("Llama 4 rotary/head dimensions are invalid")
		case s.SlidingWindow > tensor.FirstOffset &&
			(s.SlidingPattern <= tensor.SingletonExtent || !positiveFinite(s.RopeFrequencySWA) ||
				s.AttentionTempFloor == tensor.FirstOffset || !positiveFinite(s.AttentionTempScale) ||
				!finite(s.AttentionTempOffset)):
			return errors.New("Llama 4 chunked-attention metadata is invalid")
		}
	}
	if hybrid == HybridValidationSelectedSoftmaxExperts &&
		(!validExpertDimensions(s) || s.ExpertGatingFunc != expertGatingSelectedSoftmax ||
			!positiveFinite(s.ExpertWeightsScale) ||
			s.SlidingWindow == tensor.FirstOffset || s.SlidingPattern <= tensor.SingletonExtent ||
			!validFullRotaryHead(s) || !positiveFinite(s.RopeFrequencySWA)) {
		return errors.New("GPT-OSS metadata is invalid")
	}
	if hybrid == HybridValidationBasicScaledExperts &&
		(!validExpertDimensions(s) || !positiveFinite(s.ExpertWeightsScale)) {
		return errors.New("PhiMoE expert metadata is invalid")
	}
	if hybrid == HybridValidationPerLayerYaRNExperts {
		if len(s.LayerHeadCounts) != int(s.BlockCount) ||
			len(s.LayerKVHeadCounts) != int(s.BlockCount) {
			return errors.New("Laguna per-layer attention head metadata is invalid")
		}
		for block := uint32(tensor.FirstOffset); block < s.BlockCount; block++ {
			heads := s.LayerHeadCount(block)
			kvHeads := s.LayerKVHeadCount(block)
			if heads == tensor.FirstOffset || kvHeads == tensor.FirstOffset || heads%kvHeads != tensor.FirstOffset {
				return fmt.Errorf("Laguna layer %d attention head metadata is invalid", block)
			}
		}
		switch {
		case s.LeadingDenseBlocks >= s.BlockCount:
			return errors.New("Laguna leading dense block count leaves no MoE layers")
		case !validMoESelection(s.ExpertUsedCount, s.ExpertCount) ||
			s.ExpertFeedForward == tensor.FirstOffset || s.SharedExpertFF == tensor.FirstOffset:
			return errors.New("Laguna expert metadata is invalid")
		case s.ExpertGatingFunc != expertGatingSigmoid:
			return errors.New("Laguna requires sigmoid expert routing")
		case !positiveFinite(s.ExpertWeightsScale):
			return errors.New("Laguna expert weight scale is invalid")
		case !validRotaryDimension(s.RopeDimensionCount, s.KeyLength):
			return errors.New("Laguna full-attention rotary dimension count is invalid")
		case s.RopeScalingType != ropeScalingYaRN:
			return errors.New("Laguna full-attention layers require YaRN RoPE")
		case s.KeyLength != s.ValueLength:
			return errors.New("Laguna requires matching attention key and value lengths")
		case !validYaRN(s):
			return errors.New("Laguna YaRN metadata is invalid")
		case s.SlidingWindow > tensor.FirstOffset &&
			(s.SlidingPattern <= tensor.SingletonExtent || !positiveFinite(s.RopeFrequencySWA) ||
				!validRotaryDimension(s.RopeDimensionSWA, s.KeyLength)):
			return errors.New("Laguna sliding-attention metadata is invalid")
		}
	}
	if hybrid == HybridValidationSlidingSigmoidExperts {
		switch {
		case s.LeadingDenseBlocks >= s.BlockCount:
			return errors.New("AFMoE leading dense block count leaves no MoE layers")
		case !validMoESelection(s.ExpertUsedCount, s.ExpertCount) ||
			s.ExpertFeedForward == tensor.FirstOffset:
			return errors.New("AFMoE expert metadata is invalid")
		case s.ExpertGatingFunc != expertGatingSigmoid:
			return errors.New("AFMoE requires sigmoid expert routing")
		case !positiveFinite(s.ExpertWeightsScale):
			return errors.New("AFMoE expert weight scale is invalid")
		case s.HasSharedExperts() && s.SharedExpertFF/s.SharedExpertCount != s.ExpertFeedForward:
			return errors.New("AFMoE shared expert width overflows")
		case !validRotaryDimension(s.RopeDimensionCount, s.KeyLength) || s.KeyLength != s.ValueLength:
			return errors.New("AFMoE rotary/head dimensions are invalid")
		case s.SlidingWindow > tensor.FirstOffset &&
			(s.SlidingPattern <= tensor.SingletonExtent || !positiveFinite(s.RopeFrequencySWA)):
			return errors.New("AFMoE sliding-attention metadata is invalid")
		}
	}
	if hybrid == HybridValidationSlidingSharedExperts {
		switch {
		case s.LeadingDenseBlocks >= s.BlockCount:
			return errors.New("EXAONE-MoE leading dense block count leaves no MoE layers")
		case !validExpertDimensions(s) || s.SharedExpertFF == tensor.FirstOffset:
			return errors.New("EXAONE-MoE expert metadata is invalid")
		case !validExpertRouting(s):
			return errors.New("EXAONE-MoE expert routing function is unsupported")
		case !positiveFinite(s.ExpertWeightsScale):
			return errors.New("EXAONE-MoE expert weight scale is invalid")
		case !validFullRotaryHead(s):
			return errors.New("EXAONE-MoE rotary/head dimensions are invalid")
		case !validSlidingFrequency(s, true):
			return errors.New("EXAONE-MoE sliding attention metadata is invalid")
		}
	}
	shortConvolution := hybrid == HybridValidationAlternatingShortConvolution || hybrid == HybridValidationAlternatingShortConvolutionExperts
	if shortConvolution {
		if s.ShortConvCacheLength < minimumConvKernelWidth || !validMixedLayerSchedule(s.RecurrentLayers, s.BlockCount) {
			return errors.New("alternating short-convolution metadata is invalid")
		}
		if hybrid == HybridValidationAlternatingShortConvolutionExperts {
			switch {
			case s.LeadingDenseBlocks >= s.BlockCount:
				return errors.New("LFM2-MoE leading dense block count leaves no MoE layers")
			case !validExpertDimensions(s):
				return errors.New("LFM2-MoE expert metadata is invalid")
			case !validExpertRouting(s):
				return errors.New("LFM2-MoE expert routing function is unsupported")
			case !positiveFinite(s.ExpertWeightsScale):
				return errors.New("LFM2-MoE expert weight scale is invalid")
			}
		}
	}
	return nil
}

func validLongRoPE(s Spec) bool {
	return validRotaryDimension(s.RopeDimensionCount, s.KeyLength) &&
		s.OriginalContextLength > tensor.FirstOffset && positiveFinite(s.RopeAttentionFactor)
}

func validYaRN(s Spec) bool {
	return positiveFinite(s.RopeScalingFactor) && s.OriginalContextLength > tensor.FirstOffset &&
		nonNegativeFinite(s.YaRNExtFactor) && positiveFinite(s.YaRNAttentionFactor) &&
		positiveFinite(s.YaRNBetaFast) && positiveFinite(s.YaRNBetaSlow)
}

func validSlidingFrequency(s Spec, requireSchedule bool) bool {
	if s.SlidingWindow == tensor.FirstOffset || !positiveFinite(s.RopeFrequencySWA) {
		return false
	}
	return !requireSchedule || len(s.SlidingLayers) > tensor.FirstOffset || s.SlidingPattern > tensor.FirstOffset
}
