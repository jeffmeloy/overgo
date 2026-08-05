package model

import (
	"errors"
	"fmt"
	"math"
)

func (s Spec) validateHybridMoEFamilies() error {
	if s.Architecture == "qwen3next" || s.Architecture == "qwen35" || s.Architecture == "qwen35moe" {
		switch {
		case s.RopeDimensionCount == 0 || s.RopeDimensionCount > s.KeyLength ||
			s.RopeDimensionCount%2 != 0:
			return errors.New("Qwen hybrid rotary dimension count is invalid")
		case s.SSMConvKernel == 0:
			return errors.New("Qwen hybrid SSM convolution kernel is zero")
		case s.SSMInnerSize == 0:
			return errors.New("Qwen hybrid SSM inner size is zero")
		case s.SSMStateSize == 0:
			return errors.New("Qwen hybrid SSM state size is zero")
		case s.SSMTimeStepRank == 0 || s.SSMInnerSize%s.SSMTimeStepRank != 0:
			return errors.New("Qwen hybrid SSM inner size is not divisible by time-step rank")
		case s.SSMInnerSize/s.SSMTimeStepRank != s.SSMStateSize:
			return errors.New("Qwen hybrid SSM value-head width differs from state size")
		case s.SSMGroupCount == 0 || s.SSMTimeStepRank%s.SSMGroupCount != 0:
			return errors.New("Qwen hybrid SSM value heads are not divisible by key groups")
		case s.FullAttentionInterval == 0:
			return errors.New("Qwen hybrid full-attention interval is zero")
		}
		if s.Architecture != "qwen3next" {
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
	}
	if (s.Architecture == "qwen3next" || s.Architecture == "qwen35moe") &&
		(!validExpertDimensions(s) || s.SharedExpertFF == 0 ||
			s.ExpertWeightsScale == 0 || !finite(s.ExpertWeightsScale)) {
		return errors.New("Qwen3.5-MoE expert metadata is invalid")
	}
	if (s.Architecture == "qwen3moe" || s.Architecture == "qwen3vlmoe" || s.Architecture == "rnd1") &&
		(!validExpertDimensions(s) || s.ExpertWeightsScale == 0 || !finite(s.ExpertWeightsScale)) {
		return errors.New("Qwen3-MoE expert metadata is invalid")
	}
	if s.Architecture == "grovemoe" &&
		(!validExpertDimensions(s) || s.ExpertChunkFeedForward == 0 ||
			s.ExpertsPerGroup == 0 || s.ExpertCount%s.ExpertsPerGroup != 0 ||
			s.ExpertWeightsScale == 0 || !finite(s.ExpertWeightsScale) || !finite(s.ExpertGroupScale)) {
		return errors.New("GroveMoE expert metadata is invalid")
	}
	if s.Architecture == "mimo2" {
		switch {
		case !validExpertDimensions(s) || s.ExpertWeightsScale == 0 || !finite(s.ExpertWeightsScale):
			return errors.New("MiMo2 expert metadata is invalid")
		case s.RopeDimensionCount == 0 || s.RopeDimensionCount > s.KeyLength || s.RopeDimensionCount%2 != 0:
			return errors.New("MiMo2 rotary dimension count is invalid")
		case s.SlidingWindow == 0 || (len(s.SlidingLayers) == 0 && s.SlidingPattern == 0) || s.RopeFrequencySWA <= 0:
			return errors.New("MiMo2 sliding-attention metadata is invalid")
		case !finite(s.AttentionValueScale):
			return errors.New("MiMo2 attention value scale is invalid")
		}
		for block := uint32(0); block < s.BlockCount; block++ {
			kvHeads := s.LayerKVHeadCount(block)
			if kvHeads == 0 || s.HeadCount%kvHeads != 0 {
				return errors.New("MiMo2 layer KV head count is invalid")
			}
		}
	}
	if s.Architecture == "step35" {
		layerCount := int(s.BlockCount + s.NextNPredictLayers)
		switch {
		case !validExpertDimensions(s) || !positiveFinite(s.ExpertWeightsScale):
			return errors.New("Step3.5 expert metadata is invalid")
		case s.ExpertGatingFunc != expertGatingSoftmax && s.ExpertGatingFunc != expertGatingSigmoid:
			return errors.New("Step3.5 expert routing function is unsupported")
		case len(s.LayerHeadCounts) != layerCount || len(s.LayerKVHeadCounts) != layerCount:
			return errors.New("Step3.5 layer head metadata is invalid")
		case s.RopeDimensionCount == 0 || s.RopeDimensionCount > s.KeyLength || s.RopeDimensionCount%4 != 0:
			return errors.New("Step3.5 rotary dimension count is invalid")
		case s.SlidingWindow == 0 || len(s.SlidingLayers) != layerCount || s.RopeFrequencySWA <= 0:
			return errors.New("Step3.5 sliding-attention metadata is invalid")
		case len(s.LayerSwiGLUClamp) != 0 && len(s.LayerSwiGLUClamp) != layerCount:
			return errors.New("Step3.5 expert clamp metadata is invalid")
		case len(s.LayerSharedSwiGLUClamp) != 0 && len(s.LayerSharedSwiGLUClamp) != layerCount:
			return errors.New("Step3.5 shared-expert clamp metadata is invalid")
		}
		for block := uint32(0); block < uint32(layerCount); block++ {
			heads := s.LayerHeadCount(block)
			kvHeads := s.LayerKVHeadCount(block)
			if heads == 0 || kvHeads == 0 || heads%kvHeads != 0 {
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
	if s.Architecture == "llada-moe" &&
		(!validExpertDimensions(s) || !positiveFinite(s.ExpertWeightsScale)) {
		return errors.New("LLaDA-MoE expert metadata is invalid")
	}
	if s.Architecture == "qwen2moe" &&
		(!validExpertDimensions(s) || s.SharedExpertFF == 0 || !positiveFinite(s.ExpertWeightsScale)) {
		return errors.New("Qwen2-MoE expert metadata is invalid")
	}
	if s.Architecture == "arctic" &&
		(!validExpertDimensions(s) || !positiveFinite(s.ExpertWeightsScale)) {
		return errors.New("Arctic expert metadata is invalid")
	}
	if s.Architecture == "bailingmoe" {
		switch {
		case s.ExpertCount == 0 || s.ExpertUsedCount == 0 || s.ExpertUsedCount > s.ExpertCount ||
			exceedsMoETopK(s.ExpertUsedCount) || s.ExpertFeedForward == 0 || s.SharedExpertCount == 0 ||
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
			exceedsMoETopK(s.ExpertUsedCount) || s.ExpertFeedForward == 0 || s.SharedExpertCount == 0 ||
			s.SharedExpertFF == 0:
			return errors.New("DeepSeek expert metadata is invalid")
		case s.SharedExpertFF/s.SharedExpertCount != s.ExpertFeedForward:
			return errors.New("DeepSeek shared expert width overflows")
		case s.ExpertWeightsScale <= 0 || math.IsNaN(float64(s.ExpertWeightsScale)) ||
			math.IsInf(float64(s.ExpertWeightsScale), 0):
			return errors.New("DeepSeek expert weight scale is invalid")
		}
	}
	if (s.Architecture == "granitemoe" || s.Architecture == "granite" && s.ExpertCount > 0) &&
		(s.ExpertCount == 0 || s.ExpertUsedCount == 0 || s.ExpertUsedCount > s.ExpertCount ||
			exceedsMoETopK(s.ExpertUsedCount) || s.ExpertFeedForward == 0 || s.ExpertWeightsScale <= 0 ||
			math.IsNaN(float64(s.ExpertWeightsScale)) || math.IsInf(float64(s.ExpertWeightsScale), 0)) {
		return errors.New("GraniteMoE expert metadata is invalid")
	}
	if s.Architecture == "dbrx" &&
		(s.ExpertCount == 0 || s.ExpertUsedCount == 0 || s.ExpertUsedCount > s.ExpertCount ||
			exceedsMoETopK(s.ExpertUsedCount) || s.ExpertFeedForward == 0 || s.ExpertWeightsScale <= 0 ||
			s.AttentionClamp < 0 || math.IsNaN(float64(s.AttentionClamp)) ||
			math.IsInf(float64(s.AttentionClamp), 0) || math.IsNaN(float64(s.ExpertWeightsScale)) ||
			math.IsInf(float64(s.ExpertWeightsScale), 0)) {
		return errors.New("DBRX expert or attention metadata is invalid")
	}
	if s.Architecture == "grok" {
		switch {
		case s.ExpertCount == 0 || s.ExpertUsedCount == 0 || s.ExpertUsedCount > s.ExpertCount ||
			exceedsMoETopK(s.ExpertUsedCount) || s.ExpertFeedForward == 0:
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
			exceedsMoETopK(s.ExpertUsedCount) || s.ExpertFeedForward == 0:
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
			exceedsMoETopK(s.ExpertUsedCount) || s.ExpertFeedForward == 0 || s.SharedExpertFF == 0 ||
			s.RopeDimensionCount != s.KeyLength || s.KeyLength != s.ValueLength ||
			s.RopeDimensionCount%2 != 0) {
		return errors.New("Hunyuan-MoE metadata is invalid")
	}
	if s.Architecture == "hy_v3" {
		switch {
		case s.ExpertCount == 0 || s.ExpertUsedCount == 0 || s.ExpertUsedCount > s.ExpertCount ||
			exceedsMoETopK(s.ExpertUsedCount) || s.ExpertFeedForward == 0 || s.SharedExpertFF == 0:
			return errors.New("HY-V3 expert metadata is invalid")
		case s.ExpertGatingFunc != expertGatingSoftmax && s.ExpertGatingFunc != expertGatingSigmoid:
			return errors.New("HY-V3 expert routing function is unsupported")
		case s.ExpertWeightsScale <= 0 || math.IsNaN(float64(s.ExpertWeightsScale)) ||
			math.IsInf(float64(s.ExpertWeightsScale), 0):
			return errors.New("HY-V3 expert weight scale is invalid")
		case s.RopeDimensionCount != s.KeyLength || s.KeyLength != s.ValueLength ||
			s.RopeDimensionCount%2 != 0:
			return errors.New("HY-V3 rotary/head dimensions are invalid")
		}
	}
	if s.Architecture == "deepseek2-ocr" {
		switch {
		case s.LeadingDenseBlocks >= s.BlockCount:
			return errors.New("DeepSeek2-OCR leading dense block count leaves no MoE layers")
		case s.ExpertCount == 0 || s.ExpertUsedCount == 0 || s.ExpertUsedCount > s.ExpertCount ||
			exceedsMoETopK(s.ExpertUsedCount) || s.ExpertFeedForward == 0 || s.SharedExpertCount == 0 ||
			s.SharedExpertFF == 0:
			return errors.New("DeepSeek2-OCR expert metadata is invalid")
		case s.ExpertGatingFunc != expertGatingSoftmax && s.ExpertGatingFunc != expertGatingSigmoid:
			return errors.New("DeepSeek2-OCR expert routing function is unsupported")
		case s.ExpertWeightsScale <= 0 || math.IsNaN(float64(s.ExpertWeightsScale)) ||
			math.IsInf(float64(s.ExpertWeightsScale), 0):
			return errors.New("DeepSeek2-OCR expert weight scale is invalid")
		case math.Abs(float64(s.RopeFrequencyBase-10000)) >= 1e-4 || s.RopeScalingType != "" ||
			s.AttentionScale != 0:
			return errors.New("DeepSeek2-OCR attention scaling is unsupported")
		case s.HeadCountKV != s.HeadCount || s.RopeDimensionCount != s.KeyLength ||
			s.KeyLength != s.ValueLength || s.RopeDimensionCount%2 != 0:
			return errors.New("DeepSeek2-OCR attention metadata is invalid")
		}
	}
	if s.Architecture == "smallthinker" {
		switch {
		case s.ExpertCount == 0 || s.ExpertUsedCount == 0 || s.ExpertUsedCount > s.ExpertCount ||
			exceedsMoETopK(s.ExpertUsedCount) || s.ExpertFeedForward == 0:
			return errors.New("SmallThinker expert metadata is invalid")
		case s.ExpertGatingFunc != expertGatingSoftmax && s.ExpertGatingFunc != expertGatingSigmoid:
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
			exceedsMoETopK(s.ExpertUsedCount) || s.ExpertFeedForward == 0 || s.SharedExpertCount == 0 ||
			s.SharedExpertFF == 0:
			return errors.New("DOTS1 expert metadata is invalid")
		case s.SharedExpertFF/s.SharedExpertCount != s.ExpertFeedForward:
			return errors.New("DOTS1 shared expert width overflows")
		case s.ExpertGatingFunc != expertGatingSoftmax && s.ExpertGatingFunc != expertGatingSigmoid:
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
			exceedsMoETopK(s.ExpertUsedCount) || s.ExpertFeedForward == 0:
			return errors.New("MiniMax-M2 expert metadata is invalid")
		case s.ExpertGatingFunc != expertGatingSoftmax && s.ExpertGatingFunc != expertGatingSigmoid:
			return errors.New("MiniMax-M2 expert routing function is unsupported")
		case s.ExpertWeightsScale <= 0 || math.IsNaN(float64(s.ExpertWeightsScale)) ||
			math.IsInf(float64(s.ExpertWeightsScale), 0):
			return errors.New("MiniMax-M2 expert weight scale is invalid")
		case s.RopeDimensionCount == 0 || s.RopeDimensionCount > s.KeyLength ||
			s.RopeDimensionCount%2 != 0 || s.KeyLength != s.ValueLength:
			return errors.New("MiniMax-M2 rotary/head dimensions are invalid")
		}
	}
	if (s.Architecture == "granite" || s.Architecture == "granitemoe" || s.Architecture == "granitehybrid") &&
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
			exceedsMoETopK(s.ExpertUsedCount) || s.ExpertFeedForward == 0 || s.SharedExpertCount == 0 ||
			s.SharedExpertFF == 0:
			return errors.New("BailingMoE2 expert metadata is invalid")
		case s.ExpertGatingFunc != expertGatingSoftmax && s.ExpertGatingFunc != expertGatingSigmoid:
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
		(s.HeadCountKV != s.HeadCount || !validMoESelection(s.ExpertUsedCount, s.ExpertCount) ||
			s.ExpertFeedForward == 0 || s.ExpertWeightsScale <= 0 ||
			math.IsNaN(float64(s.ExpertWeightsScale)) || math.IsInf(float64(s.ExpertWeightsScale), 0)) {
		return errors.New("OLMoE expert or attention metadata is invalid")
	}
	if (s.Architecture == "llama" || s.Architecture == "llama-embed") && s.ExpertCount > 0 &&
		(!validMoESelection(s.ExpertUsedCount, s.ExpertCount) ||
			s.ExpertFeedForward == 0 || s.ExpertWeightsScale <= 0 ||
			math.IsNaN(float64(s.ExpertWeightsScale)) || math.IsInf(float64(s.ExpertWeightsScale), 0)) {
		return errors.New("Llama MoE expert metadata is invalid")
	}
	if s.RopeScalingType == "longrope" &&
		(s.Architecture == "llama" || s.Architecture == "llama-embed" ||
			s.Architecture == "minicpm" || s.Architecture == "mistral3") &&
		(s.RopeDimensionCount == 0 || s.RopeDimensionCount > s.KeyLength ||
			s.RopeDimensionCount%2 != 0 || s.OriginalContextLength == 0 ||
			s.RopeAttentionFactor <= 0 || math.IsNaN(float64(s.RopeAttentionFactor)) ||
			math.IsInf(float64(s.RopeAttentionFactor), 0)) {
		return fmt.Errorf("%s LongRoPE metadata is invalid", s.Architecture)
	}
	if s.RopeScalingType == "yarn" &&
		(s.Architecture == "llama" || s.Architecture == "llama-embed" ||
			s.Architecture == "minicpm" || s.Architecture == "mistral3") &&
		(s.RopeScalingFactor <= 0 || s.OriginalContextLength == 0 ||
			s.YaRNExtFactor < 0 || s.YaRNAttentionFactor <= 0 ||
			s.YaRNBetaFast <= 0 || s.YaRNBetaSlow <= 0 ||
			math.IsNaN(float64(s.RopeScalingFactor)) || math.IsInf(float64(s.RopeScalingFactor), 0) ||
			math.IsNaN(float64(s.YaRNExtFactor)) || math.IsInf(float64(s.YaRNExtFactor), 0) ||
			math.IsNaN(float64(s.YaRNAttentionFactor)) || math.IsInf(float64(s.YaRNAttentionFactor), 0) ||
			math.IsNaN(float64(s.YaRNBetaFast)) || math.IsInf(float64(s.YaRNBetaFast), 0) ||
			math.IsNaN(float64(s.YaRNBetaSlow)) || math.IsInf(float64(s.YaRNBetaSlow), 0)) {
		return fmt.Errorf("%s YaRN metadata is invalid", s.Architecture)
	}
	if s.Architecture == "llama4" {
		switch {
		case s.ExpertCount == 0 || s.ExpertUsedCount == 0 || s.ExpertUsedCount > s.ExpertCount ||
			exceedsMoETopK(s.ExpertUsedCount) || s.ExpertFeedForward == 0 || s.SharedExpertFF == 0 || s.MoELayerStep == 0:
			return errors.New("Llama 4 expert metadata is invalid")
		case s.ExpertGatingFunc != expertGatingSigmoid || s.ExpertWeightsScale <= 0 ||
			math.IsNaN(float64(s.ExpertWeightsScale)) || math.IsInf(float64(s.ExpertWeightsScale), 0):
			return errors.New("Llama 4 expert routing metadata is invalid")
		case s.RopeDimensionCount != s.KeyLength || s.KeyLength != s.ValueLength || s.RopeDimensionCount%2 != 0:
			return errors.New("Llama 4 rotary/head dimensions are invalid")
		case s.SlidingWindow > 0 && (s.SlidingPattern < 2 || s.RopeFrequencySWA <= 0 ||
			s.AttentionTempFloor == 0 || s.AttentionTempScale <= 0 ||
			math.IsNaN(float64(s.AttentionTempScale)) || math.IsInf(float64(s.AttentionTempScale), 0) ||
			math.IsNaN(float64(s.AttentionTempOffset)) || math.IsInf(float64(s.AttentionTempOffset), 0)):
			return errors.New("Llama 4 chunked-attention metadata is invalid")
		}
	}
	if s.Architecture == "gpt-oss" &&
		(s.ExpertCount == 0 || s.ExpertUsedCount == 0 || s.ExpertUsedCount > s.ExpertCount ||
			exceedsMoETopK(s.ExpertUsedCount) || s.ExpertFeedForward == 0 || s.ExpertGatingFunc != expertGatingSelectedSoftmax ||
			s.ExpertWeightsScale <= 0 || math.IsNaN(float64(s.ExpertWeightsScale)) || math.IsInf(float64(s.ExpertWeightsScale), 0) ||
			s.SlidingWindow == 0 || s.SlidingPattern < 2 || s.RopeDimensionCount != s.KeyLength ||
			s.KeyLength != s.ValueLength || s.RopeDimensionCount%2 != 0 || s.RopeFrequencySWA <= 0) {
		return errors.New("GPT-OSS metadata is invalid")
	}
	if s.Architecture == "phimoe" &&
		(s.ExpertCount == 0 || s.ExpertUsedCount == 0 || s.ExpertUsedCount > s.ExpertCount ||
			exceedsMoETopK(s.ExpertUsedCount) || s.ExpertFeedForward == 0 || s.ExpertWeightsScale <= 0 ||
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
		case !validMoESelection(s.ExpertUsedCount, s.ExpertCount) ||
			s.ExpertFeedForward == 0 || s.SharedExpertFF == 0:
			return errors.New("Laguna expert metadata is invalid")
		case s.ExpertGatingFunc != expertGatingSigmoid:
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
		case !validMoESelection(s.ExpertUsedCount, s.ExpertCount) ||
			s.ExpertFeedForward == 0:
			return errors.New("AFMoE expert metadata is invalid")
		case s.ExpertGatingFunc != expertGatingSigmoid:
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
			exceedsMoETopK(s.ExpertUsedCount) || s.ExpertFeedForward == 0 || s.SharedExpertFF == 0:
			return errors.New("EXAONE-MoE expert metadata is invalid")
		case s.ExpertGatingFunc != expertGatingSoftmax && s.ExpertGatingFunc != expertGatingSigmoid:
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
				exceedsMoETopK(s.ExpertUsedCount) || s.ExpertFeedForward == 0:
				return errors.New("LFM2-MoE expert metadata is invalid")
			case s.ExpertGatingFunc != expertGatingSoftmax && s.ExpertGatingFunc != expertGatingSigmoid:
				return errors.New("LFM2-MoE expert routing function is unsupported")
			case s.ExpertWeightsScale <= 0 || math.IsNaN(float64(s.ExpertWeightsScale)) ||
				math.IsInf(float64(s.ExpertWeightsScale), 0):
				return errors.New("LFM2-MoE expert weight scale is invalid")
			}
		}
	}
	return nil
}
