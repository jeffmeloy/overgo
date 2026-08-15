package model

import (
	"errors"
	"fmt"
	"math"
)

func (s Spec) validateAttentionMetadata() error {
	validation := s.Profile().Validation
	defaults := s.Profile().MetadataDefaults
	attention := validation.Attention
	if attention == AttentionValidationChameleon && s.QKNormEpsilon <= 0 {
		return errors.New("Chameleon Q/K LayerNorm epsilon must be positive")
	}
	if validation.MultiAxisRoPE {
		var sectionPairs int32
		for _, section := range s.RopeSections {
			if section < 0 {
				return fmt.Errorf("%s MRoPE section count is negative", s.Architecture)
			}
			sectionPairs += section
		}
		if !validRotaryDimension(s.RopeDimensionCount, s.KeyLength, 2) ||
			s.RopeDimensionCount != s.KeyLength || s.KeyLength != s.ValueLength || sectionPairs == 0 ||
			sectionPairs > int32(s.RopeDimensionCount/2) {
			return fmt.Errorf("%s MRoPE metadata is invalid", s.Architecture)
		}
		if validation.BoundDeepstack && s.DeepstackLayerCount > s.BlockCount {
			return errors.New("Qwen3-VL deepstack layer count exceeds block count")
		}
	}
	if attention == AttentionValidationJais2 && s.HeadCountKV != s.HeadCount {
		return errors.New("Jais2 requires matching attention and KV head counts")
	}
	if attention == AttentionValidationOpenELM {
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
	if attention == AttentionValidationPLaMo3 {
		if len(s.LayerHeadCounts) != int(s.BlockCount) ||
			len(s.LayerKVHeadCounts) != int(s.BlockCount) ||
			len(s.LayerFeedForward) != int(s.BlockCount) {
			return errors.New("PLaMo 3 per-layer metadata is invalid")
		}
		for block := uint32(0); block < s.BlockCount; block++ {
			heads := s.LayerHeadCount(block)
			kvHeads := s.LayerKVHeadCount(block)
			if heads == 0 || kvHeads == 0 || heads%kvHeads != 0 || s.LayerFeedForwardLength(block) == 0 {
				return fmt.Errorf("PLaMo 3 layer %d dimensions are invalid", block)
			}
		}
		if s.RopeDimensionCount != s.KeyLength || s.KeyLength%2 != 0 ||
			(s.SlidingWindow > 0 && (s.RopeFrequencySWA <= 0 ||
				(len(s.SlidingLayers) == 0 && s.SlidingPattern < 2))) {
			return errors.New("PLaMo 3 rotary or sliding-attention metadata is invalid")
		}
	}
	if attention == AttentionValidationDeci {
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
	if attention == AttentionValidationGemma3 {
		switch {
		case s.RopeFrequencySWA <= 0:
			return errors.New("Gemma 3 sliding RoPE frequency base must be positive")
		case s.RopeScalingFactor <= 0:
			return errors.New("Gemma 3 RoPE scaling factor must be positive")
		case s.SlidingWindow > 0 && s.SlidingPattern < 2:
			return errors.New("Gemma 3 sliding attention pattern must be at least 2")
		}
	}
	if attention == AttentionValidationGemma3N {
		switch {
		case s.BlockCount != validation.RequiredBlockCount &&
			s.BlockCount != validation.AlternateBlockCount:
			return fmt.Errorf(
				"Gemma 3n block count must be %d or %d",
				validation.RequiredBlockCount, validation.AlternateBlockCount,
			)
		case s.KVFromStart != defaults.SharedKVStartLayer || s.SharedKVLayers != s.BlockCount-s.KVFromStart:
			return errors.New("Gemma 3n shared-KV boundary is invalid")
		case s.AltUpCount != defaults.AlternateStateCount || s.AltUpActive != defaults.AlternateStateActive ||
			s.LaurelRank != defaults.LowRankResidualWidth ||
			s.EmbeddingPerLayer != defaults.PerLayerEmbeddingWidth:
			return errors.New("Gemma 3n AltUp/Laurel dimensions are invalid")
		case s.SparseLayerCount != defaults.SparseLayerCount ||
			s.SparsityStdMultiplier != defaults.SparsityStdMultiplier:
			return errors.New("Gemma 3n sparsity parameters are invalid")
		case s.KeyLength == 0 || s.KeyLength != s.ValueLength || s.HeadCount == 0 ||
			s.HeadCountKV == 0 || s.HeadCount%s.HeadCountKV != 0:
			return errors.New("Gemma 3n attention dimensions are invalid")
		case s.RopeDimensionCount != s.KeyLength || s.KeyLength%2 != 0 ||
			s.RopeFrequencySWA <= 0 || s.SlidingWindow == 0 ||
			s.SlidingPattern != validation.SlidingPeriod:
			return errors.New("Gemma 3n rotary/sliding metadata is invalid")
		case s.FinalLogitSoftcap <= 0:
			return errors.New("Gemma 3n final logit softcap must be positive")
		}
	}
	if attention == AttentionValidationGemma4 {
		switch {
		case s.KeyLength == 0 || s.ValueLength == 0 || s.KeyLength != s.ValueLength ||
			s.KeyLengthSWA == 0 || s.ValueLengthSWA == 0 || s.KeyLengthSWA != s.ValueLengthSWA:
			return errors.New("Gemma 4 attention head dimensions are invalid")
		case s.RopeDimensionCount == 0 || s.RopeDimensionCount > s.KeyLength || s.RopeDimensionCount%2 != 0 ||
			s.RopeDimensionSWA == 0 || s.RopeDimensionSWA > s.KeyLengthSWA || s.RopeDimensionSWA%2 != 0:
			return errors.New("Gemma 4 rotary dimensions are invalid")
		case s.RopeFrequencySWA <= 0 || s.SlidingWindow == 0 || len(s.SlidingLayers) != int(s.BlockCount):
			return errors.New("Gemma 4 sliding-attention metadata is invalid")
		case s.SharedKVLayers > 0 && (s.SharedKVLayers >= s.BlockCount || s.BlockCount-s.SharedKVLayers < 2):
			return errors.New("Gemma 4 shared-KV layer count is invalid")
		case len(s.LayerFeedForward) != int(s.BlockCount) || len(s.LayerKVHeadCounts) != int(s.BlockCount):
			return errors.New("Gemma 4 layer metadata is invalid")
		case s.ExpertCount > 0 && (s.ExpertUsedCount == 0 || s.ExpertUsedCount > s.ExpertCount ||
			exceedsMoETopK(s.ExpertUsedCount) || s.ExpertFeedForward == 0):
			return errors.New("Gemma 4 expert metadata is invalid")
		}
		for block := uint32(0); block < s.BlockCount; block++ {
			if s.LayerFeedForwardLength(block) == 0 || s.LayerKVHeadCount(block) == 0 ||
				s.HeadCount%s.LayerKVHeadCount(block) != 0 {
				return errors.New("Gemma 4 per-layer dimensions are invalid")
			}
		}
	}
	if attention == AttentionValidationGemma4Assistant {
		switch {
		case s.TargetHiddenSize == 0 || s.TargetHiddenSize == s.EmbeddingLength:
			return errors.New("Gemma 4 assistant target hidden size is invalid")
		case s.KeyLength == 0 || s.ValueLength == 0 || s.KeyLength != s.ValueLength ||
			s.KeyLengthSWA == 0 || s.ValueLengthSWA == 0 || s.KeyLengthSWA != s.ValueLengthSWA:
			return errors.New("Gemma 4 assistant attention head dimensions are invalid")
		case s.HeadCount == 0 || s.HeadCountKV == 0 || s.HeadCount%s.HeadCountKV != 0:
			return errors.New("Gemma 4 assistant attention head counts are invalid")
		case s.RopeDimensionCount == 0 || s.RopeDimensionCount > s.KeyLength || s.RopeDimensionCount%2 != 0 ||
			s.RopeDimensionSWA == 0 || s.RopeDimensionSWA > s.KeyLengthSWA || s.RopeDimensionSWA%2 != 0:
			return errors.New("Gemma 4 assistant rotary dimensions are invalid")
		case s.RopeFrequencySWA <= 0 || s.SlidingWindow == 0 || len(s.SlidingLayers) != int(s.BlockCount):
			return errors.New("Gemma 4 assistant sliding-attention metadata is invalid")
		}
	}
	if attention == AttentionValidationGemma2 {
		switch {
		case s.RopeFrequencySWA <= 0:
			return errors.New("Gemma 2 sliding RoPE frequency base must be positive")
		case s.SlidingWindow > 0 && s.SlidingPattern < 2:
			return errors.New("Gemma 2 sliding attention pattern must be at least 2")
		}
	}
	if attention == AttentionValidationOLMo2 {
		switch {
		case s.SlidingWindow > 0 && s.RopeFrequencySWA <= 0:
			return errors.New("OLMo2 sliding RoPE frequency base must be positive")
		case s.SlidingWindow > 0 && s.SlidingPattern < 2:
			return errors.New("OLMo2 sliding attention pattern must be at least 2")
		}
	}
	if attention == AttentionValidationCohere2 || attention == AttentionValidationCohere2MoE {
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
	if attention == AttentionValidationCohere2MoE &&
		(s.LeadingDenseBlocks >= s.BlockCount || s.ExpertCount == 0 || s.ExpertUsedCount == 0 ||
			s.ExpertUsedCount > s.ExpertCount || s.ExpertFeedForward == 0 ||
			(s.ExpertGatingFunc != expertGatingSigmoid) || (s.SharedExpertCount > 0 && s.SharedExpertFF == 0)) {
		return errors.New("Cohere2-MoE expert metadata is invalid")
	}
	if attention == AttentionValidationErnie45MoE &&
		(s.ExpertCount == 0 || s.ExpertUsedCount == 0 || s.ExpertUsedCount > s.ExpertCount ||
			s.ExpertFeedForward == 0 || s.MoELayerStep == 0 || s.LeadingDenseBlocks >= s.BlockCount) {
		return errors.New("ERNIE 4.5 MoE expert metadata is invalid")
	}
	if attention == AttentionValidationStableLM &&
		(s.RopeDimensionCount == 0 || s.RopeDimensionCount > s.KeyLength ||
			s.RopeDimensionCount%2 != 0) {
		return errors.New("StableLM rotary dimension count is invalid")
	}
	if attention == AttentionValidationPhi2 &&
		(s.RopeDimensionCount == 0 || s.RopeDimensionCount > s.KeyLength ||
			s.RopeDimensionCount%2 != 0) {
		return errors.New("Phi-2 rotary dimension count is invalid")
	}
	if attention == AttentionValidationPhi3 &&
		(s.RopeDimensionCount == 0 || s.RopeDimensionCount > s.KeyLength ||
			s.RopeDimensionCount%2 != 0 || s.OriginalContextLength == 0 ||
			s.RopeAttentionFactor <= 0 || math.IsNaN(float64(s.RopeAttentionFactor)) ||
			math.IsInf(float64(s.RopeAttentionFactor), 0)) {
		return errors.New("Phi-3 RoPE metadata is invalid")
	}
	if attention == AttentionValidationPanguEmbedded &&
		(s.RopeDimensionCount != s.KeyLength || s.RopeDimensionCount%2 != 0 ||
			s.KeyLength != s.ValueLength || s.OriginalContextLength == 0 ||
			s.RopeAttentionFactor <= 0 || math.IsNaN(float64(s.RopeAttentionFactor)) ||
			math.IsInf(float64(s.RopeAttentionFactor), 0)) {
		return errors.New("Pangu Embedded RoPE metadata is invalid")
	}
	if attention == AttentionValidationModernBERT {
		switch {
		case s.HeadCountKV != s.HeadCount || s.KeyLength != s.ValueLength:
			return errors.New("ModernBERT requires full-head matching key/value attention")
		case !validRotaryDimension(s.RopeDimensionCount, s.KeyLength, 2):
			return errors.New("ModernBERT rotary dimension count is invalid")
		case s.SlidingWindow > 0 && (s.SlidingPattern < 2 || s.RopeFrequencySWA <= 0):
			return errors.New("ModernBERT sliding attention metadata is invalid")
		}
	}
	if attention == AttentionValidationGemmaEmbedding {
		switch {
		case s.KeyLength != s.ValueLength:
			return errors.New("Gemma embedding requires matching key/value head widths")
		case !validRotaryDimension(s.RopeDimensionCount, s.KeyLength, 2):
			return errors.New("Gemma embedding rotary dimension count is invalid")
		case s.SlidingWindow == 0 || s.SlidingPattern < 2 || s.RopeFrequencySWA <= 0:
			return errors.New("Gemma embedding sliding attention metadata is invalid")
		case s.Dense2FeatureIn > 0 && s.Dense2FeatureIn != s.EmbeddingLength:
			return errors.New("Gemma embedding dense-2 input width must match embedding length")
		case s.Dense3FeatureOut > 0 && s.Dense3FeatureOut != s.EmbeddingLength:
			return errors.New("Gemma embedding dense-3 output width must match embedding length")
		}
	}
	if attention == AttentionValidationTalkie &&
		(s.KeyLength != s.ValueLength || s.RopeDimensionCount != s.KeyLength || s.RopeDimensionCount%2 != 0) {
		return errors.New("Talkie attention metadata is invalid")
	}
	if attention == AttentionValidationApertus {
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
	if attention == AttentionValidationGPTNeoX && s.RopeDimensionCount > 0 &&
		(s.RopeDimensionCount > s.KeyLength || s.RopeDimensionCount%2 != 0) {
		return errors.New("GPT-NeoX rotary dimension count is invalid")
	}
	if attention == AttentionValidationQwen &&
		(s.HeadCountKV != s.HeadCount || s.RopeDimensionCount == 0 ||
			s.RopeDimensionCount > s.KeyLength || s.RopeDimensionCount%2 != 0) {
		return errors.New("Qwen attention metadata is invalid")
	}
	if attention == AttentionValidationChatGLM &&
		(s.RopeDimensionCount == 0 || s.RopeDimensionCount > s.KeyLength ||
			s.RopeDimensionCount%2 != 0 || s.KeyLength != s.ValueLength) {
		return errors.New("ChatGLM attention metadata is invalid")
	}
	if attention == AttentionValidationCogVLM &&
		(s.HeadCountKV != s.HeadCount || s.KeyLength != s.ValueLength ||
			uint64(s.KeyLength)*uint64(s.HeadCount) != uint64(s.EmbeddingLength) ||
			s.RopeDimensionCount != s.KeyLength) {
		return errors.New("CogVLM attention metadata is invalid")
	}
	if attention == AttentionValidationHunyuan &&
		(s.RopeDimensionCount != s.KeyLength || s.KeyLength != s.ValueLength ||
			s.RopeDimensionCount%2 != 0 || s.RopeFrequencyBase <= 0 ||
			math.IsNaN(float64(s.RopeFrequencyBase)) || math.IsInf(float64(s.RopeFrequencyBase), 0)) {
		return fmt.Errorf("%s attention metadata is invalid", s.Architecture)
	}
	if attention == AttentionValidationHunyuan {
		for _, section := range s.RopeSections {
			if section < 0 {
				return errors.New("Hunyuan MRoPE section count is negative")
			}
		}
	}
	if (attention == AttentionValidationGLM4 || attention == AttentionValidationGLM4MoE) &&
		(s.RopeDimensionCount == 0 || s.RopeDimensionCount > s.KeyLength ||
			s.RopeDimensionCount%2 != 0) {
		return errors.New("GLM4 rotary dimension count is invalid")
	}
	if attention == AttentionValidationGLM4 || attention == AttentionValidationGLM4MoE {
		for _, section := range s.RopeSections {
			if section < 0 {
				return errors.New("GLM4 MRoPE section count is negative")
			}
		}
	}
	if attention == AttentionValidationGLM4MoE &&
		(s.LeadingDenseBlocks >= s.BlockCount || !validMoESelection(s.ExpertUsedCount, s.ExpertCount) ||
			s.ExpertFeedForward == 0 ||
			s.SharedExpertCount == 0 || s.SharedExpertFF == 0 ||
			(s.ExpertGatingFunc != expertGatingSoftmax && s.ExpertGatingFunc != expertGatingSigmoid) ||
			s.ExpertWeightsScale == 0 || math.IsNaN(float64(s.ExpertWeightsScale)) ||
			math.IsInf(float64(s.ExpertWeightsScale), 0)) {
		return errors.New("GLM4-MoE expert metadata is invalid")
	}
	if attention == AttentionValidationEXAOne4 {
		switch {
		case s.RopeDimensionCount != s.KeyLength || s.RopeDimensionCount%2 != 0:
			return errors.New("EXAONE 4 rotary dimension count must equal the key length")
		case s.SlidingWindow > 0 && s.SlidingPattern < 2:
			return errors.New("EXAONE 4 sliding attention pattern must be at least 2")
		case s.SlidingWindow > 0 && s.RopeFrequencySWA <= 0:
			return errors.New("EXAONE 4 sliding RoPE frequency base must be positive")
		}
	}
	if attention == AttentionValidationFalcon && s.RopeDimensionCount > 0 &&
		s.RopeDimensionCount != s.KeyLength {
		return errors.New("Falcon rotary dimension count must equal the key length")
	}
	if attention == AttentionValidationRefact && s.ExpertCount > 0 &&
		(!validExpertDimensions(s) || !positiveFinite(s.ExpertWeightsScale)) {
		return errors.New("Refact expert metadata is invalid")
	}
	return nil
}
