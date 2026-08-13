package model

import (
	"errors"
	"fmt"
	"math"

	"overgo/internal/tensor"
)

const (
	deepSeek4BlockCount        = uint32(43)
	deepSeek4HyperConnections  = uint32(4)
	kimiLinearMinimumConvWidth = uint32(2)
	deepSeek2LiteBlockCountA   = uint32(26)
	deepSeek2LiteBlockCountB   = uint32(27)
	deepSeek2LiteBlockCountC   = uint32(48)
	deepSeek2LiteVocabulary    = uint32(128256)
)

func (s Spec) validateMLAFamilies() error {
	profile := s.Profile()
	validation := profile.Validation.MLA
	kimiLinear := validation == MLAValidationKimiLinear
	if (profile.Attention == AttentionLatent || profile.Attention == AttentionSparseLatent || kimiLinear) &&
		(s.KVLoRARank == 0 || s.RopeDimensionCount == 0 ||
			s.RopeDimensionCount >= s.KeyLength ||
			(!profile.Has(ArchitectureLatentKVLayout) && !kimiLinear && s.HeadCountKV != s.HeadCount) ||
			((profile.Has(ArchitectureLatentKVLayout) || kimiLinear) && s.HeadCountKV != 1 && s.HeadCountKV != s.HeadCount)) {
		return errors.New("MLA metadata is invalid")
	}
	if profile.Attention == AttentionSparseLatent {
		switch {
		case s.QLoRARank == 0:
			return errors.New("DSA query LoRA rank is missing")
		case s.IndexerHeadCount == 0 || s.IndexerKeyLength == 0 || s.IndexerTopK == 0 ||
			s.IndexerTopK > s.ContextLength || s.IndexerKeyLength < s.RopeDimensionCount ||
			s.IndexerKeyLength&(s.IndexerKeyLength-1) != 0:
			return errors.New("DSA indexer metadata is invalid")
		case len(s.IndexerFullLayers) != int(s.BlockCount) || !s.IndexerFullLayers[0]:
			return errors.New("DSA indexer schedule is invalid")
		case profile.Validation.RequiredBlockCount > 0 &&
			(s.BlockCount != profile.Validation.RequiredBlockCount ||
				s.LayerNormEpsilon != profile.MetadataDefaults.LayerNormEpsilon):
			return errors.New("DeepSeek 3.2 layer metadata is invalid")
		}
		var sectionPairs int32
		for _, section := range s.RopeSections {
			if section < 0 {
				return errors.New("DSA RoPE section is negative")
			}
			sectionPairs += section
		}
		if sectionPairs > int32(s.RopeDimensionCount/2) {
			return errors.New("DSA RoPE sections are invalid")
		}
		seenFull := false
		for _, full := range s.IndexerFullLayers {
			if full {
				seenFull = true
			} else if !seenFull {
				return errors.New("DSA shared indexer precedes every full indexer")
			}
		}
		if validation == MLAValidationDeepSeek32 {
			for _, full := range s.IndexerFullLayers {
				if !full {
					return errors.New("DeepSeek 3.2 requires a full indexer in every layer")
				}
			}
		}
	}
	if validation == MLAValidationDeepSeek4 {
		switch {
		case s.BlockCount != deepSeek4BlockCount || s.HeadCountKV != 1 || s.KeyLength != s.ValueLength:
			return errors.New("DeepSeek 4 layer metadata is invalid")
		case s.QLoRARank == 0 || s.RopeDimensionCount == 0 || s.RopeDimensionCount > s.KeyLength || s.RopeDimensionCount%2 != 0:
			return errors.New("DeepSeek 4 attention dimensions are invalid")
		case s.SlidingWindow == 0 || s.CompressRopeBase <= 0 ||
			math.IsNaN(float64(s.CompressRopeBase)) || math.IsInf(float64(s.CompressRopeBase), 0):
			return errors.New("DeepSeek 4 compressed-attention metadata is invalid")
		case s.AttentionOutputGroups == 0 || s.HeadCount%s.AttentionOutputGroups != 0 || s.AttentionOutputRank == 0:
			return errors.New("DeepSeek 4 output LoRA metadata is invalid")
		case s.HyperConnectionCount != deepSeek4HyperConnections || s.HyperSinkhornIters == 0 || s.HyperConnectionEps <= 0 ||
			math.IsNaN(float64(s.HyperConnectionEps)) || math.IsInf(float64(s.HyperConnectionEps), 0):
			return errors.New("DeepSeek 4 hyper-connection metadata is invalid")
		case s.IndexerHeadCount == 0 || s.IndexerKeyLength < s.RopeDimensionCount || s.IndexerTopK == 0 ||
			s.IndexerTopK > s.ContextLength || s.IndexerKeyLength&(s.IndexerKeyLength-1) != 0:
			return errors.New("DeepSeek 4 indexer metadata is invalid")
		case s.ExpertCount == 0 || s.ExpertUsedCount == 0 || s.ExpertUsedCount > s.ExpertCount ||
			exceedsMoETopK(s.ExpertUsedCount) || s.ExpertFeedForward == 0 || s.SharedExpertCount == 0 || s.SharedExpertFF == 0:
			return errors.New("DeepSeek 4 expert metadata is invalid")
		case s.ExpertGatingFunc != expertGatingSqrtSoftplus:
			return errors.New("DeepSeek 4 expert routing function is unsupported")
		case s.ExpertWeightsScale <= 0 || math.IsNaN(float64(s.ExpertWeightsScale)) || math.IsInf(float64(s.ExpertWeightsScale), 0):
			return errors.New("DeepSeek 4 expert weight scale is invalid")
		case s.HashLayerCount > s.BlockCount || len(s.CompressRatios) != int(s.BlockCount) ||
			len(s.LayerSwiGLUClamp) != int(s.BlockCount) || len(s.LayerSharedSwiGLUClamp) != int(s.BlockCount):
			return errors.New("DeepSeek 4 layer schedule is invalid")
		case s.RopeScalingType == "yarn" &&
			(s.RopeScalingFactor <= 0 || s.OriginalContextLength == 0 || s.YaRNExtFactor < 0 ||
				s.YaRNAttentionFactor <= 0 || s.YaRNBetaFast <= 0 || s.YaRNBetaSlow <= 0):
			return errors.New("DeepSeek 4 YaRN metadata is invalid")
		}
		for block, ratio := range s.CompressRatios {
			if !tensor.CompressionRatio(ratio).Valid() {
				return fmt.Errorf("DeepSeek 4 layer %d compression ratio is invalid", block)
			}
			for _, limit := range []float32{s.LayerSwiGLUClamp[block], s.LayerSharedSwiGLUClamp[block]} {
				if limit < 0 || math.IsNaN(float64(limit)) || math.IsInf(float64(limit), 0) {
					return fmt.Errorf("DeepSeek 4 layer %d SwiGLU clamp is invalid", block)
				}
			}
		}
	}
	if kimiLinear {
		switch {
		case len(s.RecurrentLayers) != int(s.BlockCount) || len(s.LayerKVHeadCounts) != int(s.BlockCount):
			return errors.New("Kimi Linear layer schedule is invalid")
		case s.SSMConvKernel < kimiLinearMinimumConvWidth || s.KDAHeadDim == 0 || s.SSMInnerSize != s.HeadCount*s.KDAHeadDim:
			return errors.New("Kimi Linear KDA metadata is invalid")
		case s.LeadingDenseBlocks >= s.BlockCount || !validMoESelection(s.ExpertUsedCount, s.ExpertCount) ||
			s.ExpertFeedForward == 0 ||
			s.SharedExpertCount == 0 || s.SharedExpertFF == 0:
			return errors.New("Kimi Linear expert metadata is invalid")
		case s.ExpertGatingFunc != expertGatingSoftmax && s.ExpertGatingFunc != expertGatingSigmoid:
			return errors.New("Kimi Linear expert routing function is unsupported")
		case s.ExpertWeightsScale <= 0 || math.IsNaN(float64(s.ExpertWeightsScale)) || math.IsInf(float64(s.ExpertWeightsScale), 0):
			return errors.New("Kimi Linear expert weight scale is invalid")
		}
		var recurrent, attention bool
		for _, item := range s.RecurrentLayers {
			recurrent = recurrent || item
			attention = attention || !item
		}
		if !recurrent || !attention {
			return errors.New("Kimi Linear requires KDA and MLA layers")
		}
	}
	if profile.Has(ArchitectureLatentKVLayout) {
		lite := s.BlockCount == deepSeek2LiteBlockCountA || s.BlockCount == deepSeek2LiteBlockCountB ||
			(s.BlockCount == deepSeek2LiteBlockCountC && s.VocabularySize == deepSeek2LiteVocabulary)
		switch {
		case s.ExpertCount == 0 && s.LeadingDenseBlocks != s.BlockCount:
			return errors.New("dense DeepSeek2 requires every block to be dense")
		case s.ExpertCount > 0 && s.LeadingDenseBlocks >= s.BlockCount:
			return errors.New("DeepSeek2 leading dense block count leaves no MoE layers")
		case s.ExpertCount == 0 && (s.ExpertUsedCount != 0 || s.SharedExpertCount != 0 || s.SharedExpertFF != 0):
			return errors.New("dense DeepSeek2 expert metadata is inconsistent")
		case s.ExpertCount > 0 && (s.ExpertUsedCount == 0 || s.ExpertUsedCount > s.ExpertCount ||
			exceedsMoETopK(s.ExpertUsedCount) || s.ExpertFeedForward == 0 || s.SharedExpertCount == 0 || s.SharedExpertFF == 0):
			return errors.New("DeepSeek2 expert metadata is invalid")
		case s.ExpertCount > 0 && s.SharedExpertFF/s.SharedExpertCount != s.ExpertFeedForward:
			return errors.New("DeepSeek2 shared expert width overflows")
		case s.ExpertWeightsScale <= 0 || math.IsNaN(float64(s.ExpertWeightsScale)) || math.IsInf(float64(s.ExpertWeightsScale), 0):
			return errors.New("DeepSeek2 expert weight scale is invalid")
		case s.ExpertGatingFunc != expertGatingSoftmax && s.ExpertGatingFunc != expertGatingSigmoid:
			return errors.New("DeepSeek2 expert routing function is unsupported")
		case !lite && s.QLoRARank == 0:
			return errors.New("DeepSeek2 query LoRA rank is missing")
		case s.RopeDimensionCount%2 != 0:
			return errors.New("DeepSeek2 rotary dimension is invalid")
		case s.RopeScalingType == "yarn" &&
			(s.RopeScalingFactor <= 0 || s.OriginalContextLength == 0 || s.YaRNExtFactor < 0 ||
				s.YaRNAttentionFactor <= 0 || s.YaRNBetaFast <= 0 || s.YaRNBetaSlow <= 0 ||
				math.IsNaN(float64(s.RopeScalingFactor)) || math.IsInf(float64(s.RopeScalingFactor), 0) ||
				math.IsNaN(float64(s.YaRNExtFactor)) || math.IsInf(float64(s.YaRNExtFactor), 0) ||
				math.IsNaN(float64(s.YaRNAttentionFactor)) || math.IsInf(float64(s.YaRNAttentionFactor), 0) ||
				math.IsNaN(float64(s.YaRNBetaFast)) || math.IsInf(float64(s.YaRNBetaFast), 0) ||
				math.IsNaN(float64(s.YaRNBetaSlow)) || math.IsInf(float64(s.YaRNBetaSlow), 0)):
			return errors.New("DeepSeek2 YaRN metadata is invalid")
		case math.IsNaN(float64(s.RopeYaRNLogMultiplier)) || math.IsInf(float64(s.RopeYaRNLogMultiplier), 0):
			return errors.New("DeepSeek2 YaRN log multiplier is invalid")
		case s.AttentionTempScale != 0 && (s.AttentionTempScale <= 0 || s.AttentionTempFloor == 0 ||
			math.IsNaN(float64(s.AttentionTempScale)) || math.IsInf(float64(s.AttentionTempScale), 0)):
			return errors.New("DeepSeek2 attention temperature metadata is invalid")
		}
	}
	if validation == MLAValidationMistral3 {
		switch {
		case s.AttentionTempScale != 0 &&
			(s.AttentionTempScale <= 0 || s.AttentionTempFloor == 0 ||
				math.IsNaN(float64(s.AttentionTempScale)) || math.IsInf(float64(s.AttentionTempScale), 0)):
			return errors.New("Mistral 3 attention temperature metadata is invalid")
		case s.ExpertCount == 0 && (s.ExpertUsedCount != 0 || s.ExpertFeedForward != 0):
			return errors.New("dense Mistral 3 expert metadata is inconsistent")
		case s.ExpertCount > 0 &&
			(s.ExpertUsedCount == 0 || s.ExpertUsedCount > s.ExpertCount ||
				exceedsMoETopK(s.ExpertUsedCount) || s.ExpertFeedForward == 0 ||
				s.ExpertWeightsScale <= 0 || math.IsNaN(float64(s.ExpertWeightsScale)) ||
				math.IsInf(float64(s.ExpertWeightsScale), 0)):
			return errors.New("Mistral 3 expert metadata is invalid")
		case s.RopeScalingType == "yarn" &&
			(math.IsNaN(float64(s.RopeYaRNLogMultiplier)) || math.IsInf(float64(s.RopeYaRNLogMultiplier), 0)):
			return errors.New("Mistral 3 YaRN metadata is invalid")
		}
	}
	if validation == MLAValidationMiniCPM3 &&
		(s.QLoRARank == 0 || s.ResidualScale <= 0 || s.OriginalContextLength == 0 ||
			s.RopeAttentionFactor <= 0 || math.IsNaN(float64(s.ResidualScale)) ||
			math.IsInf(float64(s.ResidualScale), 0) || math.IsNaN(float64(s.RopeAttentionFactor)) ||
			math.IsInf(float64(s.RopeAttentionFactor), 0)) {
		return errors.New("MiniCPM3 metadata is invalid")
	}
	return nil
}
