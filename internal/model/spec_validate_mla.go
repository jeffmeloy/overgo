package model

import (
	"errors"
	"fmt"

	"overgo/internal/tensor"
)

func (s Spec) validateMLAMetadata() error {
	profile := s.Profile()
	validation := profile.Validation.MLA
	mixedLinearAttention := validation == MLAValidationHybridLinearAttention
	if (profile.Attention == AttentionLatent || profile.Attention == AttentionSparseLatent || mixedLinearAttention) &&
		(s.KVLoRARank == tensor.FirstOffset || s.RopeDimensionCount == tensor.FirstOffset ||
			s.RopeDimensionCount >= s.KeyLength ||
			(!profile.Has(ArchitectureLatentKVLayout) && !mixedLinearAttention && s.HeadCountKV != s.HeadCount) ||
			((profile.Has(ArchitectureLatentKVLayout) || mixedLinearAttention) &&
				s.HeadCountKV != tensor.SingletonExtent && s.HeadCountKV != s.HeadCount)) {
		return errors.New("MLA metadata is invalid")
	}
	if profile.Attention == AttentionSparseLatent {
		switch {
		case s.QLoRARank == tensor.FirstOffset:
			return errors.New("DSA query LoRA rank is missing")
		case s.IndexerHeadCount == tensor.FirstOffset || s.IndexerKeyLength == tensor.FirstOffset ||
			s.IndexerTopK == tensor.FirstOffset ||
			s.IndexerTopK > s.ContextLength || s.IndexerKeyLength < s.RopeDimensionCount ||
			s.IndexerKeyLength&(s.IndexerKeyLength-tensor.SingletonExtent) != tensor.FirstOffset:
			return errors.New("DSA indexer metadata is invalid")
		case len(s.IndexerFullLayers) != int(s.BlockCount) || !s.IndexerFullLayers[tensor.FirstOffset]:
			return errors.New("DSA indexer schedule is invalid")
		case profile.Validation.RequiredBlockCount > tensor.FirstOffset &&
			(s.BlockCount != profile.Validation.RequiredBlockCount ||
				s.LayerNormEpsilon != profile.MetadataDefaults.LayerNormEpsilon):
			return errors.New("DeepSeek 3.2 layer metadata is invalid")
		}
		var sectionPairs int32
		for _, section := range s.RopeSections {
			if section < tensor.FirstOffset {
				return errors.New("DSA RoPE section is negative")
			}
			sectionPairs += section
		}
		if sectionPairs > int32(s.RopeDimensionCount/rotaryPairAlignment) {
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
		if validation == MLAValidationSparseLatentIndexer {
			for _, full := range s.IndexerFullLayers {
				if !full {
					return errors.New("DeepSeek 3.2 requires a full indexer in every layer")
				}
			}
		}
	}
	if validation == MLAValidationCompressedHyper {
		_, validOutputGroups := tensor.EqualPartition(uint64(s.HeadCount), uint64(s.AttentionOutputGroups))
		switch {
		case s.HeadCountKV != tensor.SingletonExtent || s.KeyLength != s.ValueLength:
			return errors.New("DeepSeek 4 layer metadata is invalid")
		case s.QLoRARank == tensor.FirstOffset || !validRotaryDimension(s.RopeDimensionCount, s.KeyLength):
			return errors.New("DeepSeek 4 attention dimensions are invalid")
		case s.SlidingWindow == tensor.FirstOffset || !positiveFinite(s.CompressRopeBase):
			return errors.New("DeepSeek 4 compressed-attention metadata is invalid")
		case !validOutputGroups || s.AttentionOutputRank == tensor.FirstOffset:
			return errors.New("DeepSeek 4 output LoRA metadata is invalid")
		case s.HyperConnectionCount == tensor.FirstOffset || s.HyperSinkhornIters == tensor.FirstOffset ||
			!positiveFinite(s.HyperConnectionEps):
			return errors.New("DeepSeek 4 hyper-connection metadata is invalid")
		case s.IndexerHeadCount == tensor.FirstOffset || s.IndexerKeyLength < s.RopeDimensionCount ||
			s.IndexerTopK == tensor.FirstOffset ||
			s.IndexerTopK > s.ContextLength ||
			s.IndexerKeyLength&(s.IndexerKeyLength-tensor.SingletonExtent) != tensor.FirstOffset:
			return errors.New("DeepSeek 4 indexer metadata is invalid")
		case !validExpertDimensions(s) || !s.HasSharedExperts():
			return errors.New("DeepSeek 4 expert metadata is invalid")
		case s.ExpertGatingFunc != expertGatingSqrtSoftplus:
			return errors.New("DeepSeek 4 expert routing function is unsupported")
		case !positiveFinite(s.ExpertWeightsScale):
			return errors.New("DeepSeek 4 expert weight scale is invalid")
		case s.HashLayerCount > s.BlockCount || len(s.CompressRatios) != int(s.BlockCount) ||
			len(s.LayerSwiGLUClamp) != int(s.BlockCount) || len(s.LayerSharedSwiGLUClamp) != int(s.BlockCount):
			return errors.New("DeepSeek 4 layer schedule is invalid")
		case s.RopeScalingType == ropeScalingYaRN && !validYaRN(s):
			return errors.New("DeepSeek 4 YaRN metadata is invalid")
		}
		for block, ratio := range s.CompressRatios {
			if !tensor.CompressionRatio(ratio).Valid() {
				return fmt.Errorf("DeepSeek 4 layer %d compression ratio is invalid", block)
			}
			for _, limit := range []float32{s.LayerSwiGLUClamp[block], s.LayerSharedSwiGLUClamp[block]} {
				if !nonNegativeFinite(limit) {
					return fmt.Errorf("DeepSeek 4 layer %d SwiGLU clamp is invalid", block)
				}
			}
		}
	}
	if mixedLinearAttention {
		headWidth, validHeads := tensor.EqualPartition(uint64(s.SSMInnerSize), uint64(s.HeadCount))
		switch {
		case !validMixedLayerSchedule(s.RecurrentLayers, s.BlockCount) || len(s.LayerKVHeadCounts) != int(s.BlockCount):
			return errors.New("mixed latent-attention layer schedule is invalid")
		case s.SSMConvKernel < minimumConvKernelWidth || !validHeads || uint64(s.KDAHeadDim) != headWidth:
			return errors.New("mixed linear-attention state metadata is invalid")
		case s.LeadingDenseBlocks >= s.BlockCount || !validExpertDimensions(s) || !s.HasSharedExperts():
			return errors.New("Kimi Linear expert metadata is invalid")
		case !validExpertRouting(s):
			return errors.New("Kimi Linear expert routing function is unsupported")
		case !positiveFinite(s.ExpertWeightsScale):
			return errors.New("Kimi Linear expert weight scale is invalid")
		}
	}
	if profile.Has(ArchitectureLatentKVLayout) {
		switch {
		case !s.HasExperts() && s.LeadingDenseBlocks != s.BlockCount:
			return errors.New("dense DeepSeek2 requires every block to be dense")
		case s.HasExperts() && s.LeadingDenseBlocks >= s.BlockCount:
			return errors.New("DeepSeek2 leading dense block count leaves no MoE layers")
		case !s.HasExperts() && (s.ExpertUsedCount != tensor.FirstOffset || s.HasSharedExperts()):
			return errors.New("dense DeepSeek2 expert metadata is inconsistent")
		case s.HasExperts() && (!validExpertDimensions(s) || !s.HasSharedExperts()):
			return errors.New("DeepSeek2 expert metadata is invalid")
		case s.HasExperts() && s.SharedExpertFF/s.SharedExpertCount != s.ExpertFeedForward:
			return errors.New("DeepSeek2 shared expert width overflows")
		case !positiveFinite(s.ExpertWeightsScale):
			return errors.New("DeepSeek2 expert weight scale is invalid")
		case !validExpertRouting(s):
			return errors.New("DeepSeek2 expert routing function is unsupported")
		case !profile.Validation.QLoRARankOptional && s.QLoRARank == tensor.FirstOffset:
			return errors.New("DeepSeek2 query LoRA rank is missing")
		case s.RopeDimensionCount%rotaryPairAlignment != tensor.FirstOffset:
			return errors.New("DeepSeek2 rotary dimension is invalid")
		case s.RopeScalingType == ropeScalingYaRN && !validYaRN(s):
			return errors.New("DeepSeek2 YaRN metadata is invalid")
		case !finite(s.RopeYaRNLogMultiplier):
			return errors.New("DeepSeek2 YaRN log multiplier is invalid")
		case !validOptionalAttentionTemperature(s):
			return errors.New("DeepSeek2 attention temperature metadata is invalid")
		}
	}
	if validation == MLAValidationOptionalExpertsLatent {
		switch {
		case !validOptionalAttentionTemperature(s):
			return errors.New("Mistral 3 attention temperature metadata is invalid")
		case !s.HasExperts() &&
			(s.ExpertUsedCount != tensor.FirstOffset || s.ExpertFeedForward != tensor.FirstOffset):
			return errors.New("dense Mistral 3 expert metadata is inconsistent")
		case s.HasExperts() &&
			(!validExpertDimensions(s) || !positiveFinite(s.ExpertWeightsScale)):
			return errors.New("Mistral 3 expert metadata is invalid")
		case s.RopeScalingType == ropeScalingYaRN && !finite(s.RopeYaRNLogMultiplier):
			return errors.New("Mistral 3 YaRN metadata is invalid")
		}
	}
	if validation == MLAValidationScaledLatent &&
		(s.QLoRARank == tensor.FirstOffset || !positiveFinite(s.ResidualScale) ||
			s.OriginalContextLength == tensor.FirstOffset || !positiveFinite(s.RopeAttentionFactor)) {
		return errors.New("MiniCPM3 metadata is invalid")
	}
	return nil
}

func validOptionalAttentionTemperature(s Spec) bool {
	return s.AttentionTempScale == tensor.FirstOffset ||
		(positiveFinite(s.AttentionTempScale) && s.AttentionTempFloor > tensor.FirstOffset)
}
