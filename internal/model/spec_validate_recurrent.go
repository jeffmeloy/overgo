package model

import (
	"errors"
	"fmt"
)

func (s Spec) validateRecurrentFamilies() error {
	switch s.Architecture {
	case "wavtokenizer-dec":
		if s.OutputEmbeddingLength == 0 || s.PosNetBlockCount != 6 || s.ConvNextBlockCount == 0 ||
			s.PosNetEmbeddingLength == 0 || s.PosNetEmbeddingLength != s.ConvNextEmbeddingLength ||
			s.GroupNormGroups == 0 || s.GroupNormEpsilon <= 0 ||
			s.PosNetEmbeddingLength%s.GroupNormGroups != 0 {
			return errors.New("WavTokenizer metadata is invalid")
		}
	case "dflash":
		if len(s.TargetLayers) == 0 || s.DFlashBlockSize < 2 {
			return errors.New("DFlash target-layer metadata is invalid")
		}
		return validateNonNegativeTargetLayers("DFlash", s.TargetLayers)
	case "eagle3":
		if s.BlockCount != 1 || len(s.TargetLayers) != 3 || s.TargetHiddenSize == 0 {
			return errors.New("Eagle3 target-layer metadata is invalid")
		}
		return validateNonNegativeTargetLayers("Eagle3", s.TargetLayers)
	case "mamba":
		if !attentionMetadataZero(s) {
			return errors.New("Mamba attention metadata must be zero")
		}
		if s.SSMConvKernel < 2 || s.SSMInnerSize != 2*s.EmbeddingLength ||
			s.SSMStateSize == 0 || s.SSMTimeStepRank == 0 {
			return errors.New("Mamba SSM metadata is invalid")
		}
	case "mamba2":
		if !attentionMetadataZero(s) {
			return errors.New("Mamba2 attention metadata must be zero")
		}
		if !validGroupedSSM(s) {
			return errors.New("Mamba2 SSM metadata is invalid")
		}
	case "falcon-h1":
		if !validGroupedSSM(s) {
			return errors.New("Falcon-H1 SSM metadata is invalid")
		}
		if !validRotaryDimension(s.RopeDimensionCount, s.KeyLength, 2) || s.KeyLength != s.ValueLength {
			return errors.New("Falcon-H1 rotary/head dimensions are invalid")
		}
	case "rwkv6", "rwkv6qwen2":
		return s.validateRWKV6()
	case "rwkv7", "arwkv7":
		return s.validateRWKV7()
	case "jamba":
		if s.SSMConvKernel < 2 || s.SSMInnerSize != 2*s.EmbeddingLength ||
			s.SSMStateSize == 0 || s.SSMTimeStepRank == 0 {
			return errors.New("Jamba SSM metadata is invalid")
		}
		if !validRecurrentLayerSchedule(s) {
			return errors.New("Jamba layer schedule is invalid")
		}
		if !validMoESelection(s.ExpertUsedCount, s.ExpertCount) ||
			s.ExpertFeedForward == 0 || s.ExpertWeightsScale <= 0 {
			return errors.New("Jamba expert metadata is invalid")
		}
	case "granitehybrid":
		return s.validateGraniteHybrid()
	case "plamo2":
		if s.SSMConvKernel < 2 || s.SSMInnerSize == 0 || s.SSMStateSize == 0 ||
			s.SSMTimeStepRank == 0 || s.SSMGroupCount != 0 ||
			s.SSMInnerSize%s.SSMTimeStepRank != 0 {
			return errors.New("PLaMo2 SSM metadata is invalid")
		}
		if !validRecurrentLayerSchedule(s) {
			return errors.New("PLaMo2 layer schedule is invalid")
		}
	case "nemotron_h", "nemotron_h_moe":
		return s.validateNemotronH()
	}
	return nil
}

func attentionMetadataZero(s Spec) bool {
	return s.FeedForwardLength == 0 && s.HeadCount == 0 && s.HeadCountKV == 0 &&
		s.KeyLength == 0 && s.ValueLength == 0
}

func validGroupedSSM(s Spec) bool {
	return s.SSMConvKernel >= 2 && s.SSMInnerSize > 0 && s.SSMStateSize > 0 &&
		s.SSMTimeStepRank > 0 && s.SSMGroupCount > 0 &&
		s.SSMInnerSize%s.SSMTimeStepRank == 0 && s.SSMInnerSize%s.SSMGroupCount == 0 &&
		s.SSMTimeStepRank%s.SSMGroupCount == 0
}

func validRecurrentLayerSchedule(s Spec) bool {
	return len(s.RecurrentLayers) == int(s.BlockCount) &&
		len(s.LayerKVHeadCounts) == int(s.BlockCount)
}

func validateNonNegativeTargetLayers(family string, layers []int32) error {
	for _, layer := range layers {
		if layer < 0 {
			return fmt.Errorf("%s target layer is negative", family)
		}
	}
	return nil
}

func (s Spec) validateRWKV6() error {
	switch {
	case s.WKVHeadSize == 0 || s.EmbeddingLength%s.WKVHeadSize != 0 || s.HeadCount != s.EmbeddingLength/s.WKVHeadSize:
		return fmt.Errorf("%s WKV head metadata is invalid", s.Architecture)
	case s.TimeMixExtraDim == 0 || s.TimeDecayExtraDim == 0 ||
		(s.Architecture == "rwkv6" && s.TokenShiftCount != 2) ||
		(s.Architecture == "rwkv6qwen2" && s.TokenShiftCount != 1):
		return fmt.Errorf("%s time-mix metadata is invalid", s.Architecture)
	case s.KeyLength != s.WKVHeadSize || s.ValueLength != s.WKVHeadSize:
		return fmt.Errorf("%s head dimensions are invalid", s.Architecture)
	case s.Architecture == "rwkv6" && s.HeadCountKV != s.HeadCount:
		return errors.New("rwkv6 KV head count must equal query head count")
	}
	return nil
}

func (s Spec) validateRWKV7() error {
	switch {
	case s.WKVHeadSize == 0 || s.EmbeddingLength%s.WKVHeadSize != 0 || s.HeadCount != s.EmbeddingLength/s.WKVHeadSize:
		return fmt.Errorf("%s WKV head metadata is invalid", s.Architecture)
	case s.HeadCountKV != s.HeadCount || s.KeyLength != s.WKVHeadSize || s.ValueLength != s.WKVHeadSize:
		return fmt.Errorf("%s head dimensions are invalid", s.Architecture)
	case s.DecayLoRARank == 0 || s.ICLRLoRARank == 0 || s.ValueMixLoRARank == 0:
		return fmt.Errorf("%s time-mix LoRA metadata is invalid", s.Architecture)
	case s.Architecture == "rwkv7" && s.GateLoRARank == 0:
		return errors.New("rwkv7 gate LoRA metadata is invalid")
	case (s.Architecture == "rwkv7" && s.TokenShiftCount != 2) ||
		(s.Architecture == "arwkv7" && s.TokenShiftCount != 1):
		return fmt.Errorf("%s token-shift metadata is invalid", s.Architecture)
	}
	return nil
}

func (s Spec) validateGraniteHybrid() error {
	if !validGroupedSSM(s) || s.SSMInnerSize != 2*s.EmbeddingLength {
		return errors.New("Granite Hybrid SSM metadata is invalid")
	}
	if !validRecurrentLayerSchedule(s) {
		return errors.New("Granite Hybrid layer schedule is invalid")
	}
	if s.ExpertCount > 0 && (!validMoESelection(s.ExpertUsedCount, s.ExpertCount) ||
		s.ExpertFeedForward == 0 || s.ExpertWeightsScale <= 0) {
		return errors.New("Granite Hybrid expert metadata is invalid")
	}
	if s.ExpertCount == 0 && (s.ExpertUsedCount != 0 || s.ExpertFeedForward != 0 || s.SharedExpertFF != 0) {
		return errors.New("dense Granite Hybrid expert metadata is inconsistent")
	}
	return nil
}

func (s Spec) validateNemotronH() error {
	if !validGroupedSSM(s) {
		return errors.New("Nemotron-H SSM metadata is invalid")
	}
	if len(s.LayerHeadCounts) != int(s.BlockCount) || len(s.LayerKVHeadCounts) != int(s.BlockCount) ||
		len(s.LayerFeedForward) != int(s.BlockCount) || len(s.RecurrentLayers) != int(s.BlockCount) {
		return errors.New("Nemotron-H layer schedule is invalid")
	}
	for block := uint32(0); block < s.BlockCount; block++ {
		heads, kvHeads, ff := s.LayerHeadCount(block), s.LayerKVHeadCount(block), s.LayerFeedForwardLength(block)
		if ff > 0 {
			continue
		}
		if s.IsRecurrentLayer(block) {
			if heads != 0 || kvHeads != 0 {
				return errors.New("Nemotron-H recurrent layer schedule is invalid")
			}
			continue
		}
		if heads == 0 || kvHeads == 0 || heads%kvHeads != 0 {
			return errors.New("Nemotron-H attention layer schedule is invalid")
		}
	}
	if s.Architecture == "nemotron_h_moe" {
		if !validMoESelection(s.ExpertUsedCount, s.ExpertCount) || s.ExpertFeedForward == 0 ||
			s.SharedExpertFF == 0 || s.ExpertWeightsScale <= 0 {
			return errors.New("Nemotron-H MoE metadata is invalid")
		}
	} else if s.ExpertCount != 0 || s.ExpertUsedCount != 0 || s.ExpertFeedForward != 0 ||
		s.SharedExpertFF != 0 || s.MoELatentSize != 0 {
		return errors.New("dense Nemotron-H expert metadata is inconsistent")
	}
	return nil
}

func validRotaryDimension(rotary, key, alignment uint32) bool {
	return rotary > 0 && rotary <= key && alignment > 0 && rotary%alignment == 0
}
