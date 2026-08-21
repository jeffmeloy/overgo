package model

import (
	"errors"
	"fmt"

	"overgo/internal/tensor"
)

const (
	minimumConvKernelWidth = uint32(tensor.PairedExtent)
	rotaryPairAlignment    = uint32(tensor.PairedExtent)
)

func (s Spec) validateRecurrentMetadata() error {
	validation := s.Profile().Validation.Recurrent
	switch validation {
	case RecurrentValidationAudioDecoder:
		if s.OutputEmbeddingLength == tensor.FirstOffset || s.PosNetBlockCount == tensor.FirstOffset ||
			s.ConvNextBlockCount == tensor.FirstOffset || s.PosNetEmbeddingLength == tensor.FirstOffset ||
			s.PosNetEmbeddingLength != s.ConvNextEmbeddingLength || s.GroupNormGroups == tensor.FirstOffset ||
			!positiveFinite(s.GroupNormEpsilon) || s.PosNetEmbeddingLength%s.GroupNormGroups != tensor.FirstOffset {
			return errors.New("AudioDecoder metadata is invalid")
		}
	case RecurrentValidationTargetLayerBlock:
		if len(s.TargetLayers) == tensor.FirstOffset || s.DFlashBlockSize == tensor.FirstOffset {
			return errors.New("target-layer block metadata is invalid")
		}
		return validateNonNegativeTargetLayers("target-layer block", s.TargetLayers)
	case RecurrentValidationSingleBlockTarget:
		if s.BlockCount == tensor.FirstOffset || len(s.TargetLayers) == tensor.FirstOffset ||
			s.TargetHiddenSize == tensor.FirstOffset {
			return errors.New("feature-draft target-layer metadata is invalid")
		}
		return validateNonNegativeTargetLayers("feature-draft", s.TargetLayers)
	case RecurrentValidationUngroupedStateSpace:
		if !attentionMetadataZero(s) {
			return errors.New("Mamba attention metadata must be zero")
		}
		if s.SSMConvKernel < minimumConvKernelWidth || !s.Profile().Runtime.Recurrent.validInnerWidth(s) ||
			s.SSMStateSize == tensor.FirstOffset || s.SSMTimeStepRank == tensor.FirstOffset {
			return errors.New("Mamba SSM metadata is invalid")
		}
	case RecurrentValidationGroupedStateSpace:
		if !attentionMetadataZero(s) {
			return errors.New("Mamba2 attention metadata must be zero")
		}
		if !validGroupedSSM(s) {
			return errors.New("Mamba2 SSM metadata is invalid")
		}
	case RecurrentValidationGroupedStateSpaceAttention:
		if !validGroupedSSM(s) {
			return errors.New("Falcon-H1 SSM metadata is invalid")
		}
		if !validRotaryDimension(s.RopeDimensionCount, s.KeyLength) || s.KeyLength != s.ValueLength {
			return errors.New("Falcon-H1 rotary/head dimensions are invalid")
		}
	case RecurrentValidationTimeMixV6, RecurrentValidationTimeMixV6SharedKV:
		return s.validateTimeMixV6(validation)
	case RecurrentValidationTimeMixV7Gated, RecurrentValidationTimeMixV7:
		return s.validateTimeMixV7(validation)
	case RecurrentValidationStateSpaceAttentionExperts:
		if s.SSMConvKernel < minimumConvKernelWidth || !s.Profile().Runtime.Recurrent.validInnerWidth(s) ||
			s.SSMStateSize == tensor.FirstOffset || s.SSMTimeStepRank == tensor.FirstOffset {
			return errors.New("Jamba SSM metadata is invalid")
		}
		if !validRecurrentLayerSchedule(s) {
			return errors.New("Jamba layer schedule is invalid")
		}
		if !validMoESelection(s.ExpertUsedCount, s.ExpertCount) ||
			s.ExpertFeedForward == tensor.FirstOffset || !positiveFinite(s.ExpertWeightsScale) {
			return errors.New("Jamba expert metadata is invalid")
		}
	case RecurrentValidationGroupedStateSpaceOptionalExperts:
		return s.validateGroupedStateSpaceOptionalExperts()
	case RecurrentValidationUngroupedScheduledStateSpace:
		if s.SSMConvKernel < minimumConvKernelWidth || s.SSMInnerSize == tensor.FirstOffset ||
			s.SSMStateSize == tensor.FirstOffset || s.SSMTimeStepRank == tensor.FirstOffset ||
			s.SSMGroupCount != tensor.FirstOffset || s.SSMInnerSize%s.SSMTimeStepRank != tensor.FirstOffset {
			return errors.New("PLaMo2 SSM metadata is invalid")
		}
		if !validRecurrentLayerSchedule(s) {
			return errors.New("PLaMo2 layer schedule is invalid")
		}
	case RecurrentValidationScheduledStateSpaceDense, RecurrentValidationScheduledStateSpaceExperts:
		return s.validateScheduledStateSpace(validation == RecurrentValidationScheduledStateSpaceExperts)
	}
	return nil
}

func attentionMetadataZero(s Spec) bool {
	return s.FeedForwardLength == tensor.FirstOffset && s.HeadCount == tensor.FirstOffset &&
		s.HeadCountKV == tensor.FirstOffset && s.KeyLength == tensor.FirstOffset &&
		s.ValueLength == tensor.FirstOffset
}

func validGroupedSSM(s Spec) bool {
	return s.SSMConvKernel >= minimumConvKernelWidth && s.SSMInnerSize > tensor.FirstOffset &&
		s.SSMStateSize > tensor.FirstOffset && s.SSMTimeStepRank > tensor.FirstOffset &&
		s.SSMGroupCount > tensor.FirstOffset && s.SSMInnerSize%s.SSMTimeStepRank == tensor.FirstOffset &&
		s.SSMInnerSize%s.SSMGroupCount == tensor.FirstOffset &&
		s.SSMTimeStepRank%s.SSMGroupCount == tensor.FirstOffset
}

func validStateWidthGroupedSSM(s Spec) bool {
	return s.SSMInnerSize > tensor.FirstOffset && s.SSMStateSize > tensor.FirstOffset &&
		s.SSMTimeStepRank > tensor.FirstOffset && s.SSMGroupCount > tensor.FirstOffset &&
		s.SSMInnerSize%s.SSMTimeStepRank == tensor.FirstOffset &&
		s.SSMTimeStepRank%s.SSMGroupCount == tensor.FirstOffset &&
		s.SSMInnerSize/s.SSMTimeStepRank == s.SSMStateSize
}

func validRecurrentLayerSchedule(s Spec) bool {
	return len(s.RecurrentLayers) == int(s.BlockCount) &&
		len(s.LayerKVHeadCounts) == int(s.BlockCount)
}

func validMixedLayerSchedule(layers []bool, count uint32) bool {
	if len(layers) != int(count) {
		return false
	}
	var first, second bool
	for _, item := range layers {
		first = first || item
		second = second || !item
	}
	return first && second
}

func validateNonNegativeTargetLayers(label string, layers []int32) error {
	for _, layer := range layers {
		if layer < tensor.FirstOffset {
			return fmt.Errorf("%s target layer is negative", label)
		}
	}
	return nil
}

func (s Spec) validateTimeMixV6(validation RecurrentValidationPolicy) error {
	wantShifts := s.Profile().Runtime.Recurrent.TokenShiftCount
	switch {
	case s.WKVHeadSize == tensor.FirstOffset || s.EmbeddingLength%s.WKVHeadSize != tensor.FirstOffset ||
		s.HeadCount != s.EmbeddingLength/s.WKVHeadSize:
		return fmt.Errorf("%s WKV head metadata is invalid", s.Architecture)
	case s.TimeMixExtraDim == tensor.FirstOffset || s.TimeDecayExtraDim == tensor.FirstOffset ||
		s.TokenShiftCount != wantShifts:
		return fmt.Errorf("%s time-mix metadata is invalid", s.Architecture)
	case s.KeyLength != s.WKVHeadSize || s.ValueLength != s.WKVHeadSize:
		return fmt.Errorf("%s head dimensions are invalid", s.Architecture)
	case validation == RecurrentValidationTimeMixV6 && s.HeadCountKV != s.HeadCount:
		return errors.New("rwkv6 KV head count must equal query head count")
	}
	return nil
}

func (s Spec) validateTimeMixV7(validation RecurrentValidationPolicy) error {
	wantShifts := s.Profile().Runtime.Recurrent.TokenShiftCount
	switch {
	case s.WKVHeadSize == tensor.FirstOffset || s.EmbeddingLength%s.WKVHeadSize != tensor.FirstOffset ||
		s.HeadCount != s.EmbeddingLength/s.WKVHeadSize:
		return fmt.Errorf("%s WKV head metadata is invalid", s.Architecture)
	case s.HeadCountKV != s.HeadCount || s.KeyLength != s.WKVHeadSize || s.ValueLength != s.WKVHeadSize:
		return fmt.Errorf("%s head dimensions are invalid", s.Architecture)
	case s.DecayLoRARank == tensor.FirstOffset || s.ICLRLoRARank == tensor.FirstOffset ||
		s.ValueMixLoRARank == tensor.FirstOffset:
		return fmt.Errorf("%s time-mix LoRA metadata is invalid", s.Architecture)
	case validation == RecurrentValidationTimeMixV7Gated && s.GateLoRARank == tensor.FirstOffset:
		return errors.New("rwkv7 gate LoRA metadata is invalid")
	case s.TokenShiftCount != wantShifts:
		return fmt.Errorf("%s token-shift metadata is invalid", s.Architecture)
	}
	return nil
}

func (s Spec) validateGroupedStateSpaceOptionalExperts() error {
	if !validGroupedSSM(s) || !s.Profile().Runtime.Recurrent.validInnerWidth(s) {
		return errors.New("optional-expert grouped state-space metadata is invalid")
	}
	if !validRecurrentLayerSchedule(s) {
		return errors.New("optional-expert grouped state-space schedule is invalid")
	}
	if s.HasExperts() && (!validMoESelection(s.ExpertUsedCount, s.ExpertCount) ||
		s.ExpertFeedForward == tensor.FirstOffset || !positiveFinite(s.ExpertWeightsScale)) {
		return errors.New("optional-expert grouped state-space expert metadata is invalid")
	}
	if !s.HasExperts() && (s.ExpertUsedCount != tensor.FirstOffset ||
		s.ExpertFeedForward != tensor.FirstOffset || s.SharedExpertFF != tensor.FirstOffset) {
		return errors.New("dense grouped state-space expert metadata is inconsistent")
	}
	return nil
}

func (s Spec) validateScheduledStateSpace(moe bool) error {
	if !validGroupedSSM(s) {
		return errors.New("Nemotron-H SSM metadata is invalid")
	}
	if len(s.LayerHeadCounts) != int(s.BlockCount) || len(s.LayerKVHeadCounts) != int(s.BlockCount) ||
		len(s.LayerFeedForward) != int(s.BlockCount) || len(s.RecurrentLayers) != int(s.BlockCount) {
		return errors.New("Nemotron-H layer schedule is invalid")
	}
	for block := uint32(tensor.FirstOffset); block < s.BlockCount; block++ {
		heads, kvHeads, ff := s.LayerHeadCount(block), s.LayerKVHeadCount(block), s.LayerFeedForwardLength(block)
		if ff > tensor.FirstOffset {
			continue
		}
		if s.IsRecurrentLayer(block) {
			if heads != tensor.FirstOffset || kvHeads != tensor.FirstOffset {
				return errors.New("Nemotron-H recurrent layer schedule is invalid")
			}
			continue
		}
		if heads == tensor.FirstOffset || kvHeads == tensor.FirstOffset || heads%kvHeads != tensor.FirstOffset {
			return errors.New("Nemotron-H attention layer schedule is invalid")
		}
	}
	if moe {
		if !validMoESelection(s.ExpertUsedCount, s.ExpertCount) ||
			s.ExpertFeedForward == tensor.FirstOffset || s.SharedExpertFF == tensor.FirstOffset ||
			!positiveFinite(s.ExpertWeightsScale) {
			return errors.New("Nemotron-H MoE metadata is invalid")
		}
	} else if s.ExpertCount != tensor.FirstOffset || s.ExpertUsedCount != tensor.FirstOffset ||
		s.ExpertFeedForward != tensor.FirstOffset || s.SharedExpertFF != tensor.FirstOffset ||
		s.MoELatentSize != tensor.FirstOffset {
		return errors.New("dense Nemotron-H expert metadata is inconsistent")
	}
	return nil
}

func validRotaryDimension(rotary, key uint32) bool {
	return rotary > tensor.FirstOffset && rotary <= key && rotary%rotaryPairAlignment == tensor.FirstOffset
}
