package model

import (
	"errors"

	"overgo/internal/tensor"
)

func (s Spec) validateEncoderMetadata() error {
	switch s.Profile().Validation.Encoder {
	case EncoderValidationTokenTypesMatchingHeads:
		if s.TokenTypeCount > tensor.FirstOffset && s.HeadCountKV == s.HeadCount && s.KeyLength == s.ValueLength {
			return nil
		}
		return errors.New("BERT metadata is invalid")
	case EncoderValidationTokenTypesALiBi:
		if s.TokenTypeCount > tensor.FirstOffset && s.KeyLength == s.ValueLength &&
			s.MaxALiBiBias == s.Profile().MetadataDefaults.MaxALiBiBias {
			return nil
		}
		return errors.New("JinaBERT v2 metadata is invalid")
	case EncoderValidationRotaryOptionalExperts:
		if validEncoderRotary(s) && validOptionalEncoderExperts(s) {
			return nil
		}
		return errors.New("JinaBERT v3 metadata is invalid")
	case EncoderValidationFullRotary:
		if s.RopeDimensionCount == s.KeyLength && s.KeyLength == s.ValueLength &&
			validRotaryDimension(s.RopeDimensionCount, s.KeyLength) {
			return nil
		}
		return errors.New("NeoBERT attention metadata is invalid")
	case EncoderValidationRotary:
		if validEncoderRotary(s) {
			return nil
		}
		return errors.New("NomicBERT metadata is invalid")
	case EncoderValidationRotaryPeriodicExperts:
		if validEncoderRotary(s) && s.MoELayerStep > tensor.SingletonExtent {
			return nil
		}
		return errors.New("NomicBERT-MoE metadata is invalid")
	}
	return nil
}

func validEncoderRotary(s Spec) bool {
	return s.TokenTypeCount > tensor.FirstOffset &&
		validRotaryDimension(s.RopeDimensionCount, s.KeyLength) &&
		s.KeyLength == s.ValueLength
}

func validOptionalEncoderExperts(s Spec) bool {
	if !s.HasExperts() {
		return s.ExpertUsedCount == tensor.FirstOffset && s.ExpertFeedForward == tensor.FirstOffset &&
			s.MoELayerStep == tensor.FirstOffset
	}
	return validMoESelection(s.ExpertUsedCount, s.ExpertCount) &&
		s.ExpertFeedForward == s.FeedForwardLength &&
		s.MoELayerStep > tensor.SingletonExtent && positiveFinite(s.ExpertWeightsScale)
}
