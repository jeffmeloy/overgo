package model

import "errors"

func (s Spec) validateEncoderFamilies() error {
	switch s.Profile().Validation.Encoder {
	case EncoderValidationBERT:
		if s.TokenTypeCount != 0 && s.HeadCountKV == s.HeadCount && s.KeyLength == s.ValueLength {
			return nil
		}
		return errors.New("BERT metadata is invalid")
	case EncoderValidationJinaV2:
		if s.TokenTypeCount != 0 && s.KeyLength == s.ValueLength && s.MaxALiBiBias == 8 {
			return nil
		}
		return errors.New("JinaBERT v2 metadata is invalid")
	case EncoderValidationJinaV3:
		if !invalidEncoderRotary(s) && validOptionalEncoderExperts(s) {
			return nil
		}
		return errors.New("JinaBERT v3 metadata is invalid")
	case EncoderValidationNeoBERT:
		if s.RopeDimensionCount == s.KeyLength && s.KeyLength == s.ValueLength && s.RopeDimensionCount%2 == 0 {
			return nil
		}
		return errors.New("NeoBERT attention metadata is invalid")
	case EncoderValidationNomicBERT:
		if !invalidEncoderRotary(s) {
			return nil
		}
		return errors.New("NomicBERT metadata is invalid")
	case EncoderValidationNomicBERTMoE:
		if !invalidEncoderRotary(s) && s.MoELayerStep >= 2 {
			return nil
		}
		return errors.New("NomicBERT-MoE metadata is invalid")
	}
	return nil
}

func invalidEncoderRotary(s Spec) bool {
	return s.TokenTypeCount == 0 || s.RopeDimensionCount == 0 ||
		s.RopeDimensionCount > s.KeyLength || s.RopeDimensionCount%2 != 0 ||
		s.KeyLength != s.ValueLength
}

func validOptionalEncoderExperts(s Spec) bool {
	if s.ExpertCount == 0 {
		return s.ExpertUsedCount == 0 && s.ExpertFeedForward == 0 && s.MoELayerStep == 0
	}
	return s.ExpertUsedCount > 0 && s.ExpertUsedCount <= s.ExpertCount &&
		!exceedsMoETopK(s.ExpertUsedCount) && s.ExpertFeedForward == s.FeedForwardLength &&
		s.MoELayerStep >= 2 && s.ExpertWeightsScale > 0
}
