package model

import "errors"

func (s Spec) validateEncoderFamilies() error {
	if s.Architecture == "bert" &&
		(s.TokenTypeCount == 0 || s.HeadCountKV != s.HeadCount || s.KeyLength != s.ValueLength) {
		return errors.New("BERT metadata is invalid")
	}
	if s.Architecture == "jina-bert-v2" &&
		(s.TokenTypeCount == 0 || s.KeyLength != s.ValueLength || s.MaxALiBiBias != 8) {
		return errors.New("JinaBERT v2 metadata is invalid")
	}
	if s.Architecture == "jina-bert-v3" &&
		(s.TokenTypeCount == 0 || s.RopeDimensionCount == 0 ||
			s.RopeDimensionCount > s.KeyLength || s.RopeDimensionCount%2 != 0 ||
			s.KeyLength != s.ValueLength ||
			(s.ExpertCount == 0 && (s.ExpertUsedCount != 0 || s.ExpertFeedForward != 0 || s.MoELayerStep != 0)) ||
			(s.ExpertCount > 0 && (s.ExpertUsedCount == 0 || s.ExpertUsedCount > s.ExpertCount ||
				exceedsMoETopK(s.ExpertUsedCount) || s.ExpertFeedForward != s.FeedForwardLength ||
				s.MoELayerStep < 2 || s.ExpertWeightsScale <= 0))) {
		return errors.New("JinaBERT v3 metadata is invalid")
	}
	if s.Architecture == "neo-bert" &&
		(s.RopeDimensionCount != s.KeyLength || s.KeyLength != s.ValueLength ||
			s.RopeDimensionCount%2 != 0) {
		return errors.New("NeoBERT attention metadata is invalid")
	}
	if s.Architecture == "nomic-bert" &&
		(s.TokenTypeCount == 0 || s.RopeDimensionCount == 0 ||
			s.RopeDimensionCount > s.KeyLength || s.RopeDimensionCount%2 != 0 ||
			s.KeyLength != s.ValueLength) {
		return errors.New("NomicBERT metadata is invalid")
	}
	if s.Architecture == "nomic-bert-moe" &&
		(s.TokenTypeCount == 0 || s.MoELayerStep < 2 || s.RopeDimensionCount == 0 ||
			s.RopeDimensionCount > s.KeyLength || s.RopeDimensionCount%2 != 0 ||
			s.KeyLength != s.ValueLength) {
		return errors.New("NomicBERT-MoE metadata is invalid")
	}
	return nil
}
