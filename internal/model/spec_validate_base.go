package model

import (
	"errors"
	"fmt"
	"math"
)

func (s Spec) validateBaseMetadata() error {
	if (s.Architecture == "mpt" || s.Architecture == "olmo") &&
		!nonNegativeFinite(s.AttentionClamp) {
		return fmt.Errorf("%s attention QKV clamp is invalid", s.Architecture)
	}
	attentionFree := s.Architecture == "mamba" || s.Architecture == "mamba2"
	switch {
	case s.BlockCount == 0:
		return errors.New("model block count is zero")
	case s.ContextLength == 0:
		return errors.New("model context length is zero")
	case s.EmbeddingLength == 0:
		return errors.New("model embedding length is zero")
	case !attentionFree && s.FeedForwardLength == 0:
		return errors.New("model feed-forward length is zero")
	case !attentionFree && s.HeadCount == 0:
		return errors.New("model attention head count is zero")
	case !attentionFree && s.HeadCountKV == 0:
		return errors.New("model KV head count is zero")
	case !attentionFree && s.HeadCount%s.HeadCountKV != 0:
		return errors.New("attention head count is not divisible by KV head count")
	case !attentionFree && (s.KeyLength == 0 || s.ValueLength == 0):
		return errors.New("model attention key/value length is zero")
	case s.Architecture == "gptj" && !validRotaryDimension(s.RopeDimensionCount, s.KeyLength, 2):
		return errors.New("GPT-J rotary dimension count is invalid")
	case s.Architecture != "t5encoder" && !s.RopeDisabled && s.RopeFrequencyBase <= 0:
		return errors.New("model RoPE frequency base must be positive")
	case (s.UsesLayerNorm() || s.UsesWeightOnlyLayerNorm() || s.UsesUnweightedLayerNorm()) && s.LayerNormEpsilon <= 0:
		return errors.New("model LayerNorm epsilon must be positive")
	case !s.UsesLayerNorm() && !s.UsesWeightOnlyLayerNorm() && !s.UsesUnweightedLayerNorm() && s.RMSNormEpsilon <= 0:
		return errors.New("model RMSNorm epsilon must be positive")
	}
	if (s.Architecture == "t5" || s.Architecture == "t5encoder") && s.RelativeBuckets == 0 {
		return errors.New("T5 encoder relative attention bucket count is zero")
	}
	if s.Architecture == "t5" && s.DecoderBlockCount == 0 {
		return errors.New("T5 decoder block count is zero")
	}
	return nil
}

func (s Spec) validateNumericPolicies() error {
	for _, field := range []struct {
		name  string
		value float32
	}{
		{"final logit softcap", s.FinalLogitSoftcap},
		{"attention logit softcap", s.AttentionSoftcap},
		{"attention scale", s.AttentionScale},
		{"maximum ALiBi bias", s.MaxALiBiBias},
		{"embedding scale", s.EmbeddingScale},
		{"residual scale", s.ResidualScale},
	} {
		if !nonNegativeFinite(field.value) {
			return fmt.Errorf("%s must be finite and non-negative", field.name)
		}
	}
	logitScaleRequired := s.Architecture == "minicpm" || s.Architecture == "granite" ||
		s.Architecture == "granitemoe" || s.Architecture == "talkie"
	if !nonNegativeFinite(s.LogitScale) || (logitScaleRequired && s.LogitScale == 0) {
		return errors.New("logit scale must be finite and positive when required")
	}
	return nil
}

func finite(value float32) bool {
	return !math.IsNaN(float64(value)) && !math.IsInf(float64(value), 0)
}

func positiveFinite(value float32) bool {
	return value > 0 && finite(value)
}

func nonNegativeFinite(value float32) bool {
	return value >= 0 && finite(value)
}
