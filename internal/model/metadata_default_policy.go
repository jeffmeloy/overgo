package model

import (
	"errors"
	"math"

	"overgo/internal/gguf"
)

// RopeDimensionDefaultPolicy: missing rotary-width relationship.
type RopeDimensionDefaultPolicy uint8

const (
	RopeDimensionDefaultNone RopeDimensionDefaultPolicy = iota
	RopeDimensionDefaultKeyLength
	RopeDimensionDefaultKeyLengthOverride
)

// SharedExpertDefaultPolicy: omitted shared-expert width relationship.
type SharedExpertDefaultPolicy uint8

const (
	SharedExpertDefaultNone SharedExpertDefaultPolicy = iota
	SharedExpertDefaultModelFeedForward
	SharedExpertDefaultExpertFeedForward
	SharedExpertDefaultExpertProduct
)

// MetadataDefaultPolicy: serialized architecture defaults and override keys.
type MetadataDefaultPolicy struct {
	AttentionSoftcap           float32
	AttentionOutputScale       float32
	EmbeddingScale             float32
	LogitScale                 float32
	ResidualScalePerSqrtBlock  float32
	LogitScaleNumerator        float32
	LayerNormEpsilon           float32
	QKNormEpsilon              float32
	AlternateStateCount        uint32
	AlternateStateActive       uint32
	LowRankResidualWidth       uint32
	PerLayerEmbeddingWidth     uint32
	SharedKVStartLayer         uint32
	SparseLayerCount           uint32
	SparsityStdMultiplier      float32
	DraftBlockSize             uint32
	SlidingWindow              uint32
	SlidingPattern             uint32
	AttentionTemperatureScale  float32
	AttentionTemperatureOffset float32
	MaxALiBiBias               float32
	RopeAttentionFactor        float32
	RopeFrequencySWA           float32
	RopeDisabled               bool
	OriginalContext            bool
	RopeFrequencyFromBase      bool
	RopeDimension              RopeDimensionDefaultPolicy
	DecoderBlocksFromModel     bool
	FullAttentionInterval      uint32
	MoELayerStep               uint32
	ExpertChunkFromKey         bool
	SharedExpert               SharedExpertDefaultPolicy
}

func (p MetadataDefaultPolicy) uint(values map[string]gguf.Value, prefix, key string, fallback uint32) uint32 {
	return optionalOr(values, prefix+key, gguf.ValueTypeUint32, fallback)
}

func (p MetadataDefaultPolicy) readSharedExpertWidth(
	values map[string]gguf.Value,
	prefix string,
	spec Spec,
) (uint32, error) {
	var fallback uint32
	switch p.SharedExpert {
	case SharedExpertDefaultModelFeedForward:
		fallback = spec.FeedForwardLength
	case SharedExpertDefaultExpertFeedForward:
		fallback = spec.ExpertFeedForward
	case SharedExpertDefaultExpertProduct:
		if spec.ExpertFeedForward == 0 || spec.SharedExpertCount > math.MaxUint32/spec.ExpertFeedForward {
			return 0, errors.New("shared expert width overflows")
		}
		fallback = spec.ExpertFeedForward * spec.SharedExpertCount
	default:
		return 0, errors.New("shared expert width relationship is absent")
	}
	return p.uint(values, prefix, "expert_shared_feed_forward_length", fallback), nil
}

func derivedExpertFeedForward(spec Spec) (uint32, error) {
	if spec.ExpertUsedCount == 0 || spec.FeedForwardLength%spec.ExpertUsedCount != 0 {
		return 0, errors.New("expert feed-forward width is not exactly derivable")
	}
	return spec.FeedForwardLength / spec.ExpertUsedCount, nil
}

func (p MetadataDefaultPolicy) read(values map[string]gguf.Value, prefix string, spec *Spec) {
	readFloat := func(key string, fallback float32) float32 {
		if value, ok := optional[float32](values, prefix+key, gguf.ValueTypeFloat32); ok {
			return value
		}
		return fallback
	}
	if p.AttentionSoftcap > 0 {
		spec.AttentionSoftcap = readFloat("attn_logit_softcapping", p.AttentionSoftcap)
	}
	if p.AttentionOutputScale > 0 {
		spec.AttentionScale = readFloat("attention.output_scale", p.AttentionOutputScale)
	}
	if p.EmbeddingScale > 0 {
		spec.EmbeddingScale = readFloat("embedding_scale", p.EmbeddingScale)
	}
	if p.LogitScale > 0 {
		spec.LogitScale = readFloat("logit_scale", p.LogitScale)
	}
	if positiveFinite(p.ResidualScalePerSqrtBlock) {
		spec.ResidualScale = readFloat(
			"residual_scale", p.ResidualScalePerSqrtBlock/float32(math.Sqrt(float64(spec.BlockCount))),
		)
	}
	if positiveFinite(p.LogitScaleNumerator) {
		spec.LogitScale = readFloat("logit_scale", p.LogitScaleNumerator/float32(spec.EmbeddingLength))
	}
	if p.LayerNormEpsilon > 0 && spec.LayerNormEpsilon == 0 {
		spec.LayerNormEpsilon = p.LayerNormEpsilon
	}
	if p.QKNormEpsilon > 0 && spec.QKNormEpsilon == 0 {
		spec.QKNormEpsilon = p.QKNormEpsilon
	}
	if p.SlidingWindow > 0 {
		spec.SlidingWindow = p.uint(values, prefix, "attention.sliding_window", p.SlidingWindow)
	}
	if p.SlidingPattern > 0 {
		spec.SlidingPattern = optionalOr(
			values, prefix+"attention.sliding_window_pattern", gguf.ValueTypeUint32, p.SlidingPattern,
		)
	}
	if p.RopeDimension != RopeDimensionDefaultNone {
		spec.RopeDimensionCount = spec.KeyLength
		if p.RopeDimension == RopeDimensionDefaultKeyLengthOverride {
			if value, ok := optional[uint32](values, prefix+"rope.dimension_count", gguf.ValueTypeUint32); ok {
				spec.RopeDimensionCount = value
			}
		}
	}
	if p.OriginalContext {
		spec.OriginalContextLength = optionalOr(
			values, prefix+"rope.scaling.original_context_length", gguf.ValueTypeUint32, spec.ContextLength,
		)
	}
	if p.RopeAttentionFactor > 0 {
		spec.RopeAttentionFactor = readFloat("rope.scaling.attn_factor", p.RopeAttentionFactor)
	}
	if p.RopeFrequencyFromBase || p.RopeFrequencySWA > 0 {
		fallback := p.RopeFrequencySWA
		if p.RopeFrequencyFromBase {
			fallback = spec.RopeFrequencyBase
		}
		spec.RopeFrequencySWA = readFloat("rope.freq_base_swa", fallback)
	}
	if positiveFinite(p.AttentionTemperatureScale) {
		if spec.SlidingWindow == 0 {
			spec.NoRopeLayerStep = 0
		} else {
			spec.NoRopeLayerStep = spec.SlidingPattern
			spec.AttentionTempFloor = spec.SlidingWindow
			spec.AttentionTempScale = p.AttentionTemperatureScale
			spec.AttentionTempOffset = p.AttentionTemperatureOffset
		}
	}
}

func (p MetadataDefaultPolicy) readPosition(spec *Spec) {
	if p.RopeDisabled {
		spec.RopeDisabled = true
		spec.MaxALiBiBias = p.MaxALiBiBias
	}
}
