package model

import (
	"errors"
	"math"

	"overgo/internal/gguf"
	"overgo/internal/hostmath"
	"overgo/internal/tensor"
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
	RopeFrequencyBase          float32
	RopeFrequencySWA           float32
	YaRNBetaFast               float32
	RopeDisabled               bool
	OriginalContext            bool
	RopeFrequencyFromBase      bool
	RopeDimension              RopeDimensionDefaultPolicy
	DecoderBlocksFromModel     bool
	FullAttentionInterval      uint32
	NoRopeLayerStep            uint32
	MoELayerStep               uint32
	ExpertChunkFromKey         bool
	SharedExpert               SharedExpertDefaultPolicy
}

func (p MetadataDefaultPolicy) uint(values map[string]gguf.Value, prefix, key string, fallback uint32) uint32 {
	return optionalOr(values, prefix+key, gguf.ValueTypeUint32, fallback)
}

func (p MetadataDefaultPolicy) readSharedExpertFeedForward(
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
		if spec.ExpertFeedForward == tensor.FirstOffset || spec.SharedExpertCount > math.MaxUint32/spec.ExpertFeedForward {
			return 0, errors.New("shared expert feed-forward extent overflows")
		}
		fallback = spec.ExpertFeedForward * spec.SharedExpertCount
	default:
		return 0, errors.New("shared expert feed-forward relationship is absent")
	}
	return p.uint(values, prefix, "expert_shared_feed_forward_length", fallback), nil
}

func derivedExpertFeedForward(spec Spec) (uint32, error) {
	if spec.ExpertUsedCount == tensor.FirstOffset || spec.FeedForwardLength%spec.ExpertUsedCount != tensor.FirstOffset {
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
	if positiveFinite(p.AttentionSoftcap) {
		spec.AttentionSoftcap = readFloat("attn_logit_softcapping", p.AttentionSoftcap)
	}
	if positiveFinite(p.AttentionOutputScale) {
		spec.AttentionScale = readFloat("attention.output_scale", p.AttentionOutputScale)
	}
	if positiveFinite(p.EmbeddingScale) {
		spec.EmbeddingScale = readFloat("embedding_scale", p.EmbeddingScale)
	}
	if positiveFinite(p.LogitScale) {
		spec.LogitScale = readFloat("logit_scale", p.LogitScale)
	}
	if positiveFinite(p.ResidualScalePerSqrtBlock) {
		spec.ResidualScale = readFloat(
			"residual_scale", p.ResidualScalePerSqrtBlock/hostmath.Sqrt32(uint64(spec.BlockCount)),
		)
	}
	if positiveFinite(p.LogitScaleNumerator) {
		spec.LogitScale = readFloat("logit_scale", p.LogitScaleNumerator/float32(spec.EmbeddingLength))
	}
	if positiveFinite(p.LayerNormEpsilon) && spec.LayerNormEpsilon == tensor.FirstOffset {
		spec.LayerNormEpsilon = p.LayerNormEpsilon
	}
	if positiveFinite(p.QKNormEpsilon) && spec.QKNormEpsilon == tensor.FirstOffset {
		spec.QKNormEpsilon = p.QKNormEpsilon
	}
	if p.SlidingWindow > tensor.FirstOffset {
		spec.SlidingWindow = p.uint(values, prefix, "attention.sliding_window", p.SlidingWindow)
	}
	if p.SlidingPattern > tensor.FirstOffset {
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
	if positiveFinite(p.RopeAttentionFactor) {
		spec.RopeAttentionFactor = readFloat("rope.scaling.attn_factor", p.RopeAttentionFactor)
	}
	if p.RopeFrequencyFromBase || positiveFinite(p.RopeFrequencySWA) {
		fallback := p.RopeFrequencySWA
		if p.RopeFrequencyFromBase {
			fallback = spec.RopeFrequencyBase
		}
		spec.RopeFrequencySWA = readFloat("rope.freq_base_swa", fallback)
	}
	if positiveFinite(p.AttentionTemperatureScale) {
		if spec.SlidingWindow == tensor.FirstOffset {
			spec.NoRopeLayerStep = tensor.FirstOffset
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
