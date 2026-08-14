package model

import "overgo/internal/gguf"

// RopeDimensionDefaultPolicy: missing rotary-width relationship.
type RopeDimensionDefaultPolicy uint8

const (
	RopeDimensionDefaultNone RopeDimensionDefaultPolicy = iota
	RopeDimensionDefaultKeyLength
	RopeDimensionDefaultKeyLengthOverride
)

// MetadataDefaultPolicy: serialized architecture defaults and override keys.
type MetadataDefaultPolicy struct {
	AttentionSoftcap       float32
	AttentionOutputScale   float32
	EmbeddingScale         float32
	LogitScale             float32
	LayerNormEpsilon       float32
	QKNormEpsilon          float32
	AlternateStateCount    uint32
	AlternateStateActive   uint32
	LowRankResidualWidth   uint32
	PerLayerEmbeddingWidth uint32
	SharedKVStartLayer     uint32
	SparseLayerCount       uint32
	SparsityStdMultiplier  float32
	DraftBlockSize         uint32
	SlidingWindow          uint32
	SlidingPattern         uint32
	MaxALiBiBias           float32
	RopeDisabled           bool
	RopeDimension          RopeDimensionDefaultPolicy
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
	if p.LayerNormEpsilon > 0 && spec.LayerNormEpsilon == 0 {
		spec.LayerNormEpsilon = p.LayerNormEpsilon
	}
	if p.QKNormEpsilon > 0 && spec.QKNormEpsilon == 0 {
		spec.QKNormEpsilon = p.QKNormEpsilon
	}
	if p.SlidingWindow > 0 {
		spec.SlidingWindow = p.SlidingWindow
	}
	if p.SlidingPattern > 0 {
		spec.SlidingPattern = p.SlidingPattern
	}
	if p.RopeDimension != RopeDimensionDefaultNone {
		spec.RopeDimensionCount = spec.KeyLength
		if p.RopeDimension == RopeDimensionDefaultKeyLengthOverride {
			if value, ok := optional[uint32](values, prefix+"rope.dimension_count", gguf.ValueTypeUint32); ok {
				spec.RopeDimensionCount = value
			}
		}
	}
}

func (p MetadataDefaultPolicy) readPosition(spec *Spec) {
	if p.RopeDisabled {
		spec.RopeDisabled = true
		spec.MaxALiBiBias = p.MaxALiBiBias
	}
}
