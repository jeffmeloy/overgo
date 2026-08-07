package model

import "llamacpp2go/internal/gguf"

// RopeDimensionDefaultPolicy: missing rotary-width relationship.
type RopeDimensionDefaultPolicy uint8

const (
	RopeDimensionDefaultNone RopeDimensionDefaultPolicy = iota
	RopeDimensionDefaultKeyLength
	RopeDimensionDefaultKeyLengthOverride
)

// MetadataDefaultPolicy: serialized architecture defaults and override keys.
type MetadataDefaultPolicy struct {
	AttentionSoftcap     float32
	AttentionOutputScale float32
	EmbeddingScale       float32
	LogitScale           float32
	RopeDimension        RopeDimensionDefaultPolicy
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
	if p.RopeDimension != RopeDimensionDefaultNone {
		spec.RopeDimensionCount = spec.KeyLength
		if p.RopeDimension == RopeDimensionDefaultKeyLengthOverride {
			if value, ok := optional[uint32](values, prefix+"rope.dimension_count", gguf.ValueTypeUint32); ok {
				spec.RopeDimensionCount = value
			}
		}
	}
}
