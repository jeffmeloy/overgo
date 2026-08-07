package model

import (
	"slices"

	"overgo/internal/gguf"
	"overgo/internal/tensor"
)

// TensorShapePlan: layer tensor and cache dimensions.
type TensorShapePlan struct {
	Embedding   uint64
	FeedForward uint64
	QueryHeads  uint64
	KVHeads     uint64
	Key         uint64
	Value       uint64
	Experts     uint64
	ExpertWidth uint64
}

func (s Spec) TensorShapes(layer uint32) TensorShapePlan {
	return TensorShapePlan{
		Embedding: uint64(s.EmbeddingLength), FeedForward: uint64(s.LayerFeedForwardLength(layer)),
		QueryHeads: uint64(s.LayerHeadCount(layer)), KVHeads: uint64(s.LayerKVHeadCount(layer)),
		Key: uint64(s.LayerKeyLength(layer)), Value: uint64(s.LayerValueLength(layer)),
		Experts: uint64(s.ExpertCount), ExpertWidth: uint64(s.ExpertFeedForward),
	}
}

func (p TensorShapePlan) EmbeddingVector() []uint64 {
	return []uint64{p.Embedding}
}

func (p TensorShapePlan) QueryProjectionWidth() uint64 {
	return p.QueryHeads * p.Key
}

func (p TensorShapePlan) KeyProjectionWidth() uint64 {
	return p.KVHeads * p.Key
}

func (p TensorShapePlan) ValueProjectionWidth() uint64 {
	return p.KVHeads * p.Value
}

func (p TensorShapePlan) AttentionOutputWidth() uint64 {
	return p.QueryHeads * p.Value
}

func (p TensorShapePlan) FeedForwardUp(multiplier uint64) []uint64 {
	return []uint64{p.Embedding, multiplier * p.FeedForward}
}

func (p TensorShapePlan) FeedForwardDown() []uint64 {
	return []uint64{p.FeedForward, p.Embedding}
}

func (p TensorShapePlan) ExpertRouter() []uint64 {
	return []uint64{p.Embedding, p.Experts}
}

func (p TensorShapePlan) ExpertUp(multiplier uint64) []uint64 {
	return []uint64{p.Embedding, multiplier * p.ExpertWidth, p.Experts}
}

func (p TensorShapePlan) ExpertDown() []uint64 {
	return []uint64{p.ExpertWidth, p.Embedding, p.Experts}
}

func (p TensorShapePlan) KeyCache(tokens uint32) tensor.Shape {
	return tensor.MustShape(p.Key, p.KVHeads, uint64(tokens))
}

func (p TensorShapePlan) ValueCache(tokens uint32) tensor.Shape {
	return tensor.MustShape(p.Value, p.KVHeads, uint64(tokens))
}

func tensorInfoMatches(info gguf.TensorInfo, shape []uint64) bool {
	return info.Dimensions <= uint32(len(info.Shape)) && int(info.Dimensions) == len(shape) &&
		slices.Equal(info.Shape[:info.Dimensions], shape)
}
