package model

import "overgo/internal/tensor"

func (s Spec) ssmConvolutionWindow() uint64 {
	if s.SSMConvKernel < tensor.SingletonExtent {
		return tensor.FirstOffset
	}
	return uint64(s.SSMConvKernel) - tensor.SingletonExtent
}

// tensorShapePlan: layer tensor and cache dimensions.
type tensorShapePlan struct {
	Embedding   uint64
	FeedForward uint64
	QueryHeads  uint64
	KVHeads     uint64
	Key         uint64
	Value       uint64
	Experts     uint64
	ExpertWidth uint64
}

func (s Spec) TensorShapes(layer uint32) tensorShapePlan {
	return tensorShapePlan{
		Embedding: uint64(s.EmbeddingLength), FeedForward: uint64(s.LayerFeedForwardLength(layer)),
		QueryHeads: uint64(s.LayerHeadCount(layer)), KVHeads: uint64(s.LayerKVHeadCount(layer)),
		Key: uint64(s.LayerKeyLength(layer)), Value: uint64(s.LayerValueLength(layer)),
		Experts: uint64(s.ExpertCount), ExpertWidth: uint64(s.ExpertFeedForward),
	}
}

func (p tensorShapePlan) EmbeddingVector() []uint64 {
	return []uint64{p.Embedding}
}

func (p tensorShapePlan) QueryProjectionWidth() uint64 {
	return p.QueryHeads * p.Key
}

func (p tensorShapePlan) KeyProjectionWidth() uint64 {
	return p.KVHeads * p.Key
}

func (p tensorShapePlan) ValueProjectionWidth() uint64 {
	return p.KVHeads * p.Value
}

func (p tensorShapePlan) AttentionOutputWidth() uint64 {
	return p.QueryHeads * p.Value
}

func (p tensorShapePlan) FeedForwardUp(multiplier uint64) []uint64 {
	return []uint64{p.Embedding, p.FeedForwardUpWidth(multiplier)}
}

func (p tensorShapePlan) FeedForwardUpWidth(multiplier uint64) uint64 {
	return multiplier * p.FeedForward
}

func (p tensorShapePlan) FeedForwardDown() []uint64 {
	return []uint64{p.FeedForward, p.Embedding}
}

func (p tensorShapePlan) ExpertRouter() []uint64 {
	return []uint64{p.Embedding, p.Experts}
}

func (p tensorShapePlan) ExpertUp(multiplier uint64) []uint64 {
	return []uint64{p.Embedding, multiplier * p.ExpertWidth, p.Experts}
}
