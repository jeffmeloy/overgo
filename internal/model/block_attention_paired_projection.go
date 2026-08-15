package model

import (
	"errors"
	"math"

	"overgo/internal/tensor"
)

func buildPairedCausalProjectionMixCached(
	builder *tensor.Builder,
	input *tensor.Tensor,
	spec Spec,
	weights LayerGraphWeights,
	positions []uint32,
	pastKey, pastValue *tensor.Tensor,
	cacheWrite tensor.CacheWriteMode,
) (DenseBlockResult, error) {
	if builder == nil || input == nil {
		return DenseBlockResult{}, errors.New("paired causal projection input is nil")
	}
	if input.Shape.Rank != 2 || input.Shape.Dims[0] != 2*uint64(spec.EmbeddingLength) ||
		len(positions) != int(input.Shape.Dims[1]) {
		return DenseBlockResult{}, errors.New("paired causal projection shape is incompatible")
	}
	if err := requireTensorPair(pastKey, pastValue, "paired causal projection cache is incomplete"); err != nil {
		return DenseBlockResult{}, err
	}
	if err := (graphWeights{
		requireGraphWeight("attention query", weights.AttentionQ),
		requireGraphWeight("attention key", weights.AttentionK),
		requireGraphWeight("attention value", weights.AttentionV),
		requireGraphWeight("attention output", weights.AttentionOutput),
	}).validate("paired causal projection"); err != nil {
		return DenseBlockResult{}, err
	}
	tokens := input.Shape.Dims[1]
	query := builder.Reshape(builder.MulMat(weights.AttentionQ, input), uint64(spec.KeyLength), uint64(spec.HeadCount), tokens)
	key := builder.Reshape(builder.MulMat(weights.AttentionK, input), uint64(spec.KeyLength), uint64(spec.HeadCountKV), tokens)
	value := builder.Reshape(builder.MulMat(weights.AttentionV, input), uint64(spec.ValueLength), uint64(spec.HeadCountKV), tokens)
	query, key = applyRoPEPairWithOptions(builder, query, key, tensor.RoPEOptions{
		Layout: tensor.RoPELayoutNormal, Positions: positions,
		FrequencyFactors: weights.RopeFactors, RotaryDimensions: spec.RopeDimensionCount,
		FrequencyBase: spec.RopeFrequencyBase, FrequencyScale: spec.ropeFrequencyScale(),
	})
	cacheKey, cacheValue := key, value
	var queryStart uint32
	if pastKey != nil {
		if pastKey.Shape.Rank != 3 || pastKey.Shape.Dims[2] > math.MaxUint32 {
			return DenseBlockResult{}, errors.New("paired causal projection cache shape is invalid")
		}
		queryStart = builder.CacheTokenOffset(uint32(pastKey.Shape.Dims[2]))
		cacheKey = builder.WriteCache(pastKey, key, 2, cacheWrite)
		cacheValue = builder.WriteCache(pastValue, value, 2, cacheWrite)
	}
	attention := builder.AttentionWithOptions(query, cacheKey, cacheValue, tensor.AttentionOptions{
		Scale: float32(1 / math.Sqrt(float64(spec.KeyLength))), Causal: true,
		QueryStart: queryStart,
	})
	attention = builder.Reshape(attention, uint64(spec.HeadCount)*uint64(spec.ValueLength), tokens)
	attention = builder.MulMat(weights.AttentionOutput, attention)
	if err := builder.Err(); err != nil {
		return DenseBlockResult{}, err
	}
	return DenseBlockResult{Output: attention, Key: cacheKey, Value: cacheValue}, nil
}
