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
	if err := requireTensorPair(pastKey, pastValue, "paired causal projection cache is incomplete"); err != nil {
		return DenseBlockResult{}, err
	}
	if err := (graphWeights{
		weights.AttentionQ,
		weights.AttentionK,
		weights.AttentionV,
		weights.AttentionOutput,
	}).validate("paired causal projection"); err != nil {
		return DenseBlockResult{}, err
	}
	tokens, validInput := tensor.MatrixRows(input.Shape, weights.AttentionQ.Shape.Dims[0])
	if !validInput || uint64(len(positions)) != tokens {
		return DenseBlockResult{}, errors.New("paired causal projection shape is incompatible")
	}
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
		pastTokens, keyAxis, validKey := tensor.TrailingExtent32(
			pastKey.Shape, uint64(spec.KeyLength), uint64(spec.HeadCountKV),
		)
		valueTokens, valueAxis, validValue := tensor.TrailingExtent32(
			pastValue.Shape, uint64(spec.ValueLength), uint64(spec.HeadCountKV),
		)
		if !validKey || !validValue || pastTokens != valueTokens {
			return DenseBlockResult{}, errors.New("paired causal projection cache shape is invalid")
		}
		queryStart = builder.CacheTokenOffset(pastTokens)
		cacheKey = builder.WriteCache(pastKey, key, keyAxis, cacheWrite)
		cacheValue = builder.WriteCache(pastValue, value, valueAxis, cacheWrite)
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
