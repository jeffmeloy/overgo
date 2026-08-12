package model

import (
	"errors"

	"overgo/internal/tensor"
)

type cacheProjectionPolicy uint8

const (
	cacheProjectionNone cacheProjectionPolicy = iota
	cacheProjectionRotaryQKNorm
)

// BuildCacheProjection executes the compiled cache-projection policy.
func (p ModelPlan) BuildCacheProjection(
	builder *tensor.Builder,
	fused *tensor.Tensor,
	weights LayerGraphWeights,
	positions []uint32,
	pastKey, pastValue *tensor.Tensor,
) (*tensor.Tensor, *tensor.Tensor, error) {
	spec := p.spec
	if builder == nil || fused == nil || weights.AttentionK == nil || weights.AttentionV == nil || weights.AttentionKNorm == nil {
		return nil, nil, errors.New("compiled cache projection input is nil")
	}
	if p.cacheProject != cacheProjectionRotaryQKNorm || fused.Shape.Rank != 2 || fused.Shape.Dims[0] != uint64(spec.EmbeddingLength) ||
		len(positions) != int(fused.Shape.Dims[1]) || (pastKey == nil) != (pastValue == nil) {
		return nil, nil, errors.New("compiled cache projection shape is incompatible")
	}
	tokens := fused.Shape.Dims[1]
	key := builder.Reshape(
		builder.MulMat(weights.AttentionK, fused),
		uint64(spec.KeyLength), uint64(spec.HeadCountKV), tokens,
	)
	value := builder.Reshape(
		builder.MulMat(weights.AttentionV, fused),
		uint64(spec.ValueLength), uint64(spec.HeadCountKV), tokens,
	)
	key = builder.WeightedRMSNorm(key, weights.AttentionKNorm, spec.RMSNormEpsilon)
	frequencyScale := float32(1)
	if (spec.RopeScalingType == "linear" || spec.RopeScalingType == "yarn") && spec.RopeScalingFactor > 0 {
		frequencyScale = 1 / spec.RopeScalingFactor
	}
	key = builder.RoPEWithOptions(key, tensor.RoPEOptions{
		Layout: tensor.RoPELayoutNeoX, Positions: positions,
		RotaryDimensions: spec.RopeDimensionCount, FrequencyBase: spec.RopeFrequencyBase,
		FrequencyScale: frequencyScale,
	})
	if pastKey != nil {
		key = builder.Concat(pastKey, key, 2)
		value = builder.Concat(pastValue, value, 2)
	}
	if err := builder.Err(); err != nil {
		return nil, nil, err
	}
	return key, value, nil
}
