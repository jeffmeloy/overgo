package model

import (
	"errors"
	"math"

	"llamacpp2go/internal/tensor"
)

// BuildDFlashFeatureEncoder: target-layer fusion projection.
func BuildDFlashFeatureEncoder(
	builder *tensor.Builder,
	features, projection, norm *tensor.Tensor,
	spec Spec,
) (*tensor.Tensor, error) {
	if builder == nil || features == nil || projection == nil || norm == nil {
		return nil, errors.New("DFlash feature encoder input is nil")
	}
	if spec.Architecture != "dflash" || features.Shape.Rank != 2 ||
		features.Shape.Dims[0] != uint64(len(spec.TargetLayers))*uint64(spec.EmbeddingLength) {
		return nil, errors.New("DFlash feature encoder shape is incompatible")
	}
	output := builder.MulMat(projection, features)
	output = builder.WeightedRMSNorm(output, norm, spec.RMSNormEpsilon)
	if err := builder.Err(); err != nil {
		return nil, err
	}
	return output, nil
}

// BuildDFlashCacheInjection: fused-feature K/V append.
func BuildDFlashCacheInjection(
	builder *tensor.Builder,
	fused *tensor.Tensor,
	spec Spec,
	weights LayerGraphWeights,
	positions []uint32,
	pastKey, pastValue *tensor.Tensor,
) (*tensor.Tensor, *tensor.Tensor, error) {
	if builder == nil || fused == nil || weights.AttentionK == nil || weights.AttentionV == nil || weights.AttentionKNorm == nil {
		return nil, nil, errors.New("DFlash cache injection input is nil")
	}
	if spec.Architecture != "dflash" || fused.Shape.Rank != 2 || fused.Shape.Dims[0] != uint64(spec.EmbeddingLength) ||
		len(positions) != int(fused.Shape.Dims[1]) || (pastKey == nil) != (pastValue == nil) {
		return nil, nil, errors.New("DFlash cache injection shape is incompatible")
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
	key = builder.RoPENeoXScaled(
		key, positions, spec.RopeDimensionCount, spec.RopeFrequencyBase, frequencyScale,
	)
	if pastKey != nil {
		key = builder.Concat(pastKey, key, 2)
		value = builder.Concat(pastValue, value, 2)
	}
	if err := builder.Err(); err != nil {
		return nil, nil, err
	}
	return key, value, nil
}

// DFlashAttentionScale: head-width scale.
func DFlashAttentionScale(spec Spec) float32 {
	return float32(1 / math.Sqrt(float64(spec.KeyLength)))
}
