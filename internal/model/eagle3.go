package model

import (
	"errors"
	"math"

	"llamacpp2go/internal/tensor"
)

// BuildEagle3FeatureEncoder: three-layer target fusion.
func BuildEagle3FeatureEncoder(
	builder *tensor.Builder,
	features, projection *tensor.Tensor,
	spec Spec,
) (*tensor.Tensor, error) {
	if builder == nil || features == nil || projection == nil {
		return nil, errors.New("Eagle3 feature encoder input is nil")
	}
	if spec.Architecture != "eagle3" || features.Shape.Rank != 2 ||
		features.Shape.Dims[0] != 3*uint64(spec.TargetHiddenSize) {
		return nil, errors.New("Eagle3 feature encoder shape is incompatible")
	}
	output := builder.MulMat(projection, features)
	if err := builder.Err(); err != nil {
		return nil, err
	}
	return output, nil
}

// BuildEagle3BlockCached: paired token/target draft block.
func BuildEagle3BlockCached(
	builder *tensor.Builder,
	tokenEmbedding, targetFeature *tensor.Tensor,
	spec Spec,
	weights LayerGraphWeights,
	positions []uint32,
	pastKey, pastValue *tensor.Tensor,
) (DenseBlockResult, error) {
	if builder == nil || tokenEmbedding == nil || targetFeature == nil {
		return DenseBlockResult{}, errors.New("Eagle3 block input is nil")
	}
	if spec.Architecture != "eagle3" || tokenEmbedding.Shape.Rank != 2 ||
		!tokenEmbedding.Shape.Equal(targetFeature.Shape) || tokenEmbedding.Shape.Dims[0] != uint64(spec.EmbeddingLength) ||
		len(positions) != int(tokenEmbedding.Shape.Dims[1]) || (pastKey == nil) != (pastValue == nil) {
		return DenseBlockResult{}, errors.New("Eagle3 block shape is incompatible")
	}
	for _, item := range []*tensor.Tensor{
		weights.AttentionNorm, weights.AttentionNorm2, weights.AttentionQ, weights.AttentionK,
		weights.AttentionV, weights.AttentionOutput, weights.FeedForwardNorm,
		weights.FeedForwardGate, weights.FeedForwardUp, weights.FeedForwardDown,
	} {
		if item == nil {
			return DenseBlockResult{}, errors.New("Eagle3 block weight is nil")
		}
	}
	tokens := tokenEmbedding.Shape.Dims[1]
	tokenNorm := builder.WeightedRMSNorm(tokenEmbedding, weights.AttentionNorm, spec.RMSNormEpsilon)
	targetNorm := builder.WeightedRMSNorm(targetFeature, weights.AttentionNorm2, spec.RMSNormEpsilon)
	attentionInput := builder.Concat(tokenNorm, targetNorm, 0)
	query := builder.Reshape(builder.MulMat(weights.AttentionQ, attentionInput), uint64(spec.KeyLength), uint64(spec.HeadCount), tokens)
	key := builder.Reshape(builder.MulMat(weights.AttentionK, attentionInput), uint64(spec.KeyLength), uint64(spec.HeadCountKV), tokens)
	value := builder.Reshape(builder.MulMat(weights.AttentionV, attentionInput), uint64(spec.ValueLength), uint64(spec.HeadCountKV), tokens)
	frequencyScale := float32(1)
	if (spec.RopeScalingType == "linear" || spec.RopeScalingType == "yarn") && spec.RopeScalingFactor > 0 {
		frequencyScale = 1 / spec.RopeScalingFactor
	}
	query, key = applyRoPEPairWithOptions(builder, query, key, tensor.RoPEOptions{
		Layout: tensor.RoPELayoutNormal, Positions: positions,
		FrequencyFactors: weights.RopeFactors, RotaryDimensions: spec.RopeDimensionCount,
		FrequencyBase: spec.RopeFrequencyBase, FrequencyScale: frequencyScale,
	})
	cacheKey, cacheValue := key, value
	var queryStart uint32
	if pastKey != nil {
		queryStart = uint32(pastKey.Shape.Dims[2])
		cacheKey = builder.Concat(pastKey, key, 2)
		cacheValue = builder.Concat(pastValue, value, 2)
	}
	attention := builder.AttentionWithOptions(query, cacheKey, cacheValue, tensor.AttentionOptions{
		Scale: float32(1 / math.Sqrt(float64(spec.KeyLength))), Causal: true,
		QueryStart: queryStart,
	})
	attention = builder.Reshape(attention, uint64(spec.HeadCount)*uint64(spec.ValueLength), tokens)
	attention = builder.MulMat(weights.AttentionOutput, attention)
	residualInput := targetFeature
	if spec.NormBeforeResidual {
		residualInput = targetNorm
	}
	residual := builder.Add(residualInput, attention)
	normalized := builder.WeightedRMSNorm(residual, weights.FeedForwardNorm, spec.RMSNormEpsilon)
	gate := builder.MulMat(weights.FeedForwardGate, normalized)
	up := builder.MulMat(weights.FeedForwardUp, normalized)
	output := builder.Add(residual, builder.MulMat(weights.FeedForwardDown, builder.SwiGLU(gate, up)))
	if err := builder.Err(); err != nil {
		return DenseBlockResult{}, err
	}
	return DenseBlockResult{Output: output, Key: cacheKey, Value: cacheValue}, nil
}
