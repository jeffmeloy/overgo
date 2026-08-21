package model

import (
	"errors"

	"overgo/internal/tensor"
)

func buildSharedCacheQKNormMix(
	builder *tensor.Builder,
	normalized *tensor.Tensor,
	spec Spec,
	weights LayerGraphWeights,
	positions []uint32,
	sharedKey, sharedValue *tensor.Tensor,
	plan LayerPlan,
) (DenseBlockResult, error) {
	layerIndex := plan.Layer
	if builder == nil || normalized == nil || sharedKey == nil || sharedValue == nil {
		return DenseBlockResult{}, errors.New("shared-cache query attention input is nil")
	}
	tokenCount, validInput := tensor.MatrixRows32(normalized.Shape, uint64(spec.EmbeddingLength))
	if layerIndex >= spec.BlockCount || !validInput || tokenCount != tensor.SingletonExtent || len(positions) != int(tokenCount) {
		return DenseBlockResult{}, errors.New("shared-cache query attention shape is incompatible")
	}
	if err := (graphWeights{
		requireGraphWeight("attention query", weights.AttentionQ),
		requireGraphWeight("attention query norm", weights.AttentionQNorm),
		requireGraphWeight("attention output", weights.AttentionOutput),
	}).validate("shared-cache query attention"); err != nil {
		return DenseBlockResult{}, err
	}
	tokens := uint64(tokenCount)
	shapes := spec.TensorShapes(layerIndex)
	keyLength, valueLength, kvHeads := shapes.Key, shapes.Value, shapes.KVHeads
	keyTokens, _, validKey := tensor.TrailingExtent32(sharedKey.Shape, keyLength, kvHeads)
	valueTokens, _, validValue := tensor.TrailingExtent32(sharedValue.Shape, valueLength, kvHeads)
	if !validKey || !validValue || keyTokens != valueTokens || keyTokens != positions[tensor.FirstOffset] {
		return DenseBlockResult{}, errors.New("shared-cache query attention cache shape is incompatible")
	}
	headCount := shapes.QueryHeads
	query := builder.Reshape(builder.MulMat(weights.AttentionQ, normalized), keyLength, headCount, tokens)
	query = builder.WeightedRMSNorm(query, weights.AttentionQNorm, spec.RMSNormEpsilon)
	query = plan.Rotary.ApplyOne(builder, query, positions, nil, weights.RopeFactors)
	queryStart := positions[tensor.FirstOffset] - tensor.SingletonExtent
	attention := plan.AttentionGraph.Build(
		builder, query, sharedKey, sharedValue, nil, nil, tensor.SingletonExtent, queryStart,
	)
	attention = builder.Reshape(attention, shapes.AttentionOutputWidth(), tokens)
	attention = builder.MulMat(weights.AttentionOutput, attention)
	if err := builder.Err(); err != nil {
		return DenseBlockResult{}, err
	}
	return DenseBlockResult{Output: attention}, nil
}
