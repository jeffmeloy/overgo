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
	if layerIndex >= spec.BlockCount || normalized.Shape.Rank != 2 ||
		normalized.Shape.Dims[0] != uint64(spec.EmbeddingLength) ||
		len(positions) != 1 || normalized.Shape.Dims[1] != 1 {
		return DenseBlockResult{}, errors.New("shared-cache query attention shape is incompatible")
	}
	if err := (graphWeights{
		requireGraphWeight("attention query", weights.AttentionQ),
		requireGraphWeight("attention query norm", weights.AttentionQNorm),
		requireGraphWeight("attention output", weights.AttentionOutput),
	}).validate("shared-cache query attention"); err != nil {
		return DenseBlockResult{}, err
	}
	tokens := normalized.Shape.Dims[1]
	shapes := spec.TensorShapes(layerIndex)
	keyLength, valueLength, kvHeads := shapes.Key, shapes.Value, shapes.KVHeads
	if sharedKey.Shape.Rank != 3 || sharedValue.Shape.Rank != 3 ||
		sharedKey.Shape.Dims[0] != keyLength || sharedValue.Shape.Dims[0] != valueLength ||
		sharedKey.Shape.Dims[1] != kvHeads || sharedValue.Shape.Dims[1] != kvHeads ||
		sharedKey.Shape.Dims[2] != sharedValue.Shape.Dims[2] || sharedKey.Shape.Dims[2] == 0 ||
		sharedKey.Shape.Dims[2] != uint64(positions[0]) {
		return DenseBlockResult{}, errors.New("shared-cache query attention cache shape is incompatible")
	}
	headCount := shapes.QueryHeads
	query := builder.Reshape(builder.MulMat(weights.AttentionQ, normalized), keyLength, headCount, tokens)
	query = builder.WeightedRMSNorm(query, weights.AttentionQNorm, spec.RMSNormEpsilon)
	query = plan.Rotary.ApplyOne(builder, query, positions, nil, weights.RopeFactors)
	queryStart := positions[0] - 1
	attention := plan.AttentionGraph.Build(
		builder, query, sharedKey, sharedValue, nil, nil, 1, queryStart,
	)
	attention = builder.Reshape(attention, shapes.AttentionOutputWidth(), tokens)
	attention = builder.MulMat(weights.AttentionOutput, attention)
	if err := builder.Err(); err != nil {
		return DenseBlockResult{}, err
	}
	return DenseBlockResult{Output: attention}, nil
}
