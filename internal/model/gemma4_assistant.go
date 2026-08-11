package model

import (
	"errors"
	"math"

	"overgo/internal/tensor"
)

// BuildGemma4AssistantInput: target token/hidden fusion.
func BuildGemma4AssistantInput(
	builder *tensor.Builder,
	targetTokenEmbedding, targetHidden, projection *tensor.Tensor,
	spec Spec,
) (*tensor.Tensor, error) {
	if builder == nil || targetTokenEmbedding == nil || targetHidden == nil || projection == nil {
		return nil, errors.New("Gemma 4 assistant input is nil")
	}
	if spec.Profile().Forward != ForwardGemma4Assistant || targetTokenEmbedding.Shape.Rank != 2 ||
		!targetTokenEmbedding.Shape.Equal(targetHidden.Shape) ||
		targetHidden.Shape.Dims[0] != uint64(spec.TargetHiddenSize) {
		return nil, errors.New("Gemma 4 assistant input shape is incompatible")
	}
	embedding := builder.Scale(targetTokenEmbedding, float32(math.Sqrt(float64(spec.TargetHiddenSize))))
	output := builder.MulMat(projection, builder.Concat(embedding, targetHidden, 0))
	if err := builder.Err(); err != nil {
		return nil, err
	}
	return output, nil
}

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

// BuildGemma4AssistantOutputs: logits plus recurrent target-width hidden.
func BuildGemma4AssistantOutputs(
	builder *tensor.Builder,
	input, outputNorm, output, postProjection *tensor.Tensor,
	spec Spec,
) (logits, nextHidden *tensor.Tensor, err error) {
	if builder == nil || input == nil || outputNorm == nil || output == nil || postProjection == nil {
		return nil, nil, errors.New("Gemma 4 assistant output is nil")
	}
	if spec.Profile().Forward != ForwardGemma4Assistant || input.Shape.Rank != 2 ||
		input.Shape.Dims[0] != uint64(spec.EmbeddingLength) {
		return nil, nil, errors.New("Gemma 4 assistant output shape is incompatible")
	}
	normalized := builder.WeightedRMSNorm(input, outputNorm, spec.RMSNormEpsilon)
	logits = builder.MulMat(output, normalized)
	nextHidden = builder.MulMat(postProjection, normalized)
	if buildErr := builder.Err(); buildErr != nil {
		return nil, nil, buildErr
	}
	return logits, nextHidden, nil
}
