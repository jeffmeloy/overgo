package model

import (
	"errors"
	"math"

	"llamacpp2go/internal/tensor"
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
	if spec.Architecture != "gemma4-assistant" || targetTokenEmbedding.Shape.Rank != 2 ||
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

// BuildGemma4AssistantBlock: query-only target-cache block.
func BuildGemma4AssistantBlock(
	builder *tensor.Builder,
	input *tensor.Tensor,
	spec Spec,
	weights LayerGraphWeights,
	positions []uint32,
	sharedKey, sharedValue *tensor.Tensor,
	layerIndex uint32,
) (*tensor.Tensor, error) {
	if builder == nil || input == nil || sharedKey == nil || sharedValue == nil {
		return nil, errors.New("Gemma 4 assistant block input is nil")
	}
	if spec.Architecture != "gemma4-assistant" || layerIndex >= spec.BlockCount ||
		input.Shape.Rank != 2 || input.Shape.Dims[0] != uint64(spec.EmbeddingLength) ||
		len(positions) != 1 || input.Shape.Dims[1] != 1 {
		return nil, errors.New("Gemma 4 assistant block shape is incompatible")
	}
	for _, item := range []*tensor.Tensor{
		weights.AttentionNorm, weights.AttentionQ, weights.AttentionQNorm,
		weights.AttentionOutput, weights.AttentionPostNorm, weights.FeedForwardNorm,
		weights.FeedForwardGate, weights.FeedForwardUp, weights.FeedForwardDown,
		weights.FeedForwardPostNorm, weights.LayerOutputScale,
	} {
		if item == nil {
			return nil, errors.New("Gemma 4 assistant block weight is nil")
		}
	}
	tokens := input.Shape.Dims[1]
	keyLength := uint64(spec.LayerKeyLength(layerIndex))
	valueLength := uint64(spec.LayerValueLength(layerIndex))
	kvHeads := uint64(spec.HeadCountKV)
	if sharedKey.Shape.Rank != 3 || sharedValue.Shape.Rank != 3 ||
		sharedKey.Shape.Dims[0] != keyLength || sharedValue.Shape.Dims[0] != valueLength ||
		sharedKey.Shape.Dims[1] != kvHeads || sharedValue.Shape.Dims[1] != kvHeads ||
		sharedKey.Shape.Dims[2] != sharedValue.Shape.Dims[2] || sharedKey.Shape.Dims[2] == 0 ||
		sharedKey.Shape.Dims[2] != uint64(positions[0]) {
		return nil, errors.New("Gemma 4 assistant shared cache shape is incompatible")
	}
	headCount := uint64(spec.HeadCount)
	normalized := builder.WeightedRMSNorm(input, weights.AttentionNorm, spec.RMSNormEpsilon)
	query := builder.Reshape(builder.MulMat(weights.AttentionQ, normalized), keyLength, headCount, tokens)
	query = builder.WeightedRMSNorm(query, weights.AttentionQNorm, spec.RMSNormEpsilon)
	frequencyBase := spec.RopeFrequencyBase
	if spec.IsSlidingLayer(layerIndex) {
		frequencyBase = spec.RopeFrequencySWA
	}
	if weights.RopeFactors != nil {
		query = builder.RoPENeoXScaledWithFactors(
			query, positions, spec.LayerRopeDimensionCount(layerIndex), frequencyBase, 1, weights.RopeFactors,
		)
	} else {
		query = builder.RoPENeoXScaled(
			query, positions, spec.LayerRopeDimensionCount(layerIndex), frequencyBase, 1,
		)
	}
	queryStart := positions[0] - 1
	var attention *tensor.Tensor
	if spec.IsSlidingLayer(layerIndex) {
		attention = builder.AttentionWindowWithOffset(
			query, sharedKey, sharedValue, 1, true, queryStart, spec.SlidingWindow,
		)
	} else {
		attention = builder.AttentionWithOffset(query, sharedKey, sharedValue, 1, true, queryStart)
	}
	attention = builder.Reshape(attention, headCount*valueLength, tokens)
	attention = builder.MulMat(weights.AttentionOutput, attention)
	attention = builder.WeightedRMSNorm(attention, weights.AttentionPostNorm, spec.RMSNormEpsilon)
	residual := builder.Add(input, attention)
	ffnInput := builder.WeightedRMSNorm(residual, weights.FeedForwardNorm, spec.RMSNormEpsilon)
	gate := builder.MulMat(weights.FeedForwardGate, ffnInput)
	up := builder.MulMat(weights.FeedForwardUp, ffnInput)
	feedForward := builder.MulMat(weights.FeedForwardDown, builder.GEGLU(gate, up))
	feedForward = builder.WeightedRMSNorm(feedForward, weights.FeedForwardPostNorm, spec.RMSNormEpsilon)
	output := builder.Multiply(builder.Add(residual, feedForward), weights.LayerOutputScale)
	if err := builder.Err(); err != nil {
		return nil, err
	}
	return output, nil
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
	if spec.Architecture != "gemma4-assistant" || input.Shape.Rank != 2 ||
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
