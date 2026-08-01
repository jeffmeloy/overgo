package model

import (
	"errors"

	"llamacpp2go/internal/tensor"
)

// BuildQwen35MTPInput: normalized token/target-hidden fusion.
func BuildQwen35MTPInput(
	builder *tensor.Builder,
	tokenEmbedding, targetHidden, embeddingNorm, hiddenNorm, projection *tensor.Tensor,
	spec Spec,
) (*tensor.Tensor, error) {
	if builder == nil || tokenEmbedding == nil || targetHidden == nil || embeddingNorm == nil ||
		hiddenNorm == nil || projection == nil {
		return nil, errors.New("Qwen3.5 MTP input is nil")
	}
	if spec.NextNPredictLayers != 1 || (spec.Architecture != "qwen35" && spec.Architecture != "qwen35moe") ||
		tokenEmbedding.Shape.Rank != 2 || !tokenEmbedding.Shape.Equal(targetHidden.Shape) ||
		tokenEmbedding.Shape.Dims[0] != uint64(spec.EmbeddingLength) {
		return nil, errors.New("Qwen3.5 MTP input shape is incompatible")
	}
	embedding := builder.WeightedRMSNorm(tokenEmbedding, embeddingNorm, spec.RMSNormEpsilon)
	hidden := builder.WeightedRMSNorm(targetHidden, hiddenNorm, spec.RMSNormEpsilon)
	output := builder.MulMat(projection, builder.Concat(embedding, hidden, 0))
	if err := builder.Err(); err != nil {
		return nil, err
	}
	return output, nil
}

// BuildQwen35MTPBlockCached: pinned dense NextN block.
func BuildQwen35MTPBlockCached(
	builder *tensor.Builder,
	input *tensor.Tensor,
	spec Spec,
	weights LayerGraphWeights,
	positions []uint32,
	pastKey, pastValue *tensor.Tensor,
) (Qwen35BlockResult, error) {
	if spec.NextNPredictLayers != 1 || (spec.Architecture != "qwen35" && spec.Architecture != "qwen35moe") {
		return Qwen35BlockResult{}, errors.New("Qwen3.5 MTP architecture is invalid")
	}
	dense := spec
	dense.Architecture = "qwen35"
	return BuildQwen35BlockCached(
		builder, input, dense, weights, positions, false, pastKey, pastValue, nil, nil,
	)
}

// BuildQwen35MTPOutputs: shared-head logits plus next hidden.
func BuildQwen35MTPOutputs(
	builder *tensor.Builder,
	input, outputNorm, output *tensor.Tensor,
	spec Spec,
) (logits, nextHidden *tensor.Tensor, err error) {
	if builder == nil || input == nil || outputNorm == nil || output == nil {
		return nil, nil, errors.New("Qwen3.5 MTP output is nil")
	}
	if spec.NextNPredictLayers != 1 || input.Shape.Rank != 2 ||
		input.Shape.Dims[0] != uint64(spec.EmbeddingLength) {
		return nil, nil, errors.New("Qwen3.5 MTP output shape is incompatible")
	}
	nextHidden = builder.WeightedRMSNorm(input, outputNorm, spec.RMSNormEpsilon)
	logits = builder.MulMat(output, nextHidden)
	if buildErr := builder.Err(); buildErr != nil {
		return nil, nil, buildErr
	}
	return logits, nextHidden, nil
}
