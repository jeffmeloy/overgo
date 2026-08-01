package model

import (
	"errors"

	"llamacpp2go/internal/tensor"
)

// BuildHYV3MTPInput: post-norm hidden/token fusion.
func BuildHYV3MTPInput(
	builder *tensor.Builder,
	tokenEmbedding, targetHidden, embeddingNorm, hiddenNorm, projection *tensor.Tensor,
	spec Spec,
	offset uint32,
) (*tensor.Tensor, error) {
	if builder == nil || tokenEmbedding == nil || targetHidden == nil || embeddingNorm == nil ||
		hiddenNorm == nil || projection == nil {
		return nil, errors.New("HY-V3 MTP input is nil")
	}
	if spec.Architecture != "hy_v3" || offset >= spec.NextNPredictLayers ||
		tokenEmbedding.Shape.Rank != 2 || !tokenEmbedding.Shape.Equal(targetHidden.Shape) ||
		tokenEmbedding.Shape.Dims[0] != uint64(spec.EmbeddingLength) {
		return nil, errors.New("HY-V3 MTP input shape is incompatible")
	}
	embedding := builder.WeightedRMSNorm(tokenEmbedding, embeddingNorm, spec.RMSNormEpsilon)
	hidden := builder.WeightedRMSNorm(targetHidden, hiddenNorm, spec.RMSNormEpsilon)
	output := builder.MulMat(projection, builder.Concat(embedding, hidden, 0))
	if err := builder.Err(); err != nil {
		return nil, err
	}
	return output, nil
}

// BuildHYV3MTPBlockCached: selected full draft block.
func BuildHYV3MTPBlockCached(
	builder *tensor.Builder,
	input *tensor.Tensor,
	spec Spec,
	weights LayerGraphWeights,
	positions []uint32,
	pastKey, pastValue *tensor.Tensor,
	offset uint32,
) (DenseBlockResult, error) {
	if spec.Architecture != "hy_v3" || offset >= spec.NextNPredictLayers {
		return DenseBlockResult{}, errors.New("HY-V3 MTP head is invalid")
	}
	return BuildDenseBlockCachedForLayer(
		builder, input, spec, weights, positions, pastKey, pastValue,
		spec.BlockCount+offset,
	)
}

// BuildHYV3MTPOutputs: post-norm hidden plus selected logits.
func BuildHYV3MTPOutputs(
	builder *tensor.Builder,
	input, outputNorm, output *tensor.Tensor,
	spec Spec,
	offset uint32,
) (logits, nextHidden *tensor.Tensor, err error) {
	if builder == nil || input == nil || outputNorm == nil || output == nil {
		return nil, nil, errors.New("HY-V3 MTP output is nil")
	}
	if spec.Architecture != "hy_v3" || offset >= spec.NextNPredictLayers ||
		input.Shape.Rank != 2 || input.Shape.Dims[0] != uint64(spec.EmbeddingLength) {
		return nil, nil, errors.New("HY-V3 MTP output shape is incompatible")
	}
	nextHidden = builder.WeightedRMSNorm(input, outputNorm, spec.RMSNormEpsilon)
	logits = builder.MulMat(output, nextHidden)
	if buildErr := builder.Err(); buildErr != nil {
		return nil, nil, buildErr
	}
	return logits, nextHidden, nil
}
