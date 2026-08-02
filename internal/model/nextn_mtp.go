package model

import (
	"errors"

	"llamacpp2go/internal/tensor"
)

// BuildNextNMTPInput: normalized token/target fusion.
func BuildNextNMTPInput(
	builder *tensor.Builder,
	tokenEmbedding, targetHidden, embeddingNorm, hiddenNorm, projection *tensor.Tensor,
	spec Spec,
	offset uint32,
) (*tensor.Tensor, error) {
	if builder == nil || tokenEmbedding == nil || targetHidden == nil || embeddingNorm == nil ||
		hiddenNorm == nil || projection == nil {
		return nil, errors.New("NextN MTP input is nil")
	}
	if !supportsNextNMTPArchitecture(spec.Architecture) || offset >= spec.NextNPredictLayers ||
		tokenEmbedding.Shape.Rank != 2 || !tokenEmbedding.Shape.Equal(targetHidden.Shape) ||
		tokenEmbedding.Shape.Dims[0] != uint64(spec.EmbeddingLength) {
		return nil, errors.New("NextN MTP input shape is incompatible")
	}
	embedding := builder.WeightedRMSNorm(tokenEmbedding, embeddingNorm, spec.RMSNormEpsilon)
	hidden := builder.WeightedRMSNorm(targetHidden, hiddenNorm, spec.RMSNormEpsilon)
	output := builder.MulMat(projection, builder.Concat(embedding, hidden, 0))
	if err := builder.Err(); err != nil {
		return nil, err
	}
	return output, nil
}

// BuildNextNMTPBlockCached: appended decoder block.
func BuildNextNMTPBlockCached(
	builder *tensor.Builder,
	input *tensor.Tensor,
	spec Spec,
	weights LayerGraphWeights,
	positions []uint32,
	pastKey, pastValue *tensor.Tensor,
	offset uint32,
) (DenseBlockResult, error) {
	return BuildNextNMTPBlockCachedWithDSA(
		builder, input, spec, weights, positions, pastKey, pastValue, nil, nil, offset,
	)
}

// BuildNextNMTPBlockCachedWithDSA: appended block plus sparse-indexer state.
func BuildNextNMTPBlockCachedWithDSA(
	builder *tensor.Builder,
	input *tensor.Tensor,
	spec Spec,
	weights LayerGraphWeights,
	positions []uint32,
	pastKey, pastValue, pastIndexerKey, previousTopK *tensor.Tensor,
	offset uint32,
) (DenseBlockResult, error) {
	if !supportsNextNMTPArchitecture(spec.Architecture) || offset >= spec.NextNPredictLayers {
		return DenseBlockResult{}, errors.New("NextN MTP block is invalid")
	}
	executable := spec
	executable.BlockCount += spec.NextNPredictLayers
	return BuildArchitectureBlockCached(BlockDispatchOptions{
		Builder: builder, Input: input, Spec: executable, Weights: weights,
		Positions: positions, PastKey: pastKey, PastValue: pastValue,
		PastIndexerKey: pastIndexerKey, PerLayerInput: previousTopK,
		Layer: spec.BlockCount + offset,
	})
}

// BuildNextNMTPOutputs: normalized hidden and logits.
func BuildNextNMTPOutputs(
	builder *tensor.Builder,
	input, outputNorm, output *tensor.Tensor,
	spec Spec,
	offset uint32,
) (logits, nextHidden *tensor.Tensor, err error) {
	if builder == nil || input == nil || outputNorm == nil || output == nil {
		return nil, nil, errors.New("NextN MTP output is nil")
	}
	if !supportsNextNMTPArchitecture(spec.Architecture) || offset >= spec.NextNPredictLayers ||
		input.Shape.Rank != 2 || input.Shape.Dims[0] != uint64(spec.EmbeddingLength) {
		return nil, nil, errors.New("NextN MTP output shape is incompatible")
	}
	nextHidden = builder.WeightedRMSNorm(input, outputNorm, spec.RMSNormEpsilon)
	logits = builder.MulMat(output, nextHidden)
	if buildErr := builder.Err(); buildErr != nil {
		return nil, nil, buildErr
	}
	return logits, nextHidden, nil
}

func supportsNextNMTPArchitecture(architecture string) bool {
	return architecture == "glm4" || architecture == "glm4moe" || architecture == "exaone4" || architecture == "exaone-moe" || architecture == "mimo2" || architecture == "bailingmoe2" || architecture == "deepseek32" || architecture == "glm-dsa"
}
