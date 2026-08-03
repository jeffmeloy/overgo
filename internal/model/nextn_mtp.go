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
	return buildMTPInput(
		builder, tokenEmbedding, targetHidden, embeddingNorm, hiddenNorm, projection, spec,
		nextNMTPPolicy, offset,
	)
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
	if plan := spec.Profile().DraftPlan(spec.NextNPredictLayers); plan.Kind != DraftNextNMTP || !plan.HasHead(offset) {
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
	return buildMTPOutputs(
		builder, input, outputNorm, output, spec, nextNMTPPolicy, offset,
	)
}
