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
	return buildMTPInput(
		builder, tokenEmbedding, targetHidden, embeddingNorm, hiddenNorm, projection, spec,
		hyv3MTPPolicy, offset,
	)
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
	if !spec.Profile().HasDraftHead(DraftHYV3MTP, spec.NextNPredictLayers, offset) {
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
	return buildMTPOutputs(
		builder, input, outputNorm, output, spec, hyv3MTPPolicy, offset,
	)
}
