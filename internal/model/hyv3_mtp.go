package model

import "overgo/internal/tensor"

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
