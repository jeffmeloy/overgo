package model

import "overgo/internal/tensor"

// BuildStep35MTPInput: token/target-hidden fusion.
func BuildStep35MTPInput(
	builder *tensor.Builder,
	tokenEmbedding, targetHidden, embeddingNorm, hiddenNorm, projection *tensor.Tensor,
	spec Spec,
	offset uint32,
) (*tensor.Tensor, error) {
	return buildMTPInput(
		builder, tokenEmbedding, targetHidden, embeddingNorm, hiddenNorm, projection, spec,
		step35MTPPolicy, offset,
	)
}

// BuildStep35MTPBlockCached: selected full draft block.
func BuildStep35MTPBlockCached(
	builder *tensor.Builder,
	input *tensor.Tensor,
	spec Spec,
	weights LayerGraphWeights,
	positions []uint32,
	pastKey, pastValue *tensor.Tensor,
	offset uint32,
) (DenseBlockResult, error) {
	return buildMTPDenseBlock(
		builder, input, spec, spec, weights, positions, pastKey, pastValue,
		step35MTPPolicy, offset,
	)
}

// BuildStep35MTPOutputs: pre-norm hidden plus selected head logits.
func BuildStep35MTPOutputs(
	builder *tensor.Builder,
	input, outputNorm, output *tensor.Tensor,
	spec Spec,
	offset uint32,
) (logits, nextHidden *tensor.Tensor, err error) {
	return buildMTPOutputs(
		builder, input, outputNorm, output, spec, step35MTPPolicy, offset,
	)
}
