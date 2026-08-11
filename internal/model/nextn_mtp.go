package model

import "overgo/internal/tensor"

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
