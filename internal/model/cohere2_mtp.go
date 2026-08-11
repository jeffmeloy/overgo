package model

import "overgo/internal/tensor"

// BuildCohere2MTPInput: normalized token/hidden fusion.
func BuildCohere2MTPInput(
	builder *tensor.Builder,
	tokenEmbedding, targetHidden, embeddingNorm, hiddenNorm, projection *tensor.Tensor,
	spec Spec,
) (*tensor.Tensor, error) {
	return buildMTPInput(
		builder, tokenEmbedding, targetHidden, embeddingNorm, hiddenNorm, projection, spec,
		cohere2MTPPolicy, 0,
	)
}

// BuildCohere2MTPOutputs: normalized hidden plus scaled logits.
func BuildCohere2MTPOutputs(
	builder *tensor.Builder,
	input, outputNorm, output *tensor.Tensor,
	spec Spec,
) (logits, nextHidden *tensor.Tensor, err error) {
	return buildMTPOutputs(
		builder, input, outputNorm, output, spec, cohere2MTPPolicy, 0,
	)
}
