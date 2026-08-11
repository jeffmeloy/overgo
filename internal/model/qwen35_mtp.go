package model

import "overgo/internal/tensor"

// BuildQwen35MTPInput: normalized token/target-hidden fusion.
func BuildQwen35MTPInput(
	builder *tensor.Builder,
	tokenEmbedding, targetHidden, embeddingNorm, hiddenNorm, projection *tensor.Tensor,
	spec Spec,
) (*tensor.Tensor, error) {
	return buildMTPInput(
		builder, tokenEmbedding, targetHidden, embeddingNorm, hiddenNorm, projection, spec,
		qwen35MTPPolicy, 0,
	)
}

// BuildQwen35MTPOutputs: shared-head logits plus next hidden.
func BuildQwen35MTPOutputs(
	builder *tensor.Builder,
	input, outputNorm, output *tensor.Tensor,
	spec Spec,
) (logits, nextHidden *tensor.Tensor, err error) {
	return buildMTPOutputs(
		builder, input, outputNorm, output, spec, qwen35MTPPolicy, 0,
	)
}
