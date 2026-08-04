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
	return buildMTPInput(
		builder, tokenEmbedding, targetHidden, embeddingNorm, hiddenNorm, projection, spec,
		qwen35MTPPolicy, 0,
	)
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
	if !qwen35MTPPolicy.valid(spec, 0) {
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
	return buildMTPOutputs(
		builder, input, outputNorm, output, spec, qwen35MTPPolicy, 0,
	)
}
