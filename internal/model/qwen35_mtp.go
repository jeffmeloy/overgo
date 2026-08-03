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
		spec.Profile().HasSingleDraft(DraftQwen35MTP, spec.NextNPredictLayers),
		mtpWeightedRMS, "Qwen3.5 MTP input is nil", "Qwen3.5 MTP input shape is incompatible",
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
	if !spec.Profile().HasSingleDraft(DraftQwen35MTP, spec.NextNPredictLayers) {
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
		builder, input, outputNorm, output, spec, spec.NextNPredictLayers == 1, false, false, mtpWeightedRMS,
		"Qwen3.5 MTP output is nil", "Qwen3.5 MTP output shape is incompatible",
	)
}
