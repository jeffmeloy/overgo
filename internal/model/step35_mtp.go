package model

import (
	"errors"

	"llamacpp2go/internal/tensor"
)

// BuildStep35MTPInput: token/target-hidden fusion.
func BuildStep35MTPInput(
	builder *tensor.Builder,
	tokenEmbedding, targetHidden, embeddingNorm, hiddenNorm, projection *tensor.Tensor,
	spec Spec,
	offset uint32,
) (*tensor.Tensor, error) {
	return buildMTPInput(
		builder, tokenEmbedding, targetHidden, embeddingNorm, hiddenNorm, projection, spec,
		spec.Profile().HasDraftHead(DraftStep35MTP, spec.NextNPredictLayers, offset), mtpWeightedRMS,
		"Step3.5 MTP input is nil", "Step3.5 MTP input shape is incompatible",
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
	if !spec.Profile().HasDraftHead(DraftStep35MTP, spec.NextNPredictLayers, offset) {
		return DenseBlockResult{}, errors.New("Step3.5 MTP head is invalid")
	}
	return BuildDenseBlockCachedForLayer(
		builder, input, spec, weights, positions, pastKey, pastValue,
		spec.BlockCount+offset,
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
		builder, input, outputNorm, output, spec,
		spec.Profile().HasDraftHead(DraftStep35MTP, spec.NextNPredictLayers, offset), true, false, mtpWeightedRMS,
		"Step3.5 MTP output is nil", "Step3.5 MTP output shape is incompatible",
	)
}
