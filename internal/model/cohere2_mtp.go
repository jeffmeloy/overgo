package model

import (
	"errors"

	"llamacpp2go/internal/tensor"
)

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

// BuildCohere2MTPBlockCached: full-attention routed draft block.
func BuildCohere2MTPBlockCached(
	builder *tensor.Builder,
	input *tensor.Tensor,
	spec Spec,
	weights LayerGraphWeights,
	positions []uint32,
	pastKey, pastValue *tensor.Tensor,
) (DenseBlockResult, error) {
	if !spec.Profile().HasSingleDraft(DraftCohere2MTP, spec.NextNPredictLayers) ||
		weights.FeedForwardRouter == nil {
		return DenseBlockResult{}, errors.New("Cohere2-MoE MTP block is invalid")
	}
	mtp := spec
	mtp.BlockCount++
	mtp.LeadingDenseBlocks = mtp.BlockCount
	mtp.SlidingWindow = 0
	return BuildDenseBlockCachedForLayer(
		builder, input, mtp, weights, positions, pastKey, pastValue, spec.BlockCount,
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
