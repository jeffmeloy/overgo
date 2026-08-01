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
	if builder == nil || tokenEmbedding == nil || targetHidden == nil || embeddingNorm == nil ||
		hiddenNorm == nil || projection == nil {
		return nil, errors.New("Cohere2-MoE MTP input is nil")
	}
	if spec.Architecture != "cohere2moe" || spec.NextNPredictLayers != 1 ||
		tokenEmbedding.Shape.Rank != 2 || !tokenEmbedding.Shape.Equal(targetHidden.Shape) ||
		tokenEmbedding.Shape.Dims[0] != uint64(spec.EmbeddingLength) {
		return nil, errors.New("Cohere2-MoE MTP input shape is incompatible")
	}
	embedding := ApplyNormalization(builder, tokenEmbedding, embeddingNorm, nil, spec)
	hidden := ApplyNormalization(builder, targetHidden, hiddenNorm, nil, spec)
	output := builder.MulMat(projection, builder.Concat(embedding, hidden, 0))
	if err := builder.Err(); err != nil {
		return nil, err
	}
	return output, nil
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
	if spec.Architecture != "cohere2moe" || spec.NextNPredictLayers != 1 ||
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
	if builder == nil || input == nil || outputNorm == nil || output == nil {
		return nil, nil, errors.New("Cohere2-MoE MTP output is nil")
	}
	if spec.Architecture != "cohere2moe" || spec.NextNPredictLayers != 1 ||
		input.Shape.Rank != 2 || input.Shape.Dims[0] != uint64(spec.EmbeddingLength) {
		return nil, nil, errors.New("Cohere2-MoE MTP output shape is incompatible")
	}
	nextHidden = ApplyNormalization(builder, input, outputNorm, nil, spec)
	logits = builder.MulMat(output, nextHidden)
	if scale := spec.OutputLogitMultiplier(); scale != 1 {
		logits = builder.Scale(logits, scale)
	}
	if buildErr := builder.Err(); buildErr != nil {
		return nil, nil, buildErr
	}
	return logits, nextHidden, nil
}
