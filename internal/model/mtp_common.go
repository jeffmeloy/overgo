package model

import (
	"errors"

	"llamacpp2go/internal/tensor"
)

type mtpNormalization uint8

const (
	mtpWeightedRMS mtpNormalization = iota
	mtpArchitectureNorm
)

func buildMTPInput(
	builder *tensor.Builder,
	tokenEmbedding, targetHidden, embeddingNorm, hiddenNorm, projection *tensor.Tensor,
	spec Spec,
	valid bool,
	normalization mtpNormalization,
	nilMessage, shapeMessage string,
) (*tensor.Tensor, error) {
	if builder == nil || tokenEmbedding == nil || targetHidden == nil || embeddingNorm == nil ||
		hiddenNorm == nil || projection == nil {
		return nil, errors.New(nilMessage)
	}
	if !valid || tokenEmbedding.Shape.Rank != 2 || !tokenEmbedding.Shape.Equal(targetHidden.Shape) ||
		tokenEmbedding.Shape.Dims[0] != uint64(spec.EmbeddingLength) {
		return nil, errors.New(shapeMessage)
	}
	embedding := normalizeMTP(builder, tokenEmbedding, embeddingNorm, spec, normalization)
	hidden := normalizeMTP(builder, targetHidden, hiddenNorm, spec, normalization)
	output := builder.MulMat(projection, builder.Concat(embedding, hidden, 0))
	if err := builder.Err(); err != nil {
		return nil, err
	}
	return output, nil
}

func buildMTPOutputs(
	builder *tensor.Builder,
	input, outputNorm, output *tensor.Tensor,
	spec Spec,
	valid, carryRaw, scaleLogits bool,
	normalization mtpNormalization,
	nilMessage, shapeMessage string,
) (logits, nextHidden *tensor.Tensor, err error) {
	if builder == nil || input == nil || outputNorm == nil || output == nil {
		return nil, nil, errors.New(nilMessage)
	}
	if !valid || input.Shape.Rank != 2 || input.Shape.Dims[0] != uint64(spec.EmbeddingLength) {
		return nil, nil, errors.New(shapeMessage)
	}
	normalized := normalizeMTP(builder, input, outputNorm, spec, normalization)
	nextHidden = normalized
	if carryRaw {
		nextHidden = input
	}
	logits = builder.MulMat(output, normalized)
	if scaleLogits {
		if scale := spec.OutputLogitMultiplier(); scale != 1 {
			logits = builder.Scale(logits, scale)
		}
	}
	if buildErr := builder.Err(); buildErr != nil {
		return nil, nil, buildErr
	}
	return logits, nextHidden, nil
}

func normalizeMTP(
	builder *tensor.Builder,
	input, weight *tensor.Tensor,
	spec Spec,
	normalization mtpNormalization,
) *tensor.Tensor {
	if normalization == mtpArchitectureNorm {
		return ApplyNormalization(builder, input, weight, nil, spec)
	}
	return builder.WeightedRMSNorm(input, weight, spec.RMSNormEpsilon)
}
