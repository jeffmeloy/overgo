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

type mtpPolicy struct {
	kind          DraftKind
	label         string
	normalization mtpNormalization
	single        bool
	carryRaw      bool
	scaleLogits   bool
}

var (
	qwen35MTPPolicy  = mtpPolicy{kind: DraftQwen35MTP, label: "Qwen3.5", single: true}
	step35MTPPolicy  = mtpPolicy{kind: DraftStep35MTP, label: "Step3.5", carryRaw: true}
	hyv3MTPPolicy    = mtpPolicy{kind: DraftHYV3MTP, label: "HY-V3"}
	nextNMTPPolicy   = mtpPolicy{kind: DraftNextNMTP, label: "NextN"}
	cohere2MTPPolicy = mtpPolicy{
		kind: DraftCohere2MTP, label: "Cohere2-MoE", normalization: mtpArchitectureNorm,
		single: true, scaleLogits: true,
	}
)

func (p mtpPolicy) valid(spec Spec, offset uint32) bool {
	plan := spec.Profile().DraftPlan(spec.NextNPredictLayers)
	return plan.Kind == p.kind && plan.HasHead(offset) && (!p.single || offset == 0 && plan.SessionEligible())
}

func buildMTPDenseBlock(
	builder *tensor.Builder,
	input *tensor.Tensor,
	spec, executable Spec,
	weights LayerGraphWeights,
	positions []uint32,
	pastKey, pastValue *tensor.Tensor,
	policy mtpPolicy,
	offset uint32,
) (DenseBlockResult, error) {
	if !policy.valid(spec, offset) {
		return DenseBlockResult{}, errors.New(policy.label + " MTP block is invalid")
	}
	return BuildDenseBlockCachedForLayer(
		builder, input, executable, weights, positions, pastKey, pastValue,
		spec.BlockCount+offset,
	)
}

func buildMTPInput(
	builder *tensor.Builder,
	tokenEmbedding, targetHidden, embeddingNorm, hiddenNorm, projection *tensor.Tensor,
	spec Spec,
	policy mtpPolicy,
	offset uint32,
) (*tensor.Tensor, error) {
	if builder == nil || tokenEmbedding == nil || targetHidden == nil || embeddingNorm == nil ||
		hiddenNorm == nil || projection == nil {
		return nil, errors.New(policy.label + " MTP input is nil")
	}
	if !policy.valid(spec, offset) || tokenEmbedding.Shape.Rank != 2 || !tokenEmbedding.Shape.Equal(targetHidden.Shape) ||
		tokenEmbedding.Shape.Dims[0] != uint64(spec.EmbeddingLength) {
		return nil, errors.New(policy.label + " MTP input shape is incompatible")
	}
	embedding := normalizeMTP(builder, tokenEmbedding, embeddingNorm, spec, policy.normalization)
	hidden := normalizeMTP(builder, targetHidden, hiddenNorm, spec, policy.normalization)
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
	policy mtpPolicy,
	offset uint32,
) (logits, nextHidden *tensor.Tensor, err error) {
	if builder == nil || input == nil || outputNorm == nil || output == nil {
		return nil, nil, errors.New(policy.label + " MTP output is nil")
	}
	if !policy.valid(spec, offset) || input.Shape.Rank != 2 || input.Shape.Dims[0] != uint64(spec.EmbeddingLength) {
		return nil, nil, errors.New(policy.label + " MTP output shape is incompatible")
	}
	normalized := normalizeMTP(builder, input, outputNorm, spec, policy.normalization)
	nextHidden = normalized
	if policy.carryRaw {
		nextHidden = input
	}
	logits = builder.MulMat(output, normalized)
	if policy.scaleLogits {
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
