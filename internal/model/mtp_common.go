package model

import (
	"errors"

	"overgo/internal/tensor"
)

// BuildDraftInput executes the compiled draft-input policy.
func (p CompiledLayerProgram) BuildDraftInput(
	builder *tensor.Builder,
	tokenEmbedding, targetHidden, embeddingNorm, hiddenNorm, projection *tensor.Tensor,
) (*tensor.Tensor, error) {
	if p.draft.Kind == DraftNone {
		return nil, errors.New("compiled layer has no draft-input policy")
	}
	return buildMTPInput(
		builder, tokenEmbedding, targetHidden, embeddingNorm, hiddenNorm, projection,
		p.spec, p.plan.Normalization, p.draft,
	)
}

// BuildDraftOutputs executes the compiled draft-output policy.
func (p CompiledLayerProgram) BuildDraftOutputs(
	builder *tensor.Builder,
	input, outputNorm, output *tensor.Tensor,
) (logits, nextHidden *tensor.Tensor, err error) {
	if p.draft.Kind == DraftNone {
		return nil, nil, errors.New("compiled layer has no draft-output policy")
	}
	return buildMTPOutputs(
		builder, input, outputNorm, output, p.spec, p.plan.Normalization, p.draft,
	)
}

func draftExecutableSpec(spec Spec, plan DraftPlan) (Spec, uint32) {
	if plan.OptionalCatalog {
		layer := spec.BlockCount
		spec.BlockCount++
		spec.LeadingDenseBlocks = spec.BlockCount
		spec.SlidingWindow = 0
		return spec, layer
	}
	if !plan.SingleCatalog {
		return spec, spec.BlockCount
	}
	profile := spec.Profile()
	profile.Capabilities &^= ArchitectureMoE
	profile.Experts = ExpertPolicy{}
	return spec.withProfile(profile), 0
}

func buildMTPInput(
	builder *tensor.Builder,
	tokenEmbedding, targetHidden, embeddingNorm, hiddenNorm, projection *tensor.Tensor,
	spec Spec,
	architectureNormalization NormalizationPlan,
	plan DraftPlan,
) (*tensor.Tensor, error) {
	if builder == nil || tokenEmbedding == nil || targetHidden == nil || embeddingNorm == nil ||
		hiddenNorm == nil || projection == nil {
		return nil, errors.New("draft input is nil")
	}
	if tokenEmbedding.Shape.Rank != 2 || !tokenEmbedding.Shape.Equal(targetHidden.Shape) ||
		tokenEmbedding.Shape.Dims[0] != uint64(spec.EmbeddingLength) {
		return nil, errors.New("draft input shape is incompatible")
	}
	embedding := normalizeMTP(builder, tokenEmbedding, embeddingNorm, spec, architectureNormalization, plan.Normalization)
	hidden := normalizeMTP(builder, targetHidden, hiddenNorm, spec, architectureNormalization, plan.Normalization)
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
	architectureNormalization NormalizationPlan,
	plan DraftPlan,
) (logits, nextHidden *tensor.Tensor, err error) {
	if builder == nil || input == nil || outputNorm == nil || output == nil {
		return nil, nil, errors.New("draft output is nil")
	}
	if input.Shape.Rank != 2 || input.Shape.Dims[0] != uint64(spec.EmbeddingLength) {
		return nil, nil, errors.New("draft output shape is incompatible")
	}
	normalized := normalizeMTP(builder, input, outputNorm, spec, architectureNormalization, plan.Normalization)
	nextHidden = normalized
	if plan.CarryRawHidden {
		nextHidden = input
	}
	logits = builder.MulMat(output, normalized)
	if plan.ScaleLogits {
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
	architecture NormalizationPlan,
	normalization DraftNormalizationPolicy,
) *tensor.Tensor {
	if normalization == DraftNormalizationArchitecture {
		return architecture.Apply(builder, input, weight, nil)
	}
	return builder.WeightedRMSNorm(input, weight, spec.RMSNormEpsilon)
}
