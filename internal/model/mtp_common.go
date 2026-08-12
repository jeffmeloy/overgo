package model

import (
	"errors"

	"overgo/internal/tensor"
)

type mtpNormalization uint8

const (
	mtpWeightedRMS mtpNormalization = iota
	mtpArchitectureNorm
)

type mtpPolicy struct {
	label         string
	normalization mtpNormalization
	carryRaw      bool
	scaleLogits   bool
}

var draftMTPPolicies = [...]mtpPolicy{
	DraftQwen35MTP:  {label: "Qwen3.5"},
	DraftStep35MTP:  {label: "Step3.5", carryRaw: true},
	DraftHYV3MTP:    {label: "HY-V3"},
	DraftNextNMTP:   {label: "NextN"},
	DraftCohere2MTP: {label: "Cohere2-MoE", normalization: mtpArchitectureNorm, scaleLogits: true},
}

func draftMTPPolicy(kind DraftKind) (mtpPolicy, bool) {
	if kind == DraftNone || int(kind) >= len(draftMTPPolicies) {
		return mtpPolicy{}, false
	}
	return draftMTPPolicies[kind], true
}

// BuildDraftInput executes the compiled draft-input policy.
func (p CompiledLayerProgram) BuildDraftInput(
	builder *tensor.Builder,
	tokenEmbedding, targetHidden, embeddingNorm, hiddenNorm, projection *tensor.Tensor,
) (*tensor.Tensor, error) {
	policy, ok := draftMTPPolicy(p.draftKind)
	if !ok {
		return nil, errors.New("compiled layer has no draft-input policy")
	}
	return buildMTPInput(
		builder, tokenEmbedding, targetHidden, embeddingNorm, hiddenNorm, projection,
		p.spec, policy,
	)
}

// BuildDraftOutputs executes the compiled draft-output policy.
func (p CompiledLayerProgram) BuildDraftOutputs(
	builder *tensor.Builder,
	input, outputNorm, output *tensor.Tensor,
) (logits, nextHidden *tensor.Tensor, err error) {
	policy, ok := draftMTPPolicy(p.draftKind)
	if !ok {
		return nil, nil, errors.New("compiled layer has no draft-output policy")
	}
	return buildMTPOutputs(
		builder, input, outputNorm, output, p.spec, policy,
	)
}

func singleDraftExecutableSpec(spec Spec, kind DraftKind) (Spec, uint32) {
	switch kind {
	case DraftQwen35MTP:
		profile, _ := LookupArchitecture("qwen35")
		spec.Architecture = profile.Name
		return spec.withProfile(profile), 0
	case DraftCohere2MTP:
		layer := spec.BlockCount
		spec.BlockCount++
		spec.LeadingDenseBlocks = spec.BlockCount
		spec.SlidingWindow = 0
		return spec, layer
	default:
		return spec, spec.BlockCount
	}
}

func buildMTPInput(
	builder *tensor.Builder,
	tokenEmbedding, targetHidden, embeddingNorm, hiddenNorm, projection *tensor.Tensor,
	spec Spec,
	policy mtpPolicy,
) (*tensor.Tensor, error) {
	if builder == nil || tokenEmbedding == nil || targetHidden == nil || embeddingNorm == nil ||
		hiddenNorm == nil || projection == nil {
		return nil, errors.New(policy.label + " MTP input is nil")
	}
	if tokenEmbedding.Shape.Rank != 2 || !tokenEmbedding.Shape.Equal(targetHidden.Shape) ||
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
) (logits, nextHidden *tensor.Tensor, err error) {
	if builder == nil || input == nil || outputNorm == nil || output == nil {
		return nil, nil, errors.New(policy.label + " MTP output is nil")
	}
	if input.Shape.Rank != 2 || input.Shape.Dims[0] != uint64(spec.EmbeddingLength) {
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
