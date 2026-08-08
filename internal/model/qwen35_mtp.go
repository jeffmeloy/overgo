package model

import (
	"errors"

	"overgo/internal/tensor"
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
	plan LayerPlan,
) (DenseBlockResult, error) {
	if !qwen35MTPPolicy.valid(spec, 0) || plan.Layer != 0 || plan.Recurrent {
		return DenseBlockResult{}, errors.New("Qwen3.5 MTP program is invalid")
	}
	dense := qwen35MTPExecutableSpec(spec)
	return BuildArchitectureBlockCached(BlockDispatchOptions{
		Spec: dense, Weights: weights, Plan: &plan,
		Context: CachedBlockContext{
			Builder: builder, Input: input, Positions: positions,
			PastKey: pastKey, PastValue: pastValue, CacheWrite: tensor.CacheWriteConcat,
			Sequences: 1,
		},
	})
}

func qwen35MTPExecutableSpec(spec Spec) Spec {
	profile, _ := LookupArchitecture("qwen35")
	spec.Architecture = profile.Name
	return spec.withProfile(profile)
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
