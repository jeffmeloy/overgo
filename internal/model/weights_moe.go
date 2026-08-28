package model

import (
	"errors"

	"overgo/internal/tensor"
	"overgo/internal/tensor/dtype"
)

func layerUsesMoECatalog(catalog weightCatalog, prefix string, spec Spec, block uint32, draft bool) bool {
	_, routerPresent := catalog.tensors[prefix+"ffn_gate_inp.weight"]
	return spec.Profile().Experts.usesCatalog(spec, block, routerPresent, draft)
}

func compileMoEBindings(
	catalog weightCatalog,
	prefix string,
	spec Spec,
	layer *LayerWeights,
) (bool, []tensorBinding, error) {
	policy := spec.Profile().Experts
	shapes := spec.TensorShapes(tensor.FirstOffset)
	projectionWidth := shapes.Embedding
	var bindings []tensorBinding
	if policy.SupplementalCatalog.has(expertSupplementLatentProjection) {
		projectionWidth = uint64(spec.MoELatentSize)
		bindings = append(bindings,
			requiredTensorPointer("ffn_latent_down.weight", &layer.FeedForwardLatentDown,
				shapes.Embedding, projectionWidth),
			requiredTensorPointer("ffn_latent_up.weight", &layer.FeedForwardLatentUp,
				projectionWidth, shapes.Embedding),
		)
	}
	fusedGateUp := false
	if policy.FusedGateUp {
		_, fusedGateUp = catalog.tensors[prefix+"ffn_gate_up_exps.weight"]
		bindings = append(bindings, optionalTensorPointer(
			"ffn_gate_up_exps.weight", &layer.FeedForwardGateUpExperts,
			shapes.ExpertUp(FeedForwardFusedGateUp.upProjectionCopies())...,
		))
	}
	bindings = append(bindings,
		requiredTensorPointer("ffn_gate_inp.weight", &layer.FeedForwardRouter, shapes.ExpertRouter()...),
	)
	if !fusedGateUp {
		bindings = append(bindings, requiredTensorPointer(
			"ffn_up_exps.weight", &layer.FeedForwardUpExperts,
			projectionWidth, shapes.ExpertWidth, shapes.Experts,
		))
	}
	bindings = append(bindings, requiredTensorPointer(
		"ffn_down_exps.weight", &layer.FeedForwardDownExperts,
		shapes.ExpertWidth, projectionWidth, shapes.Experts,
	))
	if !fusedGateUp && !policy.OptionalGate {
		bindings = append(bindings, requiredTensorPointer(
			"ffn_gate_exps.weight", &layer.FeedForwardGateExperts,
			shapes.ExpertUp(FeedForwardSwiGLU.upProjectionCopies())...,
		))
	}
	if policy.OptionalGate {
		bindings = append(bindings, optionalTensorPointer(
			"ffn_gate_exps.weight", &layer.FeedForwardGateExperts,
			shapes.ExpertUp(FeedForwardSwiGLU.upProjectionCopies())...,
		))
	}
	if policy.SupplementalCatalog.has(expertSupplementScaledSandwichNorm) {
		bindings = append(bindings,
			requiredTensorPointer("ffn_gate_inp.scale", &layer.FeedForwardRouterScale, shapes.Embedding),
			optionalTensorPointer("ffn_down_exps.scale", &layer.FeedForwardDownExpertsScale, shapes.Experts),
			requiredTensorPointer("pre_ffw_norm_2.weight", &layer.FeedForwardPreNorm2, shapes.Embedding),
			requiredTensorPointer("post_ffw_norm_1.weight", &layer.FeedForwardPostNorm1, shapes.Embedding),
			requiredTensorPointer("post_ffw_norm_2.weight", &layer.FeedForwardPostNorm2, shapes.Embedding),
		)
	}
	bindings = append(bindings, expertPolicyBindings(catalog, prefix, spec, layer, policy)...)
	if policy.SupplementalCatalog.has(expertSupplementRequiredProjectionBiases) {
		bindings = append(bindings,
			requiredF32TensorPointer("ffn_gate_inp.bias", &layer.FeedForwardRouterBias, uint64(spec.ExpertCount)),
			requiredF32TensorPointer("ffn_gate_exps.bias", &layer.FeedForwardGateBias, uint64(spec.ExpertFeedForward), uint64(spec.ExpertCount)),
			requiredF32TensorPointer("ffn_up_exps.bias", &layer.FeedForwardUpBias, uint64(spec.ExpertFeedForward), uint64(spec.ExpertCount)),
			requiredF32TensorPointer("ffn_down_exps.bias", &layer.FeedForwardDownBias, uint64(spec.EmbeddingLength), uint64(spec.ExpertCount)),
		)
	}
	if policy.SupplementalCatalog.has(expertSupplementChunkExperts) {
		chunkExperts := uint64(spec.ExpertCount / spec.ExpertsPerGroup)
		bindings = append(bindings,
			requiredTensorPointer("ffn_gate_chexps.weight", &layer.FeedForwardGateChunkExperts, uint64(spec.EmbeddingLength), uint64(spec.ExpertChunkFeedForward), chunkExperts),
			requiredTensorPointer("ffn_up_chexps.weight", &layer.FeedForwardUpChunkExperts, uint64(spec.EmbeddingLength), uint64(spec.ExpertChunkFeedForward), chunkExperts),
			requiredTensorPointer("ffn_down_chexps.weight", &layer.FeedForwardDownChunkExperts, uint64(spec.ExpertChunkFeedForward), uint64(spec.EmbeddingLength), chunkExperts),
		)
	}
	if policy.SupplementalCatalog.has(expertSupplementOptionalDenseGEGLU) {
		dense, err := optionalDenseGEGLUBindings(catalog, prefix, spec, layer)
		return false, append(bindings, dense...), err
	}
	if policy.SupplementalCatalog.has(expertSupplementExpertInputNorm) {
		bindings = append(bindings, requiredTensorPointer(
			"ffn_norm_exps.weight", &layer.FeedForwardExpertNorm, uint64(spec.EmbeddingLength),
		))
	}
	return policy.SupplementalCatalog.has(expertSupplementDenseBranch), bindings, nil
}

func expertPolicyBindings(
	catalog weightCatalog,
	prefix string,
	spec Spec,
	layer *LayerWeights,
	policy ExpertPolicy,
) []tensorBinding {
	var bindings []tensorBinding
	switch policy.BiasCatalog {
	case expertBiasCatalogOptionalF32, expertBiasCatalogOptionalF32Bare:
		name := "exp_probs_b"
		_, present := catalog.tensors[prefix+name]
		if policy.BiasCatalog != expertBiasCatalogOptionalF32Bare || !present {
			name = "exp_probs_b.bias"
			_, present = catalog.tensors[prefix+name]
		}
		if present {
			bindings = append(bindings, requiredF32TensorPointer(
				name, &layer.FeedForwardExpertBias, uint64(spec.ExpertCount),
			))
		}
	case expertBiasCatalogRequired, expertBiasCatalogRequiredF32:
		requirement := requiredTensorPointer(
			"exp_probs_b.bias", &layer.FeedForwardExpertBias, uint64(spec.ExpertCount),
		)
		if policy.BiasCatalog == expertBiasCatalogRequiredF32 {
			requirement.storages = []dtype.Type{dtype.F32}
		}
		bindings = append(bindings, requirement)
	}
	shared := policy.SharedCatalog
	if shared == sharedExpertCatalogAlways || shared == sharedExpertCatalogGated ||
		shared == sharedExpertCatalogUngated ||
		(shared == sharedExpertCatalogWithWidth && spec.SharedExpertFF > tensor.FirstOffset) {
		bindings = append(bindings, sharedExpertBindings(
			uint64(spec.EmbeddingLength), spec, layer, shared,
		)...)
	}
	return bindings
}

func optionalDenseGEGLUBindings(
	catalog weightCatalog,
	prefix string,
	spec Spec,
	layer *LayerWeights,
) ([]tensorBinding, error) {
	names := [...]string{feedForwardGateWeightTensor, feedForwardUpWeightTensor, feedForwardDownWeightTensor}
	present := tensor.FirstOffset
	for _, name := range names {
		if _, ok := catalog.tensors[prefix+name]; ok {
			present++
		}
	}
	if present != tensor.FirstOffset && present != len(names) {
		return nil, errors.New("optional dense GEGLU tensors must be all present or all absent")
	}
	if present == tensor.FirstOffset {
		return nil, nil
	}
	return []tensorBinding{
		requiredTensorPointer(names[tensor.FirstOffset], &layer.FeedForwardGate,
			uint64(spec.EmbeddingLength), uint64(spec.FeedForwardLength)),
		requiredTensorPointer(names[tensor.SingletonExtent], &layer.FeedForwardUp,
			uint64(spec.EmbeddingLength), uint64(spec.FeedForwardLength)),
		requiredTensorPointer(names[tensor.PairedExtent], &layer.FeedForwardDown,
			uint64(spec.FeedForwardLength), uint64(spec.EmbeddingLength)),
	}, nil
}
