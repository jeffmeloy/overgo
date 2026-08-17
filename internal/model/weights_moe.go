package model

import (
	"errors"

	"overgo/internal/tensor/dtype"
)

func layerUsesMoECatalog(
	catalog weightCatalog,
	prefix string,
	spec Spec,
	block uint32,
	isDraftBlock bool,
) bool {
	_, routerPresent := catalog.tensors[prefix+"ffn_gate_inp.weight"]
	return spec.Profile().Experts.usesCatalog(spec, block, routerPresent, isDraftBlock)
}

func loadMoECatalog(
	catalog weightCatalog,
	prefix string,
	spec Spec,
	layer *LayerWeights,
) (bool, error) {
	policy := spec.Profile().Experts
	_, err := loadMoECoreCatalog(catalog, prefix, spec, layer)
	if err != nil {
		return false, err
	}
	if policy.OptionalGate {
		if err := loadOptionalExpertGate(catalog, prefix, spec, layer); err != nil {
			return false, err
		}
	}
	if policy.SupplementalCatalog.has(expertSupplementScaledSandwichNorm) {
		if err := loadScaledSandwichNormCatalog(catalog, prefix, spec, layer); err != nil {
			return false, err
		}
	}
	if err := loadMoEPolicyCatalog(catalog, prefix, spec, layer, policy); err != nil {
		return false, err
	}
	if policy.SupplementalCatalog.has(expertSupplementRequiredProjectionBiases) {
		if err := loadTensorRequirements(catalog, prefix, []tensorRequirement{
			requiredF32TensorPointer("ffn_gate_inp.bias", &layer.FeedForwardRouterBias, uint64(spec.ExpertCount)),
			requiredF32TensorPointer("ffn_gate_exps.bias", &layer.FeedForwardGateBias, uint64(spec.ExpertFeedForward), uint64(spec.ExpertCount)),
			requiredF32TensorPointer("ffn_up_exps.bias", &layer.FeedForwardUpBias, uint64(spec.ExpertFeedForward), uint64(spec.ExpertCount)),
			requiredF32TensorPointer("ffn_down_exps.bias", &layer.FeedForwardDownBias, uint64(spec.EmbeddingLength), uint64(spec.ExpertCount)),
		}); err != nil {
			return false, err
		}
	}
	if policy.SupplementalCatalog.has(expertSupplementChunkExperts) {
		chunkExperts := uint64(spec.ExpertCount / spec.ExpertsPerGroup)
		if err := loadTensorRequirements(catalog, prefix, []tensorRequirement{
			requiredTensorPointer("ffn_gate_chexps.weight", &layer.FeedForwardGateChunkExperts, uint64(spec.EmbeddingLength), uint64(spec.ExpertChunkFeedForward), chunkExperts),
			requiredTensorPointer("ffn_up_chexps.weight", &layer.FeedForwardUpChunkExperts, uint64(spec.EmbeddingLength), uint64(spec.ExpertChunkFeedForward), chunkExperts),
			requiredTensorPointer("ffn_down_chexps.weight", &layer.FeedForwardDownChunkExperts, uint64(spec.ExpertChunkFeedForward), uint64(spec.EmbeddingLength), chunkExperts),
		}); err != nil {
			return false, err
		}
	}
	if policy.SupplementalCatalog.has(expertSupplementOptionalDenseGEGLU) {
		return false, loadOptionalDenseGEGLUCatalog(catalog, prefix, spec, layer)
	}
	if policy.SupplementalCatalog.has(expertSupplementExpertInputNorm) {
		expertNorm, err := catalog.required(prefix+"ffn_norm_exps.weight", uint64(spec.EmbeddingLength))
		if err != nil {
			return false, err
		}
		layer.FeedForwardExpertNorm = &expertNorm
	}
	return policy.SupplementalCatalog.has(expertSupplementDenseBranch), nil
}

func loadMoECoreCatalog(
	catalog weightCatalog,
	prefix string,
	spec Spec,
	layer *LayerWeights,
) (bool, error) {
	fusedGateUp := false
	policy := spec.Profile().Experts
	shapes := spec.TensorShapes(0)
	if policy.FusedGateUp {
		if err := loadTensorRequirements(catalog, prefix, []tensorRequirement{
			optionalTensorPointer("ffn_gate_up_exps.weight", &layer.FeedForwardGateUpExperts, shapes.ExpertUp(2)...),
		}); err != nil {
			return false, err
		}
		fusedGateUp = layer.FeedForwardGateUpExperts != nil
	}
	requirements := []tensorRequirement{
		requiredTensorPointer("ffn_gate_inp.weight", &layer.FeedForwardRouter, shapes.ExpertRouter()...),
	}
	if !fusedGateUp {
		requirements = append(requirements,
			requiredTensorPointer("ffn_up_exps.weight", &layer.FeedForwardUpExperts, shapes.ExpertUp(1)...),
		)
	}
	requirements = append(requirements,
		requiredTensorPointer("ffn_down_exps.weight", &layer.FeedForwardDownExperts, shapes.ExpertDown()...),
	)
	if !fusedGateUp && !policy.OptionalGate {
		requirements = append(requirements,
			requiredTensorPointer("ffn_gate_exps.weight", &layer.FeedForwardGateExperts, shapes.ExpertUp(1)...),
		)
	}
	return fusedGateUp, loadTensorRequirements(catalog, prefix, requirements)
}

func loadOptionalExpertGate(
	catalog weightCatalog,
	prefix string,
	spec Spec,
	layer *LayerWeights,
) error {
	return loadTensorRequirements(catalog, prefix, []tensorRequirement{
		optionalTensorPointer("ffn_gate_exps.weight", &layer.FeedForwardGateExperts,
			spec.TensorShapes(0).ExpertUp(1)...),
	})
}

func loadScaledSandwichNormCatalog(
	catalog weightCatalog,
	prefix string,
	spec Spec,
	layer *LayerWeights,
) error {
	routerScale, err := catalog.required(prefix+"ffn_gate_inp.scale", uint64(spec.EmbeddingLength))
	if err != nil {
		return err
	}
	layer.FeedForwardRouterScale = &routerScale
	if _, ok := catalog.tensors[prefix+"ffn_down_exps.scale"]; ok {
		downScale, err := catalog.required(prefix+"ffn_down_exps.scale", uint64(spec.ExpertCount))
		if err != nil {
			return err
		}
		layer.FeedForwardDownExpertsScale = &downScale
	}
	return loadTensorRequirements(catalog, prefix, []tensorRequirement{
		requiredTensorPointer("pre_ffw_norm_2.weight", &layer.FeedForwardPreNorm2, uint64(spec.EmbeddingLength)),
		requiredTensorPointer("post_ffw_norm_1.weight", &layer.FeedForwardPostNorm1, uint64(spec.EmbeddingLength)),
		requiredTensorPointer("post_ffw_norm_2.weight", &layer.FeedForwardPostNorm2, uint64(spec.EmbeddingLength)),
	})
}

func loadMoEPolicyCatalog(
	catalog weightCatalog,
	prefix string,
	spec Spec,
	layer *LayerWeights,
	policy ExpertPolicy,
) error {
	switch policy.BiasCatalog {
	case expertBiasCatalogOptionalF32, expertBiasCatalogOptionalF32Bare:
		if err := loadOptionalF32ExpertBias(
			catalog, prefix, spec, layer, policy.BiasCatalog == expertBiasCatalogOptionalF32Bare,
		); err != nil {
			return err
		}
	case expertBiasCatalogRequired, expertBiasCatalogRequiredF32:
		requirement := requiredTensorPointer(
			"exp_probs_b.bias", &layer.FeedForwardExpertBias, uint64(spec.ExpertCount),
		)
		if policy.BiasCatalog == expertBiasCatalogRequiredF32 {
			requirement.storages = []dtype.Type{dtype.F32}
		}
		if err := loadTensorRequirements(catalog, prefix, []tensorRequirement{requirement}); err != nil {
			return err
		}
	}
	switch policy.SharedCatalog {
	case sharedExpertCatalogAlways:
		return loadSharedExpertWeights(catalog, prefix, spec, layer, false)
	case sharedExpertCatalogWithWidth:
		if spec.SharedExpertFF > 0 {
			return loadSharedExpertWeights(catalog, prefix, spec, layer, false)
		}
	case sharedExpertCatalogGated:
		return loadSharedExpertWeights(catalog, prefix, spec, layer, true)
	}
	return nil
}

func loadOptionalF32ExpertBias(
	catalog weightCatalog,
	prefix string,
	spec Spec,
	layer *LayerWeights,
	allowBare bool,
) error {
	name := "exp_probs_b"
	_, ok := catalog.tensors[prefix+name]
	if !allowBare || !ok {
		name = "exp_probs_b.bias"
		_, ok = catalog.tensors[prefix+name]
	}
	if !ok {
		return nil
	}
	return loadTensorRequirements(catalog, prefix, []tensorRequirement{
		requiredF32TensorPointer(name, &layer.FeedForwardExpertBias, uint64(spec.ExpertCount)),
	})
}

func loadOptionalDenseGEGLUCatalog(
	catalog weightCatalog,
	prefix string,
	spec Spec,
	layer *LayerWeights,
) error {
	names := []string{feedForwardGateWeightTensor, feedForwardUpWeightTensor, feedForwardDownWeightTensor}
	present := 0
	for _, name := range names {
		if _, ok := catalog.tensors[prefix+name]; ok {
			present++
		}
	}
	if present != 0 && present != len(names) {
		return errors.New("optional dense GEGLU tensors must be all present or all absent")
	}
	if present == 0 {
		return nil
	}
	return loadTensorRequirements(catalog, prefix, []tensorRequirement{
		requiredTensorPointer(names[0], &layer.FeedForwardGate,
			uint64(spec.EmbeddingLength), uint64(spec.FeedForwardLength)),
		requiredTensorPointer(names[1], &layer.FeedForwardUp,
			uint64(spec.EmbeddingLength), uint64(spec.FeedForwardLength)),
		requiredTensorPointer(names[2], &layer.FeedForwardDown,
			uint64(spec.FeedForwardLength), uint64(spec.EmbeddingLength)),
	})
}
