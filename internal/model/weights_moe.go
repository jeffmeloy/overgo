package model

import (
	"errors"

	"overgo/internal/tensor"
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
		if err := bindTensorProgram(catalog, prefix, []tensorBinding{
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
		if err := bindTensorProgram(catalog, prefix, []tensorBinding{
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
		if err := bindTensorProgram(catalog, prefix, []tensorBinding{
			requiredTensorPointer("ffn_norm_exps.weight", &layer.FeedForwardExpertNorm, uint64(spec.EmbeddingLength)),
		}); err != nil {
			return false, err
		}
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
	shapes := spec.TensorShapes(tensor.FirstOffset)
	if policy.FusedGateUp {
		if err := bindTensorProgram(catalog, prefix, []tensorBinding{
			optionalTensorPointer("ffn_gate_up_exps.weight", &layer.FeedForwardGateUpExperts,
				shapes.ExpertUp(FeedForwardFusedGateUp.upProjectionCopies())...),
		}); err != nil {
			return false, err
		}
		fusedGateUp = layer.FeedForwardGateUpExperts != nil
	}
	requirements := []tensorBinding{
		requiredTensorPointer("ffn_gate_inp.weight", &layer.FeedForwardRouter, shapes.ExpertRouter()...),
	}
	if !fusedGateUp {
		requirements = append(requirements,
			requiredTensorPointer("ffn_up_exps.weight", &layer.FeedForwardUpExperts,
				shapes.ExpertUp(FeedForwardSwiGLU.upProjectionCopies())...),
		)
	}
	requirements = append(requirements,
		requiredTensorPointer("ffn_down_exps.weight", &layer.FeedForwardDownExperts, shapes.ExpertDown()...),
	)
	if !fusedGateUp && !policy.OptionalGate {
		requirements = append(requirements,
			requiredTensorPointer("ffn_gate_exps.weight", &layer.FeedForwardGateExperts,
				shapes.ExpertUp(FeedForwardSwiGLU.upProjectionCopies())...),
		)
	}
	return fusedGateUp, bindTensorProgram(catalog, prefix, requirements)
}

func loadOptionalExpertGate(
	catalog weightCatalog,
	prefix string,
	spec Spec,
	layer *LayerWeights,
) error {
	return bindTensorProgram(catalog, prefix, []tensorBinding{
		optionalTensorPointer("ffn_gate_exps.weight", &layer.FeedForwardGateExperts,
			spec.TensorShapes(tensor.FirstOffset).ExpertUp(FeedForwardSwiGLU.upProjectionCopies())...),
	})
}

func loadScaledSandwichNormCatalog(
	catalog weightCatalog,
	prefix string,
	spec Spec,
	layer *LayerWeights,
) error {
	return bindTensorProgram(catalog, prefix, []tensorBinding{
		requiredTensorPointer("ffn_gate_inp.scale", &layer.FeedForwardRouterScale, uint64(spec.EmbeddingLength)),
		optionalTensorPointer("ffn_down_exps.scale", &layer.FeedForwardDownExpertsScale, uint64(spec.ExpertCount)),
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
		if err := bindTensorProgram(catalog, prefix, []tensorBinding{requirement}); err != nil {
			return err
		}
	}
	switch policy.SharedCatalog {
	case sharedExpertCatalogAlways:
		return loadSharedExpertWeights(catalog, prefix, uint64(spec.EmbeddingLength), spec, layer, false)
	case sharedExpertCatalogWithWidth:
		if spec.SharedExpertFF > tensor.FirstOffset {
			return loadSharedExpertWeights(catalog, prefix, uint64(spec.EmbeddingLength), spec, layer, false)
		}
	case sharedExpertCatalogGated:
		return loadSharedExpertWeights(catalog, prefix, uint64(spec.EmbeddingLength), spec, layer, true)
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
	return bindTensorProgram(catalog, prefix, []tensorBinding{
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
	present := tensor.FirstOffset
	for _, name := range names {
		if _, ok := catalog.tensors[prefix+name]; ok {
			present++
		}
	}
	if present != tensor.FirstOffset && present != len(names) {
		return errors.New("optional dense GEGLU tensors must be all present or all absent")
	}
	if present == tensor.FirstOffset {
		return nil
	}
	return bindTensorProgram(catalog, prefix, []tensorBinding{
		requiredTensorPointer(names[tensor.FirstOffset], &layer.FeedForwardGate,
			uint64(spec.EmbeddingLength), uint64(spec.FeedForwardLength)),
		requiredTensorPointer(names[tensor.SingletonExtent], &layer.FeedForwardUp,
			uint64(spec.EmbeddingLength), uint64(spec.FeedForwardLength)),
		requiredTensorPointer(names[tensor.PairedExtent], &layer.FeedForwardDown,
			uint64(spec.FeedForwardLength), uint64(spec.EmbeddingLength)),
	})
}
