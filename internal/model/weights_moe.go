package model

import (
	"errors"
	"fmt"

	"overgo/internal/gguf"
	"overgo/internal/tensor/dtype"
)

func layerUsesMoECatalog(
	tensors map[string]gguf.TensorInfo,
	prefix string,
	spec Spec,
	block uint32,
	isDraftBlock bool,
) bool {
	_, routerPresent := tensors[prefix+"ffn_gate_inp.weight"]
	return spec.Profile().Experts.usesCatalog(spec, block, routerPresent, isDraftBlock)
}

func loadMoECatalog(
	required weightRequirementLoader,
	tensors map[string]gguf.TensorInfo,
	prefix string,
	spec Spec,
	layer *LayerWeights,
) (bool, error) {
	policy := spec.Profile().Experts
	_, err := loadMoECoreCatalog(required, tensors, prefix, spec, layer)
	if err != nil {
		return false, err
	}
	if policy.OptionalGate {
		if err := loadOptionalExpertGate(tensors, prefix, spec, layer); err != nil {
			return false, err
		}
	}
	if policy.SupplementalCatalog.has(expertSupplementScaledSandwichNorm) {
		if err := loadScaledSandwichNormCatalog(required, tensors, prefix, spec, layer); err != nil {
			return false, err
		}
	}
	if err := loadMoEPolicyCatalog(required, tensors, prefix, spec, layer, policy); err != nil {
		return false, err
	}
	if policy.SupplementalCatalog.has(expertSupplementRequiredProjectionBiases) {
		if err := loadTensorRequirements(required, tensors, prefix, []tensorRequirement{
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
		if err := loadTensorRequirements(required, tensors, prefix, []tensorRequirement{
			requiredTensorPointer("ffn_gate_chexps.weight", &layer.FeedForwardGateChunkExperts, uint64(spec.EmbeddingLength), uint64(spec.ExpertChunkFeedForward), chunkExperts),
			requiredTensorPointer("ffn_up_chexps.weight", &layer.FeedForwardUpChunkExperts, uint64(spec.EmbeddingLength), uint64(spec.ExpertChunkFeedForward), chunkExperts),
			requiredTensorPointer("ffn_down_chexps.weight", &layer.FeedForwardDownChunkExperts, uint64(spec.ExpertChunkFeedForward), uint64(spec.EmbeddingLength), chunkExperts),
		}); err != nil {
			return false, err
		}
	}
	if policy.SupplementalCatalog.has(expertSupplementOptionalDenseGEGLU) {
		return false, loadOptionalDenseGEGLUCatalog(required, tensors, prefix, spec, layer)
	}
	if policy.SupplementalCatalog.has(expertSupplementExpertInputNorm) {
		expertNorm, err := required(prefix+"ffn_norm_exps.weight", uint64(spec.EmbeddingLength))
		if err != nil {
			return false, err
		}
		layer.FeedForwardExpertNorm = &expertNorm
	}
	return policy.SupplementalCatalog.has(expertSupplementDenseBranch), nil
}

func loadMoECoreCatalog(
	required weightRequirementLoader,
	tensors map[string]gguf.TensorInfo,
	prefix string,
	spec Spec,
	layer *LayerWeights,
) (bool, error) {
	fusedGateUp := false
	policy := spec.Profile().Experts
	shapes := spec.TensorShapes(0)
	if policy.FusedGateUp {
		if item, ok := tensors[prefix+"ffn_gate_up_exps.weight"]; ok {
			if !tensorInfoMatches(item, shapes.ExpertUp(2)) {
				return false, fmt.Errorf("tensor %q has incompatible shape %v", item.Name, item.Shape)
			}
			layer.FeedForwardGateUpExperts = &item
			fusedGateUp = true
		}
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
	return fusedGateUp, loadTensorRequirements(required, tensors, prefix, requirements)
}

func loadOptionalExpertGate(tensors map[string]gguf.TensorInfo, prefix string, spec Spec, layer *LayerWeights) error {
	item, ok := tensors[prefix+"ffn_gate_exps.weight"]
	if !ok {
		return nil
	}
	if !tensorInfoMatches(item, spec.TensorShapes(0).ExpertUp(1)) {
		return fmt.Errorf("tensor %q has incompatible shape %v", item.Name, item.Shape)
	}
	layer.FeedForwardGateExperts = &item
	return nil
}

func loadScaledSandwichNormCatalog(
	required weightRequirementLoader,
	tensors map[string]gguf.TensorInfo,
	prefix string,
	spec Spec,
	layer *LayerWeights,
) error {
	routerScale, err := required(prefix+"ffn_gate_inp.scale", uint64(spec.EmbeddingLength))
	if err != nil {
		return err
	}
	layer.FeedForwardRouterScale = &routerScale
	if _, ok := tensors[prefix+"ffn_down_exps.scale"]; ok {
		downScale, err := required(prefix+"ffn_down_exps.scale", uint64(spec.ExpertCount))
		if err != nil {
			return err
		}
		layer.FeedForwardDownExpertsScale = &downScale
	}
	return loadTensorRequirements(required, tensors, prefix, []tensorRequirement{
		requiredTensorPointer("pre_ffw_norm_2.weight", &layer.FeedForwardPreNorm2, uint64(spec.EmbeddingLength)),
		requiredTensorPointer("post_ffw_norm_1.weight", &layer.FeedForwardPostNorm1, uint64(spec.EmbeddingLength)),
		requiredTensorPointer("post_ffw_norm_2.weight", &layer.FeedForwardPostNorm2, uint64(spec.EmbeddingLength)),
	})
}

func loadMoEPolicyCatalog(
	required weightRequirementLoader,
	tensors map[string]gguf.TensorInfo,
	prefix string,
	spec Spec,
	layer *LayerWeights,
	policy ExpertPolicy,
) error {
	switch policy.BiasCatalog {
	case expertBiasCatalogOptionalF32, expertBiasCatalogOptionalF32Bare:
		if err := loadOptionalF32ExpertBias(
			tensors, prefix, spec, layer, policy.BiasCatalog == expertBiasCatalogOptionalF32Bare,
		); err != nil {
			return err
		}
	case expertBiasCatalogRequired, expertBiasCatalogRequiredF32:
		bias, err := required(prefix+"exp_probs_b.bias", uint64(spec.ExpertCount))
		if err != nil {
			return err
		}
		if policy.BiasCatalog == expertBiasCatalogRequiredF32 && bias.Type != dtype.F32 {
			return fmt.Errorf("tensor %q must use F32 bias storage", bias.Name)
		}
		layer.FeedForwardExpertBias = &bias
	}
	switch policy.SharedCatalog {
	case sharedExpertCatalogAlways:
		return loadSharedExpertWeights(required, prefix, spec, layer, false)
	case sharedExpertCatalogWithWidth:
		if spec.SharedExpertFF > 0 {
			return loadSharedExpertWeights(required, prefix, spec, layer, false)
		}
	case sharedExpertCatalogGated:
		return loadSharedExpertWeights(required, prefix, spec, layer, true)
	}
	return nil
}

func loadOptionalF32ExpertBias(
	tensors map[string]gguf.TensorInfo,
	prefix string,
	spec Spec,
	layer *LayerWeights,
	allowBare bool,
) error {
	bias, ok := tensors[prefix+"exp_probs_b"]
	if !allowBare || !ok {
		bias, ok = tensors[prefix+"exp_probs_b.bias"]
	}
	if !ok {
		return nil
	}
	if bias.Type != dtype.F32 || bias.Dimensions != 1 || bias.Shape[0] != uint64(spec.ExpertCount) {
		return fmt.Errorf("tensor %q has incompatible shape %v", bias.Name, bias.Shape)
	}
	layer.FeedForwardExpertBias = &bias
	return nil
}

func loadOptionalDenseGEGLUCatalog(
	required weightRequirementLoader,
	tensors map[string]gguf.TensorInfo,
	prefix string,
	spec Spec,
	layer *LayerWeights,
) error {
	names := []string{"ffn_gate.weight", "ffn_up.weight", "ffn_down.weight"}
	present := 0
	for _, name := range names {
		if _, ok := tensors[prefix+name]; ok {
			present++
		}
	}
	if present != 0 && present != len(names) {
		return errors.New("optional dense GEGLU tensors must be all present or all absent")
	}
	if present == 0 {
		return nil
	}
	var err error
	if layer.FeedForwardGate, err = required(prefix+names[0], uint64(spec.EmbeddingLength), uint64(spec.FeedForwardLength)); err != nil {
		return err
	}
	if layer.FeedForwardUp, err = required(prefix+names[1], uint64(spec.EmbeddingLength), uint64(spec.FeedForwardLength)); err != nil {
		return err
	}
	layer.FeedForwardDown, err = required(prefix+names[2], uint64(spec.FeedForwardLength), uint64(spec.EmbeddingLength))
	return err
}
