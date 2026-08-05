package model

import (
	"errors"
	"fmt"

	"llamacpp2go/internal/gguf"
	"llamacpp2go/internal/tensor/dtype"
)

func layerUsesMoECatalog(
	tensors map[string]gguf.TensorInfo,
	prefix string,
	spec Spec,
	block uint32,
	isNextNBlock bool,
) bool {
	_, routerPresent := tensors[prefix+"ffn_gate_inp.weight"]
	return spec.Profile().Experts.usesCatalog(spec, block, routerPresent, isNextNBlock)
}

func loadMoECatalog(
	required weightRequirementLoader,
	tensors map[string]gguf.TensorInfo,
	prefix string,
	spec Spec,
	layer *LayerWeights,
) (bool, error) {
	_, err := loadMoECoreCatalog(required, tensors, prefix, spec, layer)
	if err != nil {
		return false, err
	}
	if spec.Profile().Experts.OptionalGate {
		if err := loadOptionalExpertGate(tensors, prefix, spec, layer); err != nil {
			return false, err
		}
	}
	if spec.Architecture == "gemma4" {
		if err := loadGemma4MoECatalog(required, tensors, prefix, spec, layer); err != nil {
			return false, err
		}
	}
	if err := loadMoEFamilyCatalog(required, tensors, prefix, spec, layer); err != nil {
		return false, err
	}
	if spec.Architecture == "gpt-oss" {
		if err := loadTensorRequirements(required, tensors, prefix, []tensorRequirement{
			requiredF32TensorPointer("ffn_gate_inp.bias", &layer.FeedForwardRouterBias, uint64(spec.ExpertCount)),
			requiredF32TensorPointer("ffn_gate_exps.bias", &layer.FeedForwardGateBias, uint64(spec.ExpertFeedForward), uint64(spec.ExpertCount)),
			requiredF32TensorPointer("ffn_up_exps.bias", &layer.FeedForwardUpBias, uint64(spec.ExpertFeedForward), uint64(spec.ExpertCount)),
			requiredF32TensorPointer("ffn_down_exps.bias", &layer.FeedForwardDownBias, uint64(spec.EmbeddingLength), uint64(spec.ExpertCount)),
		}); err != nil {
			return false, err
		}
	}
	if spec.Architecture == "grovemoe" {
		chunkExperts := uint64(spec.ExpertCount / spec.ExpertsPerGroup)
		if err := loadTensorRequirements(required, tensors, prefix, []tensorRequirement{
			requiredTensorPointer("ffn_gate_chexps.weight", &layer.FeedForwardGateChunkExperts, uint64(spec.EmbeddingLength), uint64(spec.ExpertChunkFeedForward), chunkExperts),
			requiredTensorPointer("ffn_up_chexps.weight", &layer.FeedForwardUpChunkExperts, uint64(spec.EmbeddingLength), uint64(spec.ExpertChunkFeedForward), chunkExperts),
			requiredTensorPointer("ffn_down_chexps.weight", &layer.FeedForwardDownChunkExperts, uint64(spec.ExpertChunkFeedForward), uint64(spec.EmbeddingLength), chunkExperts),
		}); err != nil {
			return false, err
		}
	}
	if spec.Architecture == "grok" {
		return false, loadGrokDenseCatalog(required, tensors, prefix, spec, layer)
	}
	if spec.Architecture == "arctic" {
		expertNorm, err := required(prefix+"ffn_norm_exps.weight", uint64(spec.EmbeddingLength))
		if err != nil {
			return false, err
		}
		layer.FeedForwardExpertNorm = &expertNorm
	}
	return spec.Architecture == "arctic" || spec.Architecture == "gemma4", nil
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
	if policy.FusedGateUp {
		if item, ok := tensors[prefix+"ffn_gate_up_exps.weight"]; ok {
			if item.Dimensions != 3 || item.Shape[0] != uint64(spec.EmbeddingLength) ||
				item.Shape[1] != 2*uint64(spec.ExpertFeedForward) || item.Shape[2] != uint64(spec.ExpertCount) {
				return false, fmt.Errorf("tensor %q has incompatible shape %v", item.Name, item.Shape)
			}
			layer.FeedForwardGateUpExperts = &item
			fusedGateUp = true
		}
	}
	requirements := []tensorRequirement{
		requiredTensorPointer("ffn_gate_inp.weight", &layer.FeedForwardRouter, uint64(spec.EmbeddingLength), uint64(spec.ExpertCount)),
	}
	if !fusedGateUp {
		requirements = append(requirements,
			requiredTensorPointer("ffn_up_exps.weight", &layer.FeedForwardUpExperts, uint64(spec.EmbeddingLength), uint64(spec.ExpertFeedForward), uint64(spec.ExpertCount)),
		)
	}
	requirements = append(requirements,
		requiredTensorPointer("ffn_down_exps.weight", &layer.FeedForwardDownExperts, uint64(spec.ExpertFeedForward), uint64(spec.EmbeddingLength), uint64(spec.ExpertCount)),
	)
	if !fusedGateUp && !policy.OptionalGate {
		requirements = append(requirements,
			requiredTensorPointer("ffn_gate_exps.weight", &layer.FeedForwardGateExperts, uint64(spec.EmbeddingLength), uint64(spec.ExpertFeedForward), uint64(spec.ExpertCount)),
		)
	}
	return fusedGateUp, loadTensorRequirements(required, tensors, prefix, requirements)
}

func loadOptionalExpertGate(tensors map[string]gguf.TensorInfo, prefix string, spec Spec, layer *LayerWeights) error {
	item, ok := tensors[prefix+"ffn_gate_exps.weight"]
	if !ok {
		return nil
	}
	if item.Dimensions != 3 || item.Shape[0] != uint64(spec.EmbeddingLength) ||
		item.Shape[1] != uint64(spec.ExpertFeedForward) || item.Shape[2] != uint64(spec.ExpertCount) {
		return fmt.Errorf("tensor %q has incompatible shape %v", item.Name, item.Shape)
	}
	layer.FeedForwardGateExperts = &item
	return nil
}

func loadGemma4MoECatalog(
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

func loadMoEFamilyCatalog(
	required weightRequirementLoader,
	tensors map[string]gguf.TensorInfo,
	prefix string,
	spec Spec,
	layer *LayerWeights,
) error {
	architecture := spec.Architecture
	profile := spec.Profile()
	if architecture == "ernie4_5-moe" || architecture == "hy_v3" ||
		profile.Has(ArchitectureDeepSeek2Layout) || architecture == "deepseek2-ocr" ||
		architecture == "exaone-moe" || architecture == "bailingmoe2" || architecture == "dots1" ||
		architecture == "mimo2" || architecture == "step35" {
		if err := loadOptionalF32ExpertBias(tensors, prefix, spec, layer, architecture == "hy_v3"); err != nil {
			return err
		}
	}
	if architecture == "laguna" || architecture == "afmoe" || architecture == "glm4moe" ||
		architecture == "lfm2moe" || architecture == "minimax-m2" {
		bias, err := required(prefix+"exp_probs_b.bias", uint64(spec.ExpertCount))
		if err != nil {
			return err
		}
		if (architecture == "glm4moe" || architecture == "lfm2moe" || architecture == "minimax-m2") && bias.Type != dtype.F32 {
			return fmt.Errorf("tensor %q must use F32 bias storage", bias.Name)
		}
		layer.FeedForwardExpertBias = &bias
	}
	shared := architecture == "llama4" || architecture == "hunyuan-moe" || architecture == "glm4moe" ||
		architecture == "hy_v3" || profile.Has(ArchitectureDeepSeek2Layout) ||
		architecture == "deepseek2-ocr" || architecture == "exaone-moe" || architecture == "bailingmoe2" ||
		architecture == "dots1" || architecture == "bailingmoe" || architecture == "deepseek"
	shared = shared || spec.SharedExpertFF > 0 && (architecture == "granitemoe" || architecture == "granitehybrid" ||
		architecture == "granite" || architecture == "ernie4_5-moe" || architecture == "laguna" ||
		architecture == "afmoe" || architecture == "cohere2moe" || architecture == "step35")
	sharedSwiGLU := architecture == "qwen2moe" || architecture == "qwen3next" || architecture == "qwen35moe"
	if shared || sharedSwiGLU {
		return loadSharedExpertWeights(required, prefix, spec, layer, sharedSwiGLU)
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

func loadGrokDenseCatalog(
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
		return errors.New("Grok dense FFN tensors must be all present or all absent")
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
