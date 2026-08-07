package model

import (
	"fmt"

	"overgo/internal/gguf"
)

func readT5EncoderWeightCatalog(catalog weightCatalog, spec Spec) (Weights, error) {
	required, tensors := catalog.required, catalog.tensors
	plan := newT5CatalogPlan(spec)
	result := Weights{Layers: make([]LayerWeights, spec.BlockCount)}
	if err := loadTensorRequirements(required, tensors, "", []tensorRequirement{
		requiredTensor("token_embd.weight", &result.TokenEmbedding, plan.width, uint64(spec.VocabularySize)),
		requiredTensor("enc.output_norm.weight", &result.OutputNorm, plan.width),
	}); err != nil {
		return Weights{}, err
	}
	for block := uint32(0); block < spec.BlockCount; block++ {
		prefix := fmt.Sprintf("enc.blk.%d.", block)
		if err := plan.loadLayer(required, tensors, prefix, &result.Layers[block], true, nil); err != nil {
			return Weights{}, err
		}
	}
	return result, nil
}

func readT5WeightCatalog(catalog weightCatalog, spec Spec) (Weights, error) {
	required, tensors := catalog.required, catalog.tensors
	plan := newT5CatalogPlan(spec)
	result := Weights{
		EncoderLayers: make([]LayerWeights, spec.BlockCount),
		Layers:        make([]LayerWeights, spec.DecoderBlockCount),
	}
	if err := loadTensorRequirements(required, tensors, "", []tensorRequirement{
		requiredTensor("token_embd.weight", &result.TokenEmbedding, plan.width, uint64(spec.VocabularySize)),
		requiredTensor("dec.output_norm.weight", &result.OutputNorm, plan.width),
		requiredTensorPointer("enc.output_norm.weight", &result.EncoderOutputNorm, plan.width),
		optionalTensorPointer("output.weight", &result.Output, plan.width, uint64(spec.VocabularySize)),
	}); err != nil {
		return Weights{}, err
	}
	var encoderRelativeBias *gguf.TensorInfo
	for block := uint32(0); block < spec.BlockCount; block++ {
		prefix := fmt.Sprintf("enc.blk.%d.", block)
		if err := plan.loadLayer(
			required, tensors, prefix, &result.EncoderLayers[block], false, &encoderRelativeBias,
		); err != nil {
			return Weights{}, err
		}
	}
	var decoderRelativeBias *gguf.TensorInfo
	for block := uint32(0); block < spec.DecoderBlockCount; block++ {
		prefix := fmt.Sprintf("dec.blk.%d.", block)
		layer := &result.Layers[block]
		if err := plan.loadLayer(required, tensors, prefix, layer, false, &decoderRelativeBias); err != nil {
			return Weights{}, err
		}
		if err := loadTensorRequirements(required, tensors, prefix, []tensorRequirement{
			requiredTensorPointer("cross_attn_norm.weight", &layer.CrossAttentionNorm, plan.width),
			requiredTensorPointer("cross_attn_q.weight", &layer.CrossAttentionQ, plan.width, plan.query),
			requiredTensorPointer("cross_attn_k.weight", &layer.CrossAttentionK, plan.width, plan.key),
			requiredTensorPointer("cross_attn_v.weight", &layer.CrossAttentionV, plan.width, plan.value),
			requiredTensorPointer("cross_attn_o.weight", &layer.CrossAttentionOutput, plan.output, plan.width),
		}); err != nil {
			return Weights{}, err
		}
	}
	return result, nil
}
