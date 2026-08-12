package model

import (
	"errors"
	"fmt"

	"overgo/internal/gguf"
	"overgo/internal/tensor/dtype"
)

func readDraftWeightCatalog(catalog weightCatalog, spec Spec) (Weights, error) {
	policy := spec.Profile().ModelCatalog.Draft
	if policy == DraftWeightCatalogNone || int(policy) >= len(draftWeightCatalogReaders) ||
		draftWeightCatalogReaders[policy] == nil {
		return Weights{}, fmt.Errorf("draft catalog for %q is unsupported", spec.Architecture)
	}
	return draftWeightCatalogReaders[policy](catalog, spec)
}

type draftWeightCatalogReader func(weightCatalog, Spec) (Weights, error)

var draftWeightCatalogReaders = [...]draftWeightCatalogReader{
	DraftWeightCatalogTargetFeatures:   readDFlashWeightCatalog,
	DraftWeightCatalogHiddenFusion:     readEagle3WeightCatalog,
	DraftWeightCatalogPairedProjection: readGemma4AssistantWeightCatalog,
}

func readDFlashWeightCatalog(catalog weightCatalog, spec Spec) (Weights, error) {
	required, tensors := catalog.required, catalog.tensors
	width := uint64(spec.EmbeddingLength)
	result := Weights{Layers: make([]LayerWeights, spec.BlockCount)}
	if err := loadTensorRequirements(required, tensors, "", []tensorRequirement{
		requiredTensorPointer("fc.weight", &result.FeatureProjection, uint64(len(spec.TargetLayers))*width, width),
		requiredTensorPointer("enc.output_norm.weight", &result.EncoderOutputNorm, width),
		requiredTensor("output_norm.weight", &result.OutputNorm, width),
	}); err != nil {
		return Weights{}, err
	}
	query := uint64(spec.HeadCount) * uint64(spec.KeyLength)
	key := uint64(spec.HeadCountKV) * uint64(spec.KeyLength)
	value := uint64(spec.HeadCountKV) * uint64(spec.ValueLength)
	for block := uint32(0); block < spec.BlockCount; block++ {
		prefix := fmt.Sprintf("blk.%d.", block)
		layer := &result.Layers[block]
		if err := loadTensorRequirements(required, tensors, prefix, []tensorRequirement{
			requiredTensor("attn_norm.weight", &layer.AttentionNorm, width),
			requiredTensor("attn_q.weight", &layer.AttentionQ, width, query),
			requiredTensor("attn_k.weight", &layer.AttentionK, width, key),
			requiredTensor("attn_v.weight", &layer.AttentionV, width, value),
			requiredTensor("attn_output.weight", &layer.AttentionOutput, query, width),
			requiredTensor("ffn_norm.weight", &layer.FeedForwardNorm, width),
			requiredTensor("ffn_gate.weight", &layer.FeedForwardGate, width, uint64(spec.FeedForwardLength)),
			requiredTensor("ffn_up.weight", &layer.FeedForwardUp, width, uint64(spec.FeedForwardLength)),
			requiredTensor("ffn_down.weight", &layer.FeedForwardDown, uint64(spec.FeedForwardLength), width),
			requiredTensorPointer("attn_q_norm.weight", &layer.AttentionQNorm, uint64(spec.KeyLength)),
			requiredTensorPointer("attn_k_norm.weight", &layer.AttentionKNorm, uint64(spec.KeyLength)),
		}); err != nil {
			return Weights{}, err
		}
	}
	return result, nil
}

func readEagle3WeightCatalog(catalog weightCatalog, spec Spec) (Weights, error) {
	required, tensors := catalog.required, catalog.tensors
	width := uint64(spec.EmbeddingLength)
	draftVocabulary := uint64(spec.VocabularySize)
	result := Weights{Layers: make([]LayerWeights, 1)}
	if item, ok := tensors["d2t"]; ok {
		if item.Type != dtype.I64 || item.Dimensions != 1 || item.Shape[0] == 0 {
			return Weights{}, fmt.Errorf("tensor %q has incompatible shape/type", item.Name)
		}
		draftVocabulary, result.DraftToTarget = item.Shape[0], &item
	}
	if err := loadTensorRequirements(required, tensors, "", []tensorRequirement{
		requiredTensorPointer("fc.weight", &result.FeatureProjection, 3*uint64(spec.TargetHiddenSize), width),
		requiredTensor("output_norm.weight", &result.OutputNorm, width),
		optionalTensor("token_embd.weight", &result.TokenEmbedding, width, uint64(spec.VocabularySize)),
		optionalTensorPointer("output.weight", &result.Output, width, draftVocabulary),
	}); err != nil {
		return Weights{}, err
	}
	if result.DraftToTarget != nil && result.Output == nil {
		return Weights{}, errors.New(`required tensor "output.weight" is missing for Eagle3 vocabulary mapping`)
	}
	layer := &result.Layers[0]
	query := uint64(spec.HeadCount) * uint64(spec.KeyLength)
	key := uint64(spec.HeadCountKV) * uint64(spec.KeyLength)
	value := uint64(spec.HeadCountKV) * uint64(spec.ValueLength)
	if err := loadTensorRequirements(required, tensors, "blk.0.", []tensorRequirement{
		requiredTensor("attn_norm.weight", &layer.AttentionNorm, width),
		requiredTensor("attn_q.weight", &layer.AttentionQ, 2*width, query),
		requiredTensor("attn_k.weight", &layer.AttentionK, 2*width, key),
		requiredTensor("attn_v.weight", &layer.AttentionV, 2*width, value),
		requiredTensor("attn_output.weight", &layer.AttentionOutput, query, width),
		requiredTensor("ffn_norm.weight", &layer.FeedForwardNorm, width),
		requiredTensor("ffn_gate.weight", &layer.FeedForwardGate, width, uint64(spec.FeedForwardLength)),
		requiredTensor("ffn_up.weight", &layer.FeedForwardUp, width, uint64(spec.FeedForwardLength)),
		requiredTensor("ffn_down.weight", &layer.FeedForwardDown, uint64(spec.FeedForwardLength), width),
		requiredTensorPointer("attn_norm_2.weight", &layer.AttentionNorm2, width),
		optionalTensorPointer("rope_freqs.weight", &layer.RopeFactors, uint64(spec.RopeDimensionCount/2)),
	}); err != nil {
		return Weights{}, err
	}
	return result, nil
}

func readGemma4AssistantWeightCatalog(catalog weightCatalog, spec Spec) (Weights, error) {
	required, tensors := catalog.required, catalog.tensors
	width, targetWidth := uint64(spec.EmbeddingLength), uint64(spec.TargetHiddenSize)
	result := Weights{Layers: make([]LayerWeights, spec.BlockCount)}
	if err := loadTensorRequirements(required, tensors, "", []tensorRequirement{
		requiredTensor("token_embd.weight", &result.TokenEmbedding, width, uint64(spec.VocabularySize)),
		requiredTensor("output_norm.weight", &result.OutputNorm, width),
		requiredTensorPointer("blk.0.nextn.pre_projection.weight", &result.FeatureProjection, 2*targetWidth, width),
		requiredTensorPointer("nextn.post_projection.weight", &result.FeatureProjectionPost, width, targetWidth),
	}); err != nil {
		return Weights{}, err
	}
	var sharedRope *gguf.TensorInfo
	for block := uint32(0); block < spec.BlockCount; block++ {
		prefix := fmt.Sprintf("blk.%d.", block)
		layer := &result.Layers[block]
		query := uint64(spec.HeadCount) * uint64(spec.LayerKeyLength(block))
		output := uint64(spec.HeadCount) * uint64(spec.LayerValueLength(block))
		if err := loadTensorRequirements(required, tensors, prefix, []tensorRequirement{
			requiredTensor("attn_norm.weight", &layer.AttentionNorm, width),
			requiredTensor("attn_q.weight", &layer.AttentionQ, width, query),
			requiredTensor("attn_output.weight", &layer.AttentionOutput, output, width),
			requiredTensor("ffn_norm.weight", &layer.FeedForwardNorm, width),
			requiredTensor("ffn_gate.weight", &layer.FeedForwardGate, width, uint64(spec.FeedForwardLength)),
			requiredTensor("ffn_up.weight", &layer.FeedForwardUp, width, uint64(spec.FeedForwardLength)),
			requiredTensor("ffn_down.weight", &layer.FeedForwardDown, uint64(spec.FeedForwardLength), width),
			requiredTensorPointer("attn_q_norm.weight", &layer.AttentionQNorm, uint64(spec.LayerKeyLength(block))),
			requiredTensorPointer("post_attention_norm.weight", &layer.AttentionPostNorm, width),
			requiredTensorPointer("post_ffw_norm.weight", &layer.FeedForwardPostNorm, width),
			requiredTensorPointer("layer_output_scale.weight", &layer.LayerOutputScale, 1),
		}); err != nil {
			return Weights{}, err
		}
		if !spec.IsSlidingLayer(block) {
			rope, ok := tensors[prefix+"rope_freqs.weight"]
			if !ok {
				rope, ok = tensors["rope_freqs.weight"]
			}
			if !ok && sharedRope != nil {
				rope, ok = *sharedRope, true
			}
			if !ok {
				return Weights{}, fmt.Errorf("required Gemma 4 assistant RoPE factors for layer %d are missing", block)
			}
			if rope.Type != dtype.F32 || rope.Dimensions != 1 || rope.Shape[0] != uint64(spec.RopeDimensionCount/2) {
				return Weights{}, fmt.Errorf("tensor %q has incompatible Gemma 4 assistant RoPE factors", rope.Name)
			}
			layer.RopeFactors, sharedRope = &rope, &rope
		}
	}
	return result, nil
}
