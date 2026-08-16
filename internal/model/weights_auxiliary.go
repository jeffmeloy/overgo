package model

import (
	"errors"
	"fmt"

	"overgo/internal/gguf"
	"overgo/internal/tensor/dtype"
)

func readTargetFeatureWeightCatalog(catalog weightCatalog, spec Spec) (Weights, error) {
	width := uint64(spec.EmbeddingLength)
	result := Weights{Layers: make([]LayerWeights, spec.BlockCount)}
	if err := loadTensorRequirements(catalog, "", []tensorRequirement{
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
		if err := loadTensorRequirements(catalog, prefix, []tensorRequirement{
			requiredTensorPointer("attn_norm.weight", &layer.AttentionNorm, width),
			requiredTensorPointer("attn_q.weight", &layer.AttentionQ, width, query),
			requiredTensorPointer("attn_k.weight", &layer.AttentionK, width, key),
			requiredTensorPointer("attn_v.weight", &layer.AttentionV, width, value),
			requiredTensorPointer("attn_output.weight", &layer.AttentionOutput, query, width),
			requiredTensorPointer("attn_q_norm.weight", &layer.AttentionQNorm, uint64(spec.KeyLength)),
			requiredTensorPointer("attn_k_norm.weight", &layer.AttentionKNorm, uint64(spec.KeyLength)),
		}); err != nil {
			return Weights{}, err
		}
		if err := loadStandardSwiGLUCatalog(catalog, prefix, spec, layer); err != nil {
			return Weights{}, err
		}
	}
	return result, nil
}

func readHiddenFusionWeightCatalog(catalog weightCatalog, spec Spec) (Weights, error) {
	width := uint64(spec.EmbeddingLength)
	draftVocabulary := uint64(spec.VocabularySize)
	result := Weights{Layers: make([]LayerWeights, 1)}
	if err := loadTensorRequirements(catalog, "", []tensorRequirement{
		optionalRelationalTensorPointer("d2t", &result.DraftToTarget, 1, true, dtype.I64),
	}); err != nil {
		return Weights{}, err
	}
	if result.DraftToTarget != nil {
		draftVocabulary = result.DraftToTarget.Shape[0]
	}
	if err := loadTensorRequirements(catalog, "", []tensorRequirement{
		requiredTensorPointer("fc.weight", &result.FeatureProjection, 3*uint64(spec.TargetHiddenSize), width),
		requiredTensor("output_norm.weight", &result.OutputNorm, width),
		optionalTensor("token_embd.weight", &result.TokenEmbedding, width, uint64(spec.VocabularySize)),
		optionalTensorPointer("output.weight", &result.Output, width, draftVocabulary),
	}); err != nil {
		return Weights{}, err
	}
	if result.DraftToTarget != nil && result.Output == nil {
		return Weights{}, errors.New(`required tensor "output.weight" is missing for hidden-fusion vocabulary mapping`)
	}
	layer := &result.Layers[0]
	query := uint64(spec.HeadCount) * uint64(spec.KeyLength)
	key := uint64(spec.HeadCountKV) * uint64(spec.KeyLength)
	value := uint64(spec.HeadCountKV) * uint64(spec.ValueLength)
	if err := loadTensorRequirements(catalog, "blk.0.", []tensorRequirement{
		requiredTensorPointer("attn_norm.weight", &layer.AttentionNorm, width),
		requiredTensorPointer("attn_q.weight", &layer.AttentionQ, 2*width, query),
		requiredTensorPointer("attn_k.weight", &layer.AttentionK, 2*width, key),
		requiredTensorPointer("attn_v.weight", &layer.AttentionV, 2*width, value),
		requiredTensorPointer("attn_output.weight", &layer.AttentionOutput, query, width),
		requiredTensorPointer("attn_norm_2.weight", &layer.AttentionNorm2, width),
		optionalTensorPointer("rope_freqs.weight", &layer.RopeFactors, uint64(spec.RopeDimensionCount/2)),
	}); err != nil {
		return Weights{}, err
	}
	if err := loadStandardSwiGLUCatalog(catalog, "blk.0.", spec, layer); err != nil {
		return Weights{}, err
	}
	return result, nil
}

func readPairedProjectionWeightCatalog(catalog weightCatalog, spec Spec) (Weights, error) {
	width, targetWidth := uint64(spec.EmbeddingLength), uint64(spec.TargetHiddenSize)
	result := Weights{Layers: make([]LayerWeights, spec.BlockCount)}
	if err := loadTensorRequirements(catalog, "", []tensorRequirement{
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
		if err := loadTensorRequirements(catalog, prefix, []tensorRequirement{
			requiredTensorPointer("attn_norm.weight", &layer.AttentionNorm, width),
			requiredTensorPointer("attn_q.weight", &layer.AttentionQ, width, query),
			requiredTensorPointer("attn_output.weight", &layer.AttentionOutput, output, width),
			requiredTensorPointer("attn_q_norm.weight", &layer.AttentionQNorm, uint64(spec.LayerKeyLength(block))),
			requiredTensorPointer("post_attention_norm.weight", &layer.AttentionPostNorm, width),
			requiredTensorPointer("post_ffw_norm.weight", &layer.FeedForwardPostNorm, width),
			requiredTensorPointer("layer_output_scale.weight", &layer.LayerOutputScale, 1),
		}); err != nil {
			return Weights{}, err
		}
		if err := loadStandardSwiGLUCatalog(catalog, prefix, spec, layer); err != nil {
			return Weights{}, err
		}
		if !spec.IsSlidingLayer(block) {
			rope, ok := catalog.tensor(prefix + "rope_freqs.weight")
			if !ok {
				rope, ok = catalog.tensor("rope_freqs.weight")
			}
			if !ok && sharedRope != nil {
				rope, ok = *sharedRope, true
			}
			if !ok {
				return Weights{}, fmt.Errorf("required paired-projection RoPE factors for layer %d are missing", block)
			}
			if ropeErr := validateTensorInfo(
				rope, []dtype.Type{dtype.F32}, []uint64{uint64(spec.RopeDimensionCount / 2)},
			); ropeErr != nil {
				return Weights{}, ropeErr
			}
			layer.RopeFactors, sharedRope = &rope, &rope
		}
	}
	return result, nil
}
