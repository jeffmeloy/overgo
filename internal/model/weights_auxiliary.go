package model

import (
	"fmt"

	"overgo/internal/gguf"
	"overgo/internal/tensor/dtype"
)

func readTargetFeatureWeightCatalog(catalog weightCatalog, spec Spec) (Weights, error) {
	width := uint64(spec.EmbeddingLength)
	result := Weights{Layers: make([]LayerWeights, spec.BlockCount)}
	if err := bindTensorProgram(catalog, "", []tensorBinding{
		requiredTensorPointer("fc.weight", &result.FeatureProjection, uint64(len(spec.TargetLayers))*width, width),
		requiredTensorPointer("enc.output_norm.weight", &result.EncoderOutputNorm, width),
		requiredTensor(outputNormWeightTensor, &result.OutputNorm, width),
	}); err != nil {
		return Weights{}, err
	}
	query := uint64(spec.HeadCount) * uint64(spec.KeyLength)
	key := uint64(spec.HeadCountKV) * uint64(spec.KeyLength)
	value := uint64(spec.HeadCountKV) * uint64(spec.ValueLength)
	for block := uint32(0); block < spec.BlockCount; block++ {
		prefix := fmt.Sprintf("blk.%d.", block)
		layer := &result.Layers[block]
		if err := bindTensorProgram(catalog, prefix, []tensorBinding{
			requiredTensorPointer(attentionNormWeightTensor, &layer.AttentionNorm, width),
			requiredTensorPointer(attentionQueryWeightTensor, &layer.AttentionQ, width, query),
			requiredTensorPointer(attentionKeyWeightTensor, &layer.AttentionK, width, key),
			requiredTensorPointer(attentionValueWeightTensor, &layer.AttentionV, width, value),
			requiredTensorPointer(attentionOutputWeightTensor, &layer.AttentionOutput, query, width),
			requiredTensorPointer(attentionQueryNormTensor, &layer.AttentionQNorm, uint64(spec.KeyLength)),
			requiredTensorPointer(attentionKeyNormTensor, &layer.AttentionKNorm, uint64(spec.KeyLength)),
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
	if err := bindTensorProgram(catalog, "", []tensorBinding{
		optionalRelationalTensorPointer("d2t", &result.DraftToTarget, 1, true, dtype.I64),
	}); err != nil {
		return Weights{}, err
	}
	if result.DraftToTarget != nil {
		draftVocabulary = result.DraftToTarget.Shape[0]
	}
	if err := bindTensorProgram(catalog, "", []tensorBinding{
		requiredTensorPointer("fc.weight", &result.FeatureProjection, 3*uint64(spec.TargetHiddenSize), width),
		requiredTensor(outputNormWeightTensor, &result.OutputNorm, width),
		optionalTensor(tokenEmbeddingWeightTensor, &result.TokenEmbedding, width, uint64(spec.VocabularySize)),
		optionalTensorPointer(outputWeightTensor, &result.Output, width, draftVocabulary),
	}); err != nil {
		return Weights{}, err
	}
	if result.DraftToTarget != nil && result.Output == nil {
		return Weights{}, fmt.Errorf("required tensor %q is missing for hidden-fusion vocabulary mapping", outputWeightTensor)
	}
	layer := &result.Layers[0]
	query := uint64(spec.HeadCount) * uint64(spec.KeyLength)
	key := uint64(spec.HeadCountKV) * uint64(spec.KeyLength)
	value := uint64(spec.HeadCountKV) * uint64(spec.ValueLength)
	if err := bindTensorProgram(catalog, "blk.0.", []tensorBinding{
		requiredTensorPointer(attentionNormWeightTensor, &layer.AttentionNorm, width),
		requiredTensorPointer(attentionQueryWeightTensor, &layer.AttentionQ, 2*width, query),
		requiredTensorPointer(attentionKeyWeightTensor, &layer.AttentionK, 2*width, key),
		requiredTensorPointer(attentionValueWeightTensor, &layer.AttentionV, 2*width, value),
		requiredTensorPointer(attentionOutputWeightTensor, &layer.AttentionOutput, query, width),
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
	if err := bindTensorProgram(catalog, "", []tensorBinding{
		requiredTensor(tokenEmbeddingWeightTensor, &result.TokenEmbedding, width, uint64(spec.VocabularySize)),
		requiredTensor(outputNormWeightTensor, &result.OutputNorm, width),
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
		if err := bindTensorProgram(catalog, prefix, []tensorBinding{
			requiredTensorPointer(attentionNormWeightTensor, &layer.AttentionNorm, width),
			requiredTensorPointer(attentionQueryWeightTensor, &layer.AttentionQ, width, query),
			requiredTensorPointer(attentionOutputWeightTensor, &layer.AttentionOutput, output, width),
			requiredTensorPointer(attentionQueryNormTensor, &layer.AttentionQNorm, uint64(spec.LayerKeyLength(block))),
			requiredTensorPointer(postAttentionNormWeightTensor, &layer.AttentionPostNorm, width),
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
