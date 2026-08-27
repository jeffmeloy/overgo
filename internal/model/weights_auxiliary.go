package model

import (
	"fmt"

	"overgo/internal/gguf"
	"overgo/internal/tensor"
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
	for block := uint32(tensor.FirstOffset); block < spec.BlockCount; block++ {
		prefix := fmt.Sprintf("blk.%d.", block)
		layer := &result.Layers[block]
		bindings := []tensorBinding{
			requiredTensorPointer(attentionNormWeightTensor, &layer.AttentionNorm, width),
			requiredTensorPointer(attentionQueryWeightTensor, &layer.AttentionQ, width, query),
			requiredTensorPointer(attentionKeyWeightTensor, &layer.AttentionK, width, key),
			requiredTensorPointer(attentionValueWeightTensor, &layer.AttentionV, width, value),
			requiredTensorPointer(attentionOutputWeightTensor, &layer.AttentionOutput, query, width),
			requiredTensorPointer(attentionQueryNormTensor, &layer.AttentionQNorm, uint64(spec.KeyLength)),
			requiredTensorPointer(attentionKeyNormTensor, &layer.AttentionKNorm, uint64(spec.KeyLength)),
		}
		if err := bindTensorProgram(catalog, prefix, append(bindings, standardSwiGLUBindings(spec, layer)...)); err != nil {
			return Weights{}, err
		}
	}
	return result, nil
}

func readHiddenFusionWeightCatalog(catalog weightCatalog, spec Spec) (Weights, error) {
	width := uint64(spec.EmbeddingLength)
	draftVocabulary := uint64(spec.VocabularySize)
	result := Weights{Layers: make([]LayerWeights, tensor.SingletonExtent)}
	if err := bindTensorProgram(catalog, "", []tensorBinding{
		optionalRelationalTensorPointer("d2t", &result.DraftToTarget, tensor.SingletonExtent, true, dtype.I64),
	}); err != nil {
		return Weights{}, err
	}
	if result.DraftToTarget != nil {
		draftVocabulary = result.DraftToTarget.Shape[tensor.FirstOffset]
	}
	if err := bindTensorProgram(catalog, "", []tensorBinding{
		requiredTensorPointer("fc.weight", &result.FeatureProjection,
			tensor.TripleExtent*uint64(spec.TargetHiddenSize), width),
		requiredTensor(outputNormWeightTensor, &result.OutputNorm, width),
		optionalTensor(tokenEmbeddingWeightTensor, &result.TokenEmbedding, width, uint64(spec.VocabularySize)),
		optionalTensorPointer(outputWeightTensor, &result.Output, width, draftVocabulary),
	}); err != nil {
		return Weights{}, err
	}
	if result.DraftToTarget != nil && result.Output == nil {
		return Weights{}, fmt.Errorf("required tensor %q is missing for hidden-fusion vocabulary mapping", outputWeightTensor)
	}
	layer := &result.Layers[tensor.FirstOffset]
	query := uint64(spec.HeadCount) * uint64(spec.KeyLength)
	key := uint64(spec.HeadCountKV) * uint64(spec.KeyLength)
	value := uint64(spec.HeadCountKV) * uint64(spec.ValueLength)
	bindings := []tensorBinding{
		requiredTensorPointer(attentionNormWeightTensor, &layer.AttentionNorm, width),
		requiredTensorPointer(attentionQueryWeightTensor, &layer.AttentionQ, tensor.PairedExtent*width, query),
		requiredTensorPointer(attentionKeyWeightTensor, &layer.AttentionK, tensor.PairedExtent*width, key),
		requiredTensorPointer(attentionValueWeightTensor, &layer.AttentionV, tensor.PairedExtent*width, value),
		requiredTensorPointer(attentionOutputWeightTensor, &layer.AttentionOutput, query, width),
		requiredTensorPointer("attn_norm_2.weight", &layer.AttentionNorm2, width),
		optionalTensorPointer("rope_freqs.weight", &layer.RopeFactors,
			uint64(spec.RopeDimensionCount/rotaryPairAlignment)),
	}
	if err := bindTensorProgram(catalog, firstBlockTensorPrefix,
		append(bindings, standardSwiGLUBindings(spec, layer)...)); err != nil {
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
		requiredTensorPointer(firstBlockTensorPrefix+"nextn.pre_projection.weight", &result.FeatureProjection,
			tensor.PairedExtent*targetWidth, width),
		requiredTensorPointer("nextn.post_projection.weight", &result.FeatureProjectionPost, width, targetWidth),
	}); err != nil {
		return Weights{}, err
	}
	var sharedRope *gguf.TensorInfo
	for block := uint32(tensor.FirstOffset); block < spec.BlockCount; block++ {
		prefix := fmt.Sprintf("blk.%d.", block)
		layer := &result.Layers[block]
		query := uint64(spec.HeadCount) * uint64(spec.LayerKeyLength(block))
		output := uint64(spec.HeadCount) * uint64(spec.LayerValueLength(block))
		bindings := []tensorBinding{
			requiredTensorPointer(attentionNormWeightTensor, &layer.AttentionNorm, width),
			requiredTensorPointer(attentionQueryWeightTensor, &layer.AttentionQ, width, query),
			requiredTensorPointer(attentionOutputWeightTensor, &layer.AttentionOutput, output, width),
			requiredTensorPointer(attentionQueryNormTensor, &layer.AttentionQNorm, uint64(spec.LayerKeyLength(block))),
			requiredTensorPointer(postAttentionNormWeightTensor, &layer.AttentionPostNorm, width),
			requiredTensorPointer("post_ffw_norm.weight", &layer.FeedForwardPostNorm, width),
			requiredTensorPointer("layer_output_scale.weight", &layer.LayerOutputScale, tensor.SingletonExtent),
		}
		if err := bindTensorProgram(catalog, prefix, append(bindings, standardSwiGLUBindings(spec, layer)...)); err != nil {
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
				rope, []dtype.Type{dtype.F32},
				[]uint64{uint64(spec.RopeDimensionCount / rotaryPairAlignment)},
			); ropeErr != nil {
				return Weights{}, ropeErr
			}
			layer.RopeFactors, sharedRope = &rope, &rope
		}
	}
	return result, nil
}
