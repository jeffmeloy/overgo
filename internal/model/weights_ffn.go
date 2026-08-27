package model

import (
	"overgo/internal/gguf"
	"overgo/internal/tensor"
)

func standardSwiGLUBindings(spec Spec, layer *LayerWeights) []tensorBinding {
	width, feedForward := uint64(spec.EmbeddingLength), uint64(spec.FeedForwardLength)
	return []tensorBinding{
		requiredTensorPointer(feedForwardNormWeightTensor, &layer.FeedForwardNorm, width),
		requiredTensorPointer(feedForwardGateWeightTensor, &layer.FeedForwardGate, width, feedForward),
		requiredTensorPointer(feedForwardUpWeightTensor, &layer.FeedForwardUp, width, feedForward),
		requiredTensorPointer(feedForwardDownWeightTensor, &layer.FeedForwardDown, feedForward, width),
	}
}

func compileDenseFFNBindings(
	catalog weightCatalog,
	prefix string,
	spec Spec,
	layer *LayerWeights,
	block uint32,
) []tensorBinding {
	profile := spec.Profile()
	feedForwardLength := spec.LayerFeedForwardLength(block)
	if profile.DeciSparse && feedForwardLength == tensor.FirstOffset {
		return nil
	}
	if spec.expertCompositionPlan(profile).kind == expertDenseRoutedSeparateNorm {
		feedForwardLength = spec.EmbeddingLength
	}
	shapes := spec.TensorShapes(block)
	shapes.FeedForward = uint64(feedForwardLength)
	if profile.EncoderOperator.usesALiBiQKNorm() {
		upCopies := FeedForwardFusedGateUp.upProjectionCopies()
		var bindings []tensorBinding
		if _, separateGate := catalog.tensor(prefix + feedForwardGateWeightTensor); separateGate {
			upCopies = FeedForwardSwiGLU.upProjectionCopies()
			bindings = append(bindings, requiredTensorPointer(
				feedForwardGateWeightTensor, &layer.FeedForwardGate, shapes.FeedForwardUp(upCopies)...,
			))
		}
		return append(bindings,
			requiredTensorPointer(feedForwardUpWeightTensor, &layer.FeedForwardUp,
				shapes.FeedForwardUp(upCopies)...),
			optionalF32TensorPointer("ffn_up.bias", &layer.FeedForwardUpBias,
				shapes.FeedForwardUpWidth(upCopies)),
			requiredTensorPointer(feedForwardDownWeightTensor, &layer.FeedForwardDown,
				uint64(feedForwardLength), uint64(spec.EmbeddingLength)),
			requiredF32TensorPointer("ffn_down.bias", &layer.FeedForwardDownBias,
				uint64(spec.EmbeddingLength)),
			requiredF32TensorPointer("attn_output.bias", &layer.AttentionOutputBias,
				uint64(spec.EmbeddingLength)),
		)
	}
	var bindings []tensorBinding
	if profile.FeedForward.separateGate() {
		bindings = append(bindings, requiredTensorPointer(
			feedForwardGateWeightTensor, &layer.FeedForwardGate,
			shapes.FeedForwardUp(FeedForwardSwiGLU.upProjectionCopies())...,
		))
	}
	bindings = append(bindings,
		requiredTensorPointer(feedForwardUpWeightTensor, &layer.FeedForwardUp,
			shapes.FeedForwardUp(profile.FeedForward.upProjectionCopies())...),
		requiredTensorPointer(feedForwardDownWeightTensor, &layer.FeedForwardDown, shapes.FeedForwardDown()...),
	)
	requireGateBias := profile.DenseWeights.BiasCatalog == denseBiasCatalogJais
	requireUpBias := profile.FeedForward == FeedForwardSequentialGELU ||
		profile.DenseWeights.BiasCatalog == denseBiasCatalogGPTJ ||
		profile.DenseWeights.BiasCatalog == denseBiasCatalogJais2 || requireGateBias
	requireDownBias := requireUpBias
	requireAttentionBias := profile.FeedForward == FeedForwardSequentialGELU ||
		profile.DenseWeights.BiasCatalog == denseBiasCatalogJais2 || requireGateBias
	bindings = append(bindings,
		f32BiasBinding("ffn_gate.bias", &layer.FeedForwardGateBias, requireGateBias,
			uint64(feedForwardLength)),
		f32BiasBinding("ffn_up.bias", &layer.FeedForwardUpBias, requireUpBias,
			uint64(feedForwardLength)),
		f32BiasBinding("ffn_down.bias", &layer.FeedForwardDownBias, requireDownBias,
			shapes.EmbeddingVector()...),
	)
	if requireAttentionBias {
		bindings = append(bindings, requiredF32TensorPointer(
			"attn_output.bias", &layer.AttentionOutputBias, uint64(spec.EmbeddingLength),
		))
	}
	if profile.DenseWeights.BiasCatalog == denseBiasCatalogJais2 {
		bindings = append(bindings,
			requiredF32TensorPointer("attn_q.bias", &layer.AttentionQBias, layer.AttentionQ.MatrixRows()),
			requiredF32TensorPointer("attn_k.bias", &layer.AttentionKBias, layer.AttentionK.MatrixRows()),
			requiredF32TensorPointer("attn_v.bias", &layer.AttentionVBias, layer.AttentionV.MatrixRows()),
		)
	}
	if profile.Overrides == EmbeddingOverrideVisualSpan {
		bindings = append(bindings,
			requiredTensorPointer("vis_attn_qkv.weight", &layer.VisualAttentionQKV,
				uint64(spec.EmbeddingLength), tensor.TripleExtent*uint64(spec.EmbeddingLength)),
			requiredTensorPointer("vis_attn_output.weight", &layer.VisualAttentionOutput,
				uint64(spec.EmbeddingLength), uint64(spec.EmbeddingLength)),
			requiredTensorPointer("vis_gate.weight", &layer.VisualFeedForwardGate,
				uint64(spec.EmbeddingLength), uint64(spec.FeedForwardLength)),
			requiredTensorPointer("vis_up.weight", &layer.VisualFeedForwardUp,
				uint64(spec.EmbeddingLength), uint64(spec.FeedForwardLength)),
			requiredTensorPointer("vis_down.weight", &layer.VisualFeedForwardDown,
				uint64(spec.FeedForwardLength), uint64(spec.EmbeddingLength)),
		)
	}
	return bindings
}

func f32BiasBinding(name string, destination **gguf.TensorInfo, required bool, shape ...uint64) tensorBinding {
	if required {
		return requiredF32TensorPointer(name, destination, shape...)
	}
	return optionalF32TensorPointer(name, destination, shape...)
}
