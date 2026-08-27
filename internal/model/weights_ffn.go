package model

import (
	"fmt"

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

func loadDenseFFNCatalog(
	catalog weightCatalog,
	prefix string,
	spec Spec,
	layer *LayerWeights,
	block uint32,
) error {
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
		upWidth := uint64(feedForwardLength)
		upCopies := FeedForwardFusedGateUp.upProjectionCopies()
		var program []tensorBinding
		if _, separateGate := catalog.tensor(prefix + feedForwardGateWeightTensor); separateGate {
			upCopies = FeedForwardSwiGLU.upProjectionCopies()
			program = append(program, requiredTensorPointer(
				feedForwardGateWeightTensor, &layer.FeedForwardGate, shapes.FeedForwardUp(upCopies)...,
			))
		}
		program = append(program, requiredTensorPointer(
			feedForwardUpWeightTensor, &layer.FeedForwardUp, shapes.FeedForwardUp(upCopies)...,
		))
		if err := bindTensorProgram(catalog, prefix, program); err != nil {
			return err
		}
		if err := bindTensorProgram(catalog, prefix, []tensorBinding{
			optionalF32TensorPointer("ffn_up.bias", &layer.FeedForwardUpBias, shapes.FeedForwardUpWidth(upCopies)),
			requiredTensorPointer(feedForwardDownWeightTensor, &layer.FeedForwardDown, upWidth, uint64(spec.EmbeddingLength)),
			requiredF32TensorPointer("ffn_down.bias", &layer.FeedForwardDownBias, uint64(spec.EmbeddingLength)),
		}); err != nil {
			return err
		}
		if layer.AttentionOutputBias == nil {
			return fmt.Errorf("required tensor %q is missing", prefix+"attn_output.bias")
		}
		return nil
	}
	var program []tensorBinding
	if profile.FeedForward.separateGate() {
		program = append(program, requiredTensorPointer(
			feedForwardGateWeightTensor, &layer.FeedForwardGate,
			shapes.FeedForwardUp(FeedForwardSwiGLU.upProjectionCopies())...,
		))
	}
	program = append(program,
		requiredTensorPointer(feedForwardUpWeightTensor, &layer.FeedForwardUp,
			shapes.FeedForwardUp(profile.FeedForward.upProjectionCopies())...),
		requiredTensorPointer(feedForwardDownWeightTensor, &layer.FeedForwardDown, shapes.FeedForwardDown()...),
		optionalF32TensorPointer("ffn_gate.bias", &layer.FeedForwardGateBias, uint64(feedForwardLength)),
		optionalF32TensorPointer("ffn_up.bias", &layer.FeedForwardUpBias, uint64(feedForwardLength)),
		optionalF32TensorPointer("ffn_down.bias", &layer.FeedForwardDownBias, shapes.EmbeddingVector()...),
	)
	if err := bindTensorProgram(catalog, prefix, program); err != nil {
		return err
	}
	if profile.FeedForward == FeedForwardSequentialGELU {
		if err := requireCatalogBindings(prefix, []catalogBinding{
			{name: "attn_output.bias", item: layer.AttentionOutputBias},
			{name: "ffn_up.bias", item: layer.FeedForwardUpBias},
			{name: "ffn_down.bias", item: layer.FeedForwardDownBias},
		}); err != nil {
			return err
		}
	}
	switch profile.DenseWeights.BiasCatalog {
	case denseBiasCatalogGPTJ:
		if err := requireCatalogBindings(prefix, []catalogBinding{
			{name: "ffn_up.bias", item: layer.FeedForwardUpBias},
			{name: "ffn_down.bias", item: layer.FeedForwardDownBias},
		}); err != nil {
			return err
		}
	case denseBiasCatalogJais2:
		if err := requireCatalogBindings(prefix, []catalogBinding{
			{name: "attn_q.bias", item: layer.AttentionQBias},
			{name: "attn_k.bias", item: layer.AttentionKBias},
			{name: "attn_v.bias", item: layer.AttentionVBias},
			{name: "attn_output.bias", item: layer.AttentionOutputBias},
			{name: "ffn_up.bias", item: layer.FeedForwardUpBias},
			{name: "ffn_down.bias", item: layer.FeedForwardDownBias},
		}); err != nil {
			return err
		}
	case denseBiasCatalogJais:
		if err := requireCatalogBindings(prefix, []catalogBinding{
			{name: "attn_output.bias", item: layer.AttentionOutputBias},
			{name: "ffn_gate.bias", item: layer.FeedForwardGateBias},
			{name: "ffn_up.bias", item: layer.FeedForwardUpBias},
			{name: "ffn_down.bias", item: layer.FeedForwardDownBias},
		}); err != nil {
			return err
		}
	}
	if profile.Overrides == EmbeddingOverrideVisualSpan {
		return bindTensorProgram(catalog, prefix, []tensorBinding{
			requiredTensorPointer("vis_attn_qkv.weight", &layer.VisualAttentionQKV,
				uint64(spec.EmbeddingLength), tensor.TripleExtent*uint64(spec.EmbeddingLength)),
			requiredTensorPointer("vis_attn_output.weight", &layer.VisualAttentionOutput, uint64(spec.EmbeddingLength), uint64(spec.EmbeddingLength)),
			requiredTensorPointer("vis_gate.weight", &layer.VisualFeedForwardGate, uint64(spec.EmbeddingLength), uint64(spec.FeedForwardLength)),
			requiredTensorPointer("vis_up.weight", &layer.VisualFeedForwardUp, uint64(spec.EmbeddingLength), uint64(spec.FeedForwardLength)),
			requiredTensorPointer("vis_down.weight", &layer.VisualFeedForwardDown, uint64(spec.FeedForwardLength), uint64(spec.EmbeddingLength)),
		})
	}
	return nil
}

type catalogBinding struct {
	name string
	item *gguf.TensorInfo
}

func requireCatalogBindings(prefix string, bindings []catalogBinding) error {
	for _, binding := range bindings {
		if binding.item == nil {
			return fmt.Errorf("required tensor %q is missing", prefix+binding.name)
		}
	}
	return nil
}
