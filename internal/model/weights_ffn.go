package model

import (
	"errors"
	"fmt"

	"overgo/internal/gguf"
)

func loadStandardSwiGLUCatalog(
	catalog weightCatalog,
	prefix string,
	spec Spec,
	layer *LayerWeights,
) error {
	width, feedForward := uint64(spec.EmbeddingLength), uint64(spec.FeedForwardLength)
	return bindTensorProgram(catalog, prefix, []tensorBinding{
		requiredTensorPointer(feedForwardNormWeightTensor, &layer.FeedForwardNorm, width),
		requiredTensorPointer(feedForwardGateWeightTensor, &layer.FeedForwardGate, width, feedForward),
		requiredTensorPointer(feedForwardUpWeightTensor, &layer.FeedForwardUp, width, feedForward),
		requiredTensorPointer(feedForwardDownWeightTensor, &layer.FeedForwardDown, feedForward, width),
	})
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
	if profile.DeciSparse && feedForwardLength == 0 {
		return nil
	}
	if spec.expertCompositionPlan(profile).kind == expertDenseRoutedSeparateNorm {
		feedForwardLength = spec.EmbeddingLength
	}
	shapes := spec.TensorShapes(block)
	shapes.FeedForward = uint64(feedForwardLength)
	if profile.EncoderOperator.usesALiBiQKNorm() {
		upWidth := uint64(feedForwardLength)
		if err := bindTensorProgram(catalog, prefix, []tensorBinding{
			optionalTensorPointer(feedForwardGateWeightTensor, &layer.FeedForwardGate, uint64(spec.EmbeddingLength), upWidth),
			requiredTensorPointerShapes(feedForwardUpWeightTensor, &layer.FeedForwardUp,
				[]uint64{uint64(spec.EmbeddingLength), upWidth},
				[]uint64{uint64(spec.EmbeddingLength), 2 * upWidth}),
		}); err != nil {
			return err
		}
		if layer.FeedForwardGate != nil && layer.FeedForwardUp.Shape[1] != upWidth {
			return errors.New("JinaBERT v2 separate and fused FFN gates cannot be combined")
		}
		if err := bindTensorProgram(catalog, prefix, []tensorBinding{
			optionalF32TensorPointer("ffn_up.bias", &layer.FeedForwardUpBias, layer.FeedForwardUp.Shape[1]),
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
	program := make([]tensorBinding, 0, 6)
	if profile.FeedForward == FeedForwardSwiGLU || profile.FeedForward == FeedForwardGEGLU {
		program = append(program, requiredTensorPointer(
			feedForwardGateWeightTensor, &layer.FeedForwardGate, shapes.FeedForwardUp(1)...,
		))
	}
	upMultiplier := uint64(1)
	if profile.FeedForward == FeedForwardFusedGateUp {
		upMultiplier = 2
	}
	program = append(program,
		requiredTensorPointer(feedForwardUpWeightTensor, &layer.FeedForwardUp, shapes.FeedForwardUp(upMultiplier)...),
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
			requiredTensorPointer("vis_attn_qkv.weight", &layer.VisualAttentionQKV, uint64(spec.EmbeddingLength), 3*uint64(spec.EmbeddingLength)),
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
