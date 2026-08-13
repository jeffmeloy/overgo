package model

import (
	"errors"
	"fmt"

	"overgo/internal/gguf"
	"overgo/internal/tensor/dtype"
)

func loadDenseFFNCatalog(
	required weightRequirementLoader,
	tensors map[string]gguf.TensorInfo,
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
	if spec.expertCompositionPlan().kind == expertArctic {
		feedForwardLength = spec.EmbeddingLength
	}
	shapes := spec.TensorShapes(block)
	shapes.FeedForward = uint64(feedForwardLength)
	if profile.EncoderOperator.usesALiBiQKNorm() {
		if gate, ok := tensors[prefix+"ffn_gate.weight"]; ok {
			if gate.Dimensions != 2 || gate.Shape[0] != uint64(spec.EmbeddingLength) ||
				gate.Shape[1] != uint64(feedForwardLength) {
				return fmt.Errorf("tensor %q has incompatible shape %v", gate.Name, gate.Shape)
			}
			layer.FeedForwardGate = gate
		}
		up, ok := tensors[prefix+"ffn_up.weight"]
		if !ok {
			return fmt.Errorf("required tensor %q is missing", prefix+"ffn_up.weight")
		}
		upWidth := uint64(feedForwardLength)
		if up.Dimensions != 2 || up.Shape[0] != uint64(spec.EmbeddingLength) ||
			(up.Shape[1] != upWidth && up.Shape[1] != 2*upWidth) {
			return fmt.Errorf("tensor %q has incompatible shape %v", up.Name, up.Shape)
		}
		if layer.FeedForwardGate.Name != "" && up.Shape[1] != upWidth {
			return errors.New("JinaBERT v2 separate and fused FFN gates cannot be combined")
		}
		layer.FeedForwardUp = up
		if _, ok := tensors[prefix+"ffn_up.bias"]; ok {
			bias, err := required(prefix+"ffn_up.bias", up.Shape[1])
			if err != nil {
				return err
			}
			if bias.Type != dtype.F32 {
				return fmt.Errorf("tensor %q must use F32 bias storage", bias.Name)
			}
			layer.FeedForwardUpBias = &bias
		}
		down, err := required(prefix+"ffn_down.weight", upWidth, uint64(spec.EmbeddingLength))
		if err != nil {
			return err
		}
		layer.FeedForwardDown = down
		downBias, err := required(prefix+"ffn_down.bias", uint64(spec.EmbeddingLength))
		if err != nil {
			return err
		}
		if downBias.Type != dtype.F32 {
			return fmt.Errorf("tensor %q must use F32 bias storage", downBias.Name)
		}
		layer.FeedForwardDownBias = &downBias
		if layer.AttentionOutputBias == nil {
			return fmt.Errorf("required tensor %q is missing", prefix+"attn_output.bias")
		}
		return nil
	}
	if profile.FeedForward == FeedForwardSwiGLU {
		gate, err := required(prefix+"ffn_gate.weight", shapes.FeedForwardUp(1)...)
		if err != nil {
			return err
		}
		layer.FeedForwardGate = gate
	}
	upMultiplier := uint64(1)
	if profile.FeedForward == FeedForwardFusedGateUp {
		upMultiplier = 2
	}
	up, err := required(prefix+"ffn_up.weight", shapes.FeedForwardUp(upMultiplier)...)
	if err != nil {
		return err
	}
	layer.FeedForwardUp = up
	down, err := required(prefix+"ffn_down.weight", shapes.FeedForwardDown()...)
	if err != nil {
		return err
	}
	layer.FeedForwardDown = down
	if err := loadTensorRequirements(required, tensors, prefix, []tensorRequirement{
		optionalF32TensorPointer("ffn_gate.bias", &layer.FeedForwardGateBias, uint64(feedForwardLength)),
		optionalF32TensorPointer("ffn_up.bias", &layer.FeedForwardUpBias, uint64(feedForwardLength)),
		optionalF32TensorPointer("ffn_down.bias", &layer.FeedForwardDownBias, shapes.EmbeddingVector()...),
	}); err != nil {
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
		return loadTensorRequirements(required, tensors, prefix, []tensorRequirement{
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
