package model

import (
	"errors"
	"fmt"
	"reflect"

	"overgo/internal/gguf"
	"overgo/internal/tensor"
)

// hyperDraftArchitecture names a draft file that attaches a multi-token-prediction layer to a hyper-connection trunk: Qwen3.8-Flash-Next's mtp.* block, in its checkpoint's tensor names.
const hyperDraftArchitecture = "qwen4exp-mtp"

// HyperDraftWeights is the draft layer a draft file attaches to a hyper-connection trunk.
//   - Layer has the shape of one of the trunk's attention layers: gated attention, both hyper-connection mixes, and routed experts in one gate-and-up bank
//   - the layer's input fuses the next token's embedding (EmbeddingNorm, EmbeddingProjection) with the trunk's residual streams (HiddenNorm over all streams at once, then HiddenProjection on each)
//   - Mixer mixes the layer's streams for the trunk's head
//   - each one-dimensional norm is stored as the checkpoint holds it: its scale is one plus the stored weight, an offset the trunk's file folds in.
type HyperDraftWeights struct {
	Layer                                 LayerWeights
	EmbeddingNorm, HiddenNorm             gguf.TensorInfo
	EmbeddingProjection, HiddenProjection gguf.TensorInfo
	MixerNorm, MixerDown, MixerUp         gguf.TensorInfo
}

// BindHyperDraftWeights binds a draft file's tensors to the trunk the spec describes, checking each shape against the trunk's.
func BindHyperDraftWeights(file *gguf.File, spec Spec) (HyperDraftWeights, error) {
	if value, ok := file.MetadataValue("general.architecture"); !ok || value.Data != hyperDraftArchitecture {
		return HyperDraftWeights{}, fmt.Errorf("draft file is not a %s file", hyperDraftArchitecture)
	}
	if spec.HyperConnectionCount < tensor.PairedExtent || spec.HyperConnectionRank == 0 || spec.ExpertCount == 0 {
		return HyperDraftWeights{}, fmt.Errorf("a %s draft needs a hyper-connection trunk with routed experts", hyperDraftArchitecture)
	}
	catalog, err := newWeightCatalog(file)
	if err != nil {
		return HyperDraftWeights{}, err
	}
	width, head := uint64(spec.EmbeddingLength), uint64(spec.KeyLength)
	streams, rank := uint64(spec.HyperConnectionCount), uint64(spec.HyperConnectionRank)
	wide := width * streams
	heads, keyHeads := uint64(spec.HeadCount), uint64(spec.HeadCountKV)
	experts, expert, shared := uint64(spec.ExpertCount), uint64(spec.ExpertFeedForward), uint64(spec.SharedExpertFF)
	var draft HyperDraftWeights
	layer := &draft.Layer
	if err := bindTensorProgram(catalog, "mtp.", []tensorBinding{
		requiredTensor("pre_fc_norm_embedding.weight", &draft.EmbeddingNorm, width),
		requiredTensor("pre_fc_norm_hidden.weight", &draft.HiddenNorm, wide),
		requiredTensor("fc_embedding.weight", &draft.EmbeddingProjection, width, width),
		requiredTensor("fc_hidden.weight", &draft.HiddenProjection, width, width),
		requiredTensor("hyper_connection_mixer.hc_norm.weight", &draft.MixerNorm, wide),
		requiredTensor("hyper_connection_mixer.input_mix_weight_down.weight", &draft.MixerDown, wide, rank),
		requiredTensor("hyper_connection_mixer.input_mix_weight_up.weight", &draft.MixerUp, rank, wide),
	}); err != nil {
		return HyperDraftWeights{}, err
	}
	if err := bindTensorProgram(catalog, "mtp.layers.0.", []tensorBinding{
		requiredTensorPointer("self_attn.q_proj.weight", &layer.AttentionQ, width, tensor.PairedExtent*heads*head),
		requiredTensorPointer("self_attn.k_proj.weight", &layer.AttentionK, width, keyHeads*head),
		requiredTensorPointer("self_attn.v_proj.weight", &layer.AttentionV, width, keyHeads*uint64(spec.ValueLength)),
		requiredTensorPointer("self_attn.o_proj.weight", &layer.AttentionOutput, heads*uint64(spec.ValueLength), width),
		requiredTensorPointer("self_attn.q_norm.weight", &layer.AttentionQNorm, head),
		requiredTensorPointer("self_attn.k_norm.weight", &layer.AttentionKNorm, head),
		requiredTensorPointer("attn_hyper_connection.hc_norm.weight", &layer.HyperAttentionNorm, wide),
		requiredTensorPointer("attn_hyper_connection.input_mix_weight_down.weight", &layer.HyperAttentionDown, wide, rank),
		requiredTensorPointer("attn_hyper_connection.input_mix_weight_up.weight", &layer.HyperAttentionUp, rank, wide),
		requiredTensorPointer("attn_hyper_connection.block_inject_weight.weight", &layer.HyperAttentionInject, wide, streams),
		requiredTensorPointer("mlp_hyper_connection.hc_norm.weight", &layer.HyperFeedForwardNorm, wide),
		requiredTensorPointer("mlp_hyper_connection.input_mix_weight_down.weight", &layer.HyperFeedForwardDown, wide, rank),
		requiredTensorPointer("mlp_hyper_connection.input_mix_weight_up.weight", &layer.HyperFeedForwardUp, rank, wide),
		requiredTensorPointer("mlp_hyper_connection.block_inject_weight.weight", &layer.HyperFeedForwardInject, wide, streams),
		requiredTensorPointer("mlp.gate.weight", &layer.FeedForwardRouter, width, experts),
		requiredTensorPointer("mlp.experts.gate_up_proj", &layer.FeedForwardGateUpExperts, width, tensor.PairedExtent*expert, experts),
		requiredTensorPointer("mlp.experts.down_proj", &layer.FeedForwardDownExperts, expert, width, experts),
		requiredTensorPointerShapes("mlp.shared_expert_gate.weight", &layer.FeedForwardSharedRouter, []uint64{width}, []uint64{width, tensor.SingletonExtent}),
		requiredTensorPointer("mlp.shared_expert.gate_proj.weight", &layer.FeedForwardSharedGate, width, shared),
		requiredTensorPointer("mlp.shared_expert.up_proj.weight", &layer.FeedForwardSharedUp, width, shared),
		requiredTensorPointer("mlp.shared_expert.down_proj.weight", &layer.FeedForwardSharedDown, shared, width),
	}); err != nil {
		return HyperDraftWeights{}, err
	}
	return draft, nil
}

// Tensors lists every tensor the draft layer reads.
func (w *HyperDraftWeights) Tensors() []gguf.TensorInfo {
	var result []gguf.TensorInfo
	collectTensorInfos(reflect.ValueOf(*w), make(map[string]struct{}), &result)
	return result
}

// HyperDraftProgram is the program of an attached hyper-connection draft layer: the trunk's last attention layer at index BlockCount, attending densely over the draft's own keys.
//   - its head mix is the trunk's last layer's alone, so the layer leaves its streams unmixed.
func (p ModelPlan) HyperDraftProgram() (CompiledLayerProgram, error) {
	if p.spec.HyperConnectionCount < tensor.PairedExtent {
		return CompiledLayerProgram{}, errors.New("model plan has no hyper-connection trunk to draft for")
	}
	for index := len(p.layers) - 1; index >= 0; index-- {
		plan := p.layers[index]
		if plan.Recurrent {
			continue
		}
		plan.Layer, plan.AttentionGraph.BlockSize = p.spec.BlockCount, 0
		return CompiledLayerProgram{spec: p.spec, plan: plan}, nil
	}
	return CompiledLayerProgram{}, errors.New("model plan has no attention layer to draft with")
}

// BuildHyperDraftInput fuses the next token's embedding, [embedding, 1], with the trunk's residual streams, [embedding*streams, 1], into the draft layer's streams, [embedding, streams, 1].
func BuildHyperDraftInput(
	builder *tensor.Builder,
	spec Spec,
	embedding, streams, embeddingNorm, hiddenNorm, embeddingProjection, hiddenProjection *tensor.Tensor,
) (*tensor.Tensor, error) {
	if err := (graphWeights{embedding, streams, embeddingNorm, hiddenNorm, embeddingProjection, hiddenProjection}).validate("hyper-connection draft input"); err != nil {
		return nil, err
	}
	width, count := uint64(spec.EmbeddingLength), uint64(spec.HyperConnectionCount)
	fused := builder.MulMat(embeddingProjection, builder.WeightedRMSNorm(embedding, embeddingNorm, spec.RMSNormEpsilon))
	normalized := builder.WeightedRMSNorm(builder.Reshape(streams, width*count, tensor.SingletonExtent), hiddenNorm, spec.RMSNormEpsilon)
	hidden := builder.MulMat(hiddenProjection, builder.Reshape(normalized, width, count))
	output := builder.Add(builder.Reshape(hidden, width, count, tensor.SingletonExtent), repeatHyperStreams(builder, fused, spec.HyperConnectionCount))
	return output, builder.Err()
}

// BuildHyperDraftHeadInput mixes the draft layer's streams, [embedding, streams, 1], into the trunk head's input, [embedding, 1].
func BuildHyperDraftHeadInput(builder *tensor.Builder, spec Spec, streams, norm, down, up *tensor.Tensor) (*tensor.Tensor, error) {
	mixed, _, err := buildGatedHyperMix(builder, streams, norm, down, up, nil, spec)
	return mixed, err
}
