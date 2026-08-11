package model

import (
	"testing"

	"overgo/internal/tensor"
	"overgo/internal/tensor/dtype"
	"overgo/internal/tensor/reference"
)

func TestBuildGemma4AssistantPipeline(t *testing.T) {
	builder := tensor.NewBuilder()
	spec := Spec{CommonSpec: CommonSpec{Architecture: "gemma4-assistant", BlockCount: 1, EmbeddingLength: 2,
		TargetHiddenSize: 3, FeedForwardLength: 4,

		RMSNormEpsilon: 1e-6,
		VocabularySize: 4}, AttentionSpec: AttentionSpec{HeadCount: 1, HeadCountKV: 1,
		KeyLength: 2, ValueLength: 2, KeyLengthSWA: 2, ValueLengthSWA: 2,
		RopeDimensionCount: 2, RopeDimensionSWA: 2,
		RopeFrequencyBase: 10000, RopeFrequencySWA: 10000,
		SlidingWindow: 4, SlidingLayers: []bool{false}},
	}
	input := func(name string, shape ...uint64) *tensor.Tensor {
		return builder.Input(name, dtype.F32, tensor.MustShape(shape...))
	}
	token := input("target_token", 3, 1)
	hidden := input("target_hidden", 3, 1)
	pre := input("pre", 6, 2)
	current, err := BuildGemma4AssistantInput(builder, token, hidden, pre, spec)
	if err != nil {
		t.Fatal(err)
	}
	weights := LayerGraphWeights{
		AttentionNorm:       input("attn_norm", 2),
		AttentionQ:          input("attn_q", 2, 2),
		AttentionQNorm:      input("attn_q_norm", 2),
		AttentionOutput:     input("attn_output", 2, 2),
		AttentionPostNorm:   input("post_attention_norm", 2),
		FeedForwardNorm:     input("ffn_norm", 2),
		FeedForwardGate:     input("ffn_gate", 2, 4),
		FeedForwardUp:       input("ffn_up", 2, 4),
		FeedForwardDown:     input("ffn_down", 4, 2),
		FeedForwardPostNorm: input("post_ffw_norm", 2),
		LayerOutputScale:    input("layer_output_scale", 1),
	}
	sharedKey := input("shared_key", 2, 1, 2)
	sharedValue := input("shared_value", 2, 1, 2)
	plan := fixtureLayerPlan(t, spec, Weights{}, 0)
	block, err := buildFixtureLayerWithPlan(
		builder, current, spec, weights, []uint32{2}, sharedKey, sharedValue, plan,
	)
	if err != nil {
		t.Fatal(err)
	}
	current = block.Output
	outputNorm := input("output_norm", 2)
	output := input("output", 2, 4)
	post := input("post", 2, 3)
	logits, nextHidden, err := BuildGemma4AssistantOutputs(builder, current, outputNorm, output, post, spec)
	if err != nil {
		t.Fatal(err)
	}
	feeds := make(map[*tensor.Tensor]reference.Value)
	for _, node := range []*tensor.Tensor{
		token, hidden, pre, weights.AttentionNorm, weights.AttentionQ, weights.AttentionQNorm,
		weights.AttentionOutput, weights.AttentionPostNorm, weights.FeedForwardNorm,
		weights.FeedForwardGate, weights.FeedForwardUp, weights.FeedForwardDown,
		weights.FeedForwardPostNorm, sharedKey, sharedValue, outputNorm, output, post,
	} {
		elements, elementErr := node.Shape.Elements()
		if elementErr != nil {
			t.Fatal(elementErr)
		}
		feeds[node] = reference.Value{Shape: node.Shape, Data: make([]float32, elements)}
	}
	feeds[weights.LayerOutputScale] = reference.Value{Shape: weights.LayerOutputScale.Shape, Data: []float32{1}}
	for _, norm := range []*tensor.Tensor{
		weights.AttentionNorm, weights.AttentionQNorm, weights.AttentionPostNorm,
		weights.FeedForwardNorm, weights.FeedForwardPostNorm, outputNorm,
	} {
		value := feeds[norm]
		for index := range value.Data {
			value.Data[index] = 1
		}
		feeds[norm] = value
	}
	results, err := reference.Execute([]*tensor.Tensor{logits, nextHidden}, feeds)
	if err != nil {
		t.Fatal(err)
	}
	if !results[logits].Shape.Equal(tensor.MustShape(4, 1)) ||
		!results[nextHidden].Shape.Equal(tensor.MustShape(3, 1)) {
		t.Fatalf("Gemma 4 assistant outputs = %v/%v", results[logits].Shape.Slice(), results[nextHidden].Shape.Slice())
	}
}
