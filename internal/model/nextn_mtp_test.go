package model

import (
	"testing"

	"overgo/internal/tensor"
	"overgo/internal/tensor/dtype"
)

func TestBuildGLM4NextNMTPPipeline(t *testing.T) {
	builder := tensor.NewBuilder()
	spec := Spec{CommonSpec: CommonSpec{Architecture: "glm4", BlockCount: 1,
		NextNPredictLayers: 1, EmbeddingLength: 8, FeedForwardLength: 12,
		RMSNormEpsilon: 1e-5}, AttentionSpec: AttentionSpec{HeadCount: 2,
		HeadCountKV: 1, KeyLength: 4, ValueLength: 4, RopeDimensionCount: 4,
		RopeFrequencyBase: 10000}}
	token := builder.Input("token", dtype.F32, tensor.MustShape(8, 2))
	hidden := builder.Input("hidden", dtype.F32, token.Shape)
	norm := builder.Input("norm", dtype.F32, tensor.MustShape(8))
	projection := builder.Input("projection", dtype.F32, tensor.MustShape(16, 8))
	current, err := BuildNextNMTPInput(builder, token, hidden, norm, norm, projection, spec, 0)
	if err != nil {
		t.Fatal(err)
	}
	weights := denseBlockInputs(builder, spec)
	weights.AttentionQ = nil
	weights.AttentionK = nil
	weights.AttentionV = nil
	weights.AttentionQKV = builder.Input("qkv", dtype.F32, tensor.MustShape(8, 16))
	weights.AttentionPostNorm = builder.Input("attn_post", dtype.F32, tensor.MustShape(8))
	weights.FeedForwardPostNorm = builder.Input("ffn_post", dtype.F32, tensor.MustShape(8))
	weights.FeedForwardGate = nil
	weights.FeedForwardUp = builder.Input("gate_up", dtype.F32, tensor.MustShape(8, 24))
	draft := fixtureDraftProgram(t, spec, Weights{}, 0)
	block, err := buildFixtureLayerWithPlan(
		builder, current, draft.Spec, weights, []uint32{0, 1}, nil, nil, draft.Plan,
	)
	if err != nil {
		t.Fatal(err)
	}
	head := builder.Input("head", dtype.F32, tensor.MustShape(8, 32))
	logits, nextHidden, err := BuildNextNMTPOutputs(builder, block.Output, norm, head, spec, 0)
	if err != nil {
		t.Fatal(err)
	}
	if !logits.Shape.Equal(tensor.MustShape(32, 2)) || !nextHidden.Shape.Equal(token.Shape) ||
		!block.Key.Shape.Equal(tensor.MustShape(4, 1, 2)) {
		t.Fatalf("unexpected GLM4 NextN outputs: logits=%v hidden=%v key=%v", logits.Shape, nextHidden.Shape, block.Key.Shape)
	}
}

func TestBuildEXAONE4NextNMTPBlock(t *testing.T) {
	builder := tensor.NewBuilder()
	spec := Spec{CommonSpec: CommonSpec{Architecture: "exaone4", BlockCount: 3,
		NextNPredictLayers: 1, EmbeddingLength: 8, FeedForwardLength: 12,
		RMSNormEpsilon: 1e-5}, AttentionSpec: AttentionSpec{HeadCount: 2,
		HeadCountKV: 1, KeyLength: 4, ValueLength: 4, RopeDimensionCount: 4,
		RopeFrequencyBase: 1_000_000, SlidingWindow: 4096, SlidingPattern: 4,
		NoRopeLayerStep: 4}}
	input := builder.Input("input", dtype.F32, tensor.MustShape(8, 2))
	weights := denseBlockInputs(builder, spec)
	weights.AttentionNorm, weights.FeedForwardNorm = nil, nil
	weights.AttentionQ, weights.AttentionK, weights.AttentionV = nil, nil, nil
	weights.AttentionQKV = builder.Input("qkv", dtype.F32, tensor.MustShape(8, 16))
	weights.AttentionQKVBias = builder.Input("qkv_bias", dtype.F32, tensor.MustShape(16))
	weights.AttentionPostNorm = builder.Input("attn_post", dtype.F32, tensor.MustShape(8))
	weights.FeedForwardPostNorm = builder.Input("ffn_post", dtype.F32, tensor.MustShape(8))
	draft := fixtureDraftProgram(t, spec, Weights{}, 0)
	result, err := buildFixtureLayerWithPlan(
		builder, input, draft.Spec, weights, []uint32{0, 1}, nil, nil, draft.Plan,
	)
	if err != nil {
		t.Fatal(err)
	}
	if !result.Output.Shape.Equal(input.Shape) || !result.Key.Shape.Equal(tensor.MustShape(4, 1, 2)) {
		t.Fatalf("unexpected EXAONE 4 NextN block: %+v", result)
	}
}
