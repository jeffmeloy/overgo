package model

import (
	"testing"

	"llamacpp2go/internal/tensor"
	"llamacpp2go/internal/tensor/dtype"
	"llamacpp2go/internal/tensor/reference"
)

func TestBuildEagle3FeatureAndDecoder(t *testing.T) {
	builder := tensor.NewBuilder()
	feeds := make(map[*tensor.Tensor]reference.Value)
	input := func(name string, shape tensor.Shape, fill float32) *tensor.Tensor {
		item := builder.Input(name, dtype.F32, shape)
		elements, err := shape.Elements()
		if err != nil {
			t.Fatal(err)
		}
		data := make([]float32, elements)
		for index := range data {
			data[index] = fill + float32(index%7)*0.01
		}
		feeds[item] = reference.Value{Shape: shape, Data: data}
		return item
	}
	spec := Spec{
		Architecture: "eagle3", EmbeddingLength: 4, TargetHiddenSize: 3,
		TargetLayers: []int32{1, 3, 5}, FeedForwardLength: 6,
		HeadCount: 2, HeadCountKV: 1, KeyLength: 2, ValueLength: 2,
		RopeDimensionCount: 2, RopeFrequencyBase: 10000, RMSNormEpsilon: 1e-5,
	}
	features := input("features", tensor.MustShape(9, 3), 0.1)
	projection := input("fc", tensor.MustShape(9, 4), 0.02)
	fused, err := BuildEagle3FeatureEncoder(builder, features, projection, spec)
	if err != nil {
		t.Fatal(err)
	}
	weights := LayerGraphWeights{
		AttentionNorm:   input("token_norm", tensor.MustShape(4), 0.9),
		AttentionNorm2:  input("target_norm", tensor.MustShape(4), 0.9),
		AttentionQ:      input("q", tensor.MustShape(8, 4), 0.02),
		AttentionK:      input("k", tensor.MustShape(8, 2), 0.02),
		AttentionV:      input("v", tensor.MustShape(8, 2), 0.02),
		AttentionOutput: input("o", tensor.MustShape(4, 4), 0.02),
		FeedForwardNorm: input("ffn_norm", tensor.MustShape(4), 0.9),
		FeedForwardGate: input("gate", tensor.MustShape(4, 6), 0.02),
		FeedForwardUp:   input("up", tensor.MustShape(4, 6), 0.02),
		FeedForwardDown: input("down", tensor.MustShape(6, 4), 0.02),
	}
	tokens := input("tokens", tensor.MustShape(4, 3), 0.1)
	result, err := BuildEagle3BlockCached(builder, tokens, fused, spec, weights, []uint32{0, 1, 2}, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	results, err := reference.Execute([]*tensor.Tensor{result.Output, result.Key, result.Value}, feeds)
	if err != nil {
		t.Fatal(err)
	}
	if !results[result.Output].Shape.Equal(tensor.MustShape(4, 3)) ||
		!results[result.Key].Shape.Equal(tensor.MustShape(2, 1, 3)) {
		t.Fatalf("unexpected Eagle3 shapes: output=%v key=%v", results[result.Output].Shape.Slice(), results[result.Key].Shape.Slice())
	}
}
