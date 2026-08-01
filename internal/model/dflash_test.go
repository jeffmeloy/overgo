package model

import (
	"testing"

	"llamacpp2go/internal/tensor"
	"llamacpp2go/internal/tensor/dtype"
	"llamacpp2go/internal/tensor/reference"
)

func TestBuildDFlashFeatureInjectionAndNoiseBlock(t *testing.T) {
	builder := tensor.NewBuilder()
	feeds := make(map[*tensor.Tensor]reference.Value)
	input := func(name string, shape tensor.Shape, value float32) *tensor.Tensor {
		item := builder.Input(name, dtype.F32, shape)
		elements, err := shape.Elements()
		if err != nil {
			t.Fatal(err)
		}
		data := make([]float32, elements)
		for index := range data {
			data[index] = value + float32(index%5)*0.01
		}
		feeds[item] = reference.Value{Shape: shape, Data: data}
		return item
	}
	spec := Spec{
		Architecture: "dflash", EmbeddingLength: 4, FeedForwardLength: 6,
		HeadCount: 2, HeadCountKV: 1, KeyLength: 2, ValueLength: 2,
		RopeDimensionCount: 2, RopeFrequencyBase: 10000, RMSNormEpsilon: 1e-5,
		NonCausalAttention: true, TargetLayers: []int32{1, 3},
	}
	features := input("features", tensor.MustShape(8, 2), 0.1)
	projection := input("fc", tensor.MustShape(8, 4), 0.02)
	encoderNorm := input("enc_norm", tensor.MustShape(4), 0.9)
	fused, err := BuildDFlashFeatureEncoder(builder, features, projection, encoderNorm, spec)
	if err != nil {
		t.Fatal(err)
	}
	weights := LayerGraphWeights{
		AttentionNorm:   input("attn_norm", tensor.MustShape(4), 0.9),
		AttentionQ:      input("q", tensor.MustShape(4, 4), 0.02),
		AttentionK:      input("k", tensor.MustShape(4, 2), 0.02),
		AttentionV:      input("v", tensor.MustShape(4, 2), 0.02),
		AttentionOutput: input("o", tensor.MustShape(4, 4), 0.02),
		AttentionQNorm:  input("q_norm", tensor.MustShape(2), 0.9),
		AttentionKNorm:  input("k_norm", tensor.MustShape(2), 0.9),
		FeedForwardNorm: input("ffn_norm", tensor.MustShape(4), 0.9),
		FeedForwardGate: input("gate", tensor.MustShape(4, 6), 0.02),
		FeedForwardUp:   input("up", tensor.MustShape(4, 6), 0.02),
		FeedForwardDown: input("down", tensor.MustShape(6, 4), 0.02),
	}
	cacheKey, cacheValue, err := BuildDFlashCacheInjection(builder, fused, spec, weights, []uint32{0, 1}, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	noise := input("noise", tensor.MustShape(4, 3), 0.1)
	result, err := BuildDenseBlockCachedForLayer(
		builder, noise, spec, weights, []uint32{2, 3, 4}, cacheKey, cacheValue, 0,
	)
	if err != nil {
		t.Fatal(err)
	}
	results, err := reference.Execute([]*tensor.Tensor{result.Output, result.Key, result.Value}, feeds)
	if err != nil {
		t.Fatal(err)
	}
	if !results[result.Output].Shape.Equal(tensor.MustShape(4, 3)) ||
		!results[result.Key].Shape.Equal(tensor.MustShape(2, 1, 5)) ||
		!results[result.Value].Shape.Equal(tensor.MustShape(2, 1, 5)) {
		t.Fatalf("unexpected DFlash shapes: output=%v key=%v value=%v",
			results[result.Output].Shape.Slice(), results[result.Key].Shape.Slice(), results[result.Value].Shape.Slice())
	}
	nodes, err := tensor.Topological(result.Output)
	if err != nil {
		t.Fatal(err)
	}
	for _, node := range nodes {
		if node.Op == tensor.OpAttention && node.Attrs.(tensor.AttentionAttributes).Causal {
			t.Fatal("DFlash noise block used causal attention")
		}
	}
}
