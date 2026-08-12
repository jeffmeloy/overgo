package model

import (
	"testing"

	"overgo/internal/tensor"
	"overgo/internal/tensor/dtype"
	"overgo/internal/tensor/reference"
)

func TestBuildWavTokenizerDecoder(t *testing.T) {
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
			data[index] = fill
		}
		feeds[item] = reference.Value{Shape: shape, Data: data}
		return item
	}
	spec := Spec{CommonSpec: CommonSpec{Architecture: "wavtokenizer-dec", EmbeddingLength: 2, OutputEmbeddingLength: 3,

		FeedForwardLength: 4,
		LayerNormEpsilon:  1e-5}, MultimodalSpec: MultimodalSpec{PosNetEmbeddingLength: 2, PosNetBlockCount: 6,
		ConvNextEmbeddingLength: 2, ConvNextBlockCount: 1,
		GroupNormGroups: 1, GroupNormEpsilon: 1e-5},
	}
	weights := WavTokenizerGraphWeights{
		InputConv:      input("conv1d.weight", tensor.MustShape(7, 2, 2), 0),
		InputConvBias:  input("conv1d.bias", tensor.MustShape(1, 2), 0),
		PosNet:         make([]WavPosNetGraphWeights, 6),
		TokenNorm:      input("token_norm", tensor.MustShape(2), 1),
		TokenNormBias:  input("token_norm_bias", tensor.MustShape(2), 0),
		ConvNext:       make([]WavConvNextGraphWeights, 1),
		OutputNorm:     input("output_norm", tensor.MustShape(2), 1),
		OutputNormBias: input("output_norm_bias", tensor.MustShape(2), 0),
		Output:         input("output", tensor.MustShape(2, 3), 0),
		OutputBias:     input("output_bias", tensor.MustShape(3), 0),
	}
	feeds[weights.OutputBias] = reference.Value{Shape: weights.OutputBias.Shape, Data: []float32{1, 2, 3}}
	for _, block := range []int{0, 1, 3, 4} {
		prefix := "posnet"
		weights.PosNet[block] = WavPosNetGraphWeights{
			Norm1: input(prefix, tensor.MustShape(1, 2), 1), Norm1Bias: input(prefix, tensor.MustShape(1, 2), 0),
			Conv1: input(prefix, tensor.MustShape(3, 2, 2), 0), Conv1Bias: input(prefix, tensor.MustShape(1, 2), 0),
			Norm2: input(prefix, tensor.MustShape(1, 2), 1), Norm2Bias: input(prefix, tensor.MustShape(1, 2), 0),
			Conv2: input(prefix, tensor.MustShape(3, 2, 2), 0), Conv2Bias: input(prefix, tensor.MustShape(1, 2), 0),
		}
	}
	weights.PosNet[2] = WavPosNetGraphWeights{
		AttentionNorm: input("attn", tensor.MustShape(1, 2), 1), AttentionNormBias: input("attn", tensor.MustShape(1, 2), 0),
		AttentionQ: input("attn", tensor.MustShape(1, 2, 2), 0), AttentionQBias: input("attn", tensor.MustShape(1, 2), 0),
		AttentionK: input("attn", tensor.MustShape(1, 2, 2), 0), AttentionKBias: input("attn", tensor.MustShape(1, 2), 0),
		AttentionV: input("attn", tensor.MustShape(1, 2, 2), 0), AttentionVBias: input("attn", tensor.MustShape(1, 2), 0),
		AttentionOutput: input("attn", tensor.MustShape(1, 2, 2), 0), AttentionOutBias: input("attn", tensor.MustShape(1, 2), 0),
	}
	weights.PosNet[5] = WavPosNetGraphWeights{
		AttentionNorm:     input("final_group_norm", tensor.MustShape(1, 2), 1),
		AttentionNormBias: input("final_group_norm_bias", tensor.MustShape(1, 2), 0),
	}
	weights.ConvNext[0] = WavConvNextGraphWeights{
		Depthwise: input("dw", tensor.MustShape(7, 1, 2), 0), DepthwiseBias: input("dw_bias", tensor.MustShape(1, 2), 0),
		Norm: input("norm", tensor.MustShape(2), 1), NormBias: input("norm_bias", tensor.MustShape(2), 0),
		Pointwise1: input("pw1", tensor.MustShape(2, 4), 0), Pointwise1Bias: input("pw1_bias", tensor.MustShape(4), 0),
		Pointwise2: input("pw2", tensor.MustShape(4, 2), 0), Pointwise2Bias: input("pw2_bias", tensor.MustShape(2), 0),
		Gamma: input("gamma", tensor.MustShape(2), 1),
	}
	embeddings := input("embeddings", tensor.MustShape(2, 3), 0.5)
	program := fixtureModelPlan(t, spec, Weights{})
	output, err := program.BuildSequenceOutput(builder, embeddings, weights)
	if err != nil {
		t.Fatal(err)
	}
	results, err := reference.Execute([]*tensor.Tensor{output}, feeds)
	if err != nil {
		t.Fatal(err)
	}
	got := results[output]
	want := []float32{1, 2, 3, 1, 2, 3, 1, 2, 3}
	if !got.Shape.Equal(tensor.MustShape(3, 3)) || len(got.Data) != len(want) {
		t.Fatalf("output shape/data = %v/%v", got.Shape.Slice(), got.Data)
	}
	for index := range want {
		if got.Data[index] != want[index] {
			t.Fatalf("output[%d] = %g, want %g; output=%v", index, got.Data[index], want[index], got.Data)
		}
	}
}
