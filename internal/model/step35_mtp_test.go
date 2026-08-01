package model

import (
	"math"
	"testing"

	"llamacpp2go/internal/tensor"
	"llamacpp2go/internal/tensor/dtype"
	"llamacpp2go/internal/tensor/reference"
)

func TestBuildStep35MTPPipeline(t *testing.T) {
	builder := tensor.NewBuilder()
	spec := Spec{
		Architecture: "step35", BlockCount: 1, NextNPredictLayers: 2,
		EmbeddingLength: 8, FeedForwardLength: 12,
		HeadCount: 2, HeadCountKV: 1,
		LayerHeadCounts: []uint32{2, 4, 2}, LayerKVHeadCounts: []uint32{1, 2, 1},
		KeyLength: 4, ValueLength: 4, RopeDimensionCount: 4,
		RopeFrequencyBase: 10000, RMSNormEpsilon: 1e-5,
		SlidingWindow: 32, SlidingLayers: []bool{false, true, false},
	}
	token := builder.Input("token", dtype.F32, tensor.MustShape(8, 1))
	hidden := builder.Input("hidden", dtype.F32, tensor.MustShape(8, 1))
	enorm := builder.Input("enorm", dtype.F32, tensor.MustShape(8))
	hnorm := builder.Input("hnorm", dtype.F32, tensor.MustShape(8))
	projection := builder.Input("eh", dtype.F32, tensor.MustShape(16, 8))
	current, err := BuildStep35MTPInput(builder, token, hidden, enorm, hnorm, projection, spec, 1)
	if err != nil {
		t.Fatal(err)
	}
	weights := LayerGraphWeights{
		AttentionNorm:   builder.Input("attn_norm", dtype.F32, tensor.MustShape(8)),
		AttentionQ:      builder.Input("q", dtype.F32, tensor.MustShape(8, 8)),
		AttentionK:      builder.Input("k", dtype.F32, tensor.MustShape(8, 4)),
		AttentionV:      builder.Input("v", dtype.F32, tensor.MustShape(8, 4)),
		AttentionOutput: builder.Input("o", dtype.F32, tensor.MustShape(8, 8)),
		FeedForwardNorm: builder.Input("ffn_norm", dtype.F32, tensor.MustShape(8)),
		FeedForwardGate: builder.Input("gate", dtype.F32, tensor.MustShape(8, 12)),
		FeedForwardUp:   builder.Input("up", dtype.F32, tensor.MustShape(8, 12)),
		FeedForwardDown: builder.Input("down", dtype.F32, tensor.MustShape(12, 8)),
	}
	block, err := BuildStep35MTPBlockCached(
		builder, current, spec, weights, []uint32{7}, nil, nil, 1,
	)
	if err != nil {
		t.Fatal(err)
	}
	outputNorm := builder.Input("output_norm", dtype.F32, tensor.MustShape(8))
	head := builder.Input("head", dtype.F32, tensor.MustShape(8, 13))
	logits, nextHidden, err := BuildStep35MTPOutputs(builder, block.Output, outputNorm, head, spec, 1)
	if err != nil {
		t.Fatal(err)
	}
	if nextHidden != block.Output {
		t.Fatal("Step3.5 MTP hidden state was normalized")
	}
	outputs := []*tensor.Tensor{logits, nextHidden, block.Key, block.Value}
	nodes, err := tensor.Topological(outputs...)
	if err != nil {
		t.Fatal(err)
	}
	feeds := make(map[*tensor.Tensor]reference.Value)
	for _, node := range nodes {
		if node.Op != tensor.OpInput {
			continue
		}
		elements, elementErr := node.Shape.Elements()
		if elementErr != nil {
			t.Fatal(elementErr)
		}
		data := make([]float32, elements)
		for index := range data {
			data[index] = 0.05 + float32(index%11)*0.01
		}
		feeds[node] = reference.Value{Shape: node.Shape, Data: data}
	}
	results, err := reference.Execute(outputs, feeds)
	if err != nil {
		t.Fatal(err)
	}
	if !results[logits].Shape.Equal(tensor.MustShape(13, 1)) ||
		!results[nextHidden].Shape.Equal(tensor.MustShape(8, 1)) ||
		!results[block.Key].Shape.Equal(tensor.MustShape(4, 1, 1)) {
		t.Fatalf("unexpected Step3.5 MTP shapes: %v %v %v", results[logits].Shape, results[nextHidden].Shape, results[block.Key].Shape)
	}
	for _, value := range results[logits].Data {
		if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
			t.Fatalf("non-finite Step3.5 MTP logit: %v", value)
		}
	}
}
