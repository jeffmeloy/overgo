package model

import (
	"testing"

	"overgo/internal/tensor"
	"overgo/internal/tensor/dtype"
)

func TestBuildHYV3MTPPipeline(t *testing.T) {
	builder := tensor.NewBuilder()
	spec := Spec{CommonSpec: CommonSpec{Architecture: "hy_v3", BlockCount: 1, NextNPredictLayers: 1,
		EmbeddingLength: 8, FeedForwardLength: 12,

		RMSNormEpsilon: 1e-5}, AttentionSpec: AttentionSpec{HeadCount: 2, HeadCountKV: 1, KeyLength: 4, ValueLength: 4,
		RopeDimensionCount: 4, RopeFrequencyBase: 10000},
	}
	token := builder.Input("token", dtype.F32, tensor.MustShape(8, 1))
	hidden := builder.Input("hidden", dtype.F32, tensor.MustShape(8, 1))
	enorm := builder.Input("enorm", dtype.F32, tensor.MustShape(8))
	hnorm := builder.Input("hnorm", dtype.F32, tensor.MustShape(8))
	projection := builder.Input("eh", dtype.F32, tensor.MustShape(16, 8))
	current, err := BuildHYV3MTPInput(builder, token, hidden, enorm, hnorm, projection, spec, 0)
	if err != nil {
		t.Fatal(err)
	}
	weights := LayerGraphWeights{
		AttentionNorm:   builder.Input("attn_norm", dtype.F32, tensor.MustShape(8)),
		AttentionQ:      builder.Input("q", dtype.F32, tensor.MustShape(8, 8)),
		AttentionK:      builder.Input("k", dtype.F32, tensor.MustShape(8, 4)),
		AttentionV:      builder.Input("v", dtype.F32, tensor.MustShape(8, 4)),
		AttentionOutput: builder.Input("o", dtype.F32, tensor.MustShape(8, 8)),
		AttentionQNorm:  builder.Input("q_norm", dtype.F32, tensor.MustShape(4)),
		AttentionKNorm:  builder.Input("k_norm", dtype.F32, tensor.MustShape(4)),
		FeedForwardNorm: builder.Input("ffn_norm", dtype.F32, tensor.MustShape(8)),
		FeedForwardGate: builder.Input("gate", dtype.F32, tensor.MustShape(8, 12)),
		FeedForwardUp:   builder.Input("up", dtype.F32, tensor.MustShape(8, 12)),
		FeedForwardDown: builder.Input("down", dtype.F32, tensor.MustShape(12, 8)),
	}
	block, err := BuildHYV3MTPBlockCached(
		builder, current, spec, weights, []uint32{4}, nil, nil, 0,
	)
	if err != nil {
		t.Fatal(err)
	}
	outputNorm := builder.Input("output_norm", dtype.F32, tensor.MustShape(8))
	head := builder.Input("head", dtype.F32, tensor.MustShape(8, 13))
	logits, nextHidden, err := BuildHYV3MTPOutputs(builder, block.Output, outputNorm, head, spec, 0)
	if err != nil {
		t.Fatal(err)
	}
	if nextHidden == block.Output || !logits.Shape.Equal(tensor.MustShape(13, 1)) ||
		!nextHidden.Shape.Equal(tensor.MustShape(8, 1)) ||
		!block.Key.Shape.Equal(tensor.MustShape(4, 1, 1)) {
		t.Fatalf("unexpected HY-V3 MTP graph: logits=%v hidden=%v key=%v", logits.Shape, nextHidden.Shape, block.Key.Shape)
	}
}
