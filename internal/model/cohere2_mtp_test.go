package model

import (
	"testing"

	"llamacpp2go/internal/tensor"
	"llamacpp2go/internal/tensor/dtype"
)

func TestBuildCohere2MTPPipeline(t *testing.T) {
	builder := tensor.NewBuilder()
	spec := Spec{
		Architecture: "cohere2moe", BlockCount: 2, NextNPredictLayers: 1,
		EmbeddingLength: 8, HeadCount: 2, HeadCountKV: 1,
		KeyLength: 4, ValueLength: 4, RopeDimensionCount: 4,
		RopeFrequencyBase: 10000, RopeFrequencySWA: 20000,
		RMSNormEpsilon: 1e-5, SlidingWindow: 128,
		SlidingLayers: []bool{false, true}, LeadingDenseBlocks: 1,
		ExpertCount: 4, ExpertUsedCount: 2, ExpertFeedForward: 6,
		ExpertGatingFunc: 2, ExpertWeightsScale: 1, SharedExpertCount: 1, SharedExpertFF: 6,
		VocabularySize: 32, LogitScale: 0.5,
	}
	input := builder.Input("token", dtype.F32, tensor.MustShape(8, 2))
	hidden := builder.Input("hidden", dtype.F32, tensor.MustShape(8, 2))
	norm := builder.Input("norm", dtype.F32, tensor.MustShape(8))
	projection := builder.Input("projection", dtype.F32, tensor.MustShape(16, 8))
	current, err := BuildCohere2MTPInput(builder, input, hidden, norm, norm, projection, spec)
	if err != nil {
		t.Fatal(err)
	}
	weights := LayerGraphWeights{
		AttentionNorm:            norm,
		AttentionQ:               builder.Input("q", dtype.F32, tensor.MustShape(8, 8)),
		AttentionK:               builder.Input("k", dtype.F32, tensor.MustShape(8, 4)),
		AttentionV:               builder.Input("v", dtype.F32, tensor.MustShape(8, 4)),
		AttentionOutput:          builder.Input("o", dtype.F32, tensor.MustShape(8, 8)),
		FeedForwardRouter:        builder.Input("router", dtype.F32, tensor.MustShape(8, 4)),
		FeedForwardGateUpExperts: builder.Input("gate_up", dtype.F32, tensor.MustShape(8, 12, 4)),
		FeedForwardDownExperts:   builder.Input("down", dtype.F32, tensor.MustShape(6, 8, 4)),
		FeedForwardSharedGate:    builder.Input("sg", dtype.F32, tensor.MustShape(8, 6)),
		FeedForwardSharedUp:      builder.Input("su", dtype.F32, tensor.MustShape(8, 6)),
		FeedForwardSharedDown:    builder.Input("sd", dtype.F32, tensor.MustShape(6, 8)),
	}
	block, err := BuildCohere2MTPBlockCached(builder, current, spec, weights, []uint32{0, 1}, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	head := builder.Input("head", dtype.F32, tensor.MustShape(8, 32))
	logits, nextHidden, err := BuildCohere2MTPOutputs(builder, block.Output, norm, head, spec)
	if err != nil {
		t.Fatal(err)
	}
	nodes, err := tensor.Topological(logits)
	if err != nil {
		t.Fatal(err)
	}
	var rope, half, scaled int
	var window uint32
	for _, node := range nodes {
		switch node.Op {
		case tensor.OpRoPENormal:
			rope++
		case tensor.OpAttention:
			window = node.Attrs.(tensor.AttentionAttributes).Window
		case tensor.OpScale:
			value := node.Attrs.(tensor.ScaleAttributes).Value
			if value == 0.5 && node == logits {
				scaled++
			} else if value == 0.5 {
				half++
			}
		}
	}
	if logits.Shape.Dims[0] != 32 || !nextHidden.Shape.Equal(tensor.MustShape(8, 2)) ||
		rope != 2 || window != 0 || half != 1 || scaled != 1 {
		t.Fatalf("unexpected Cohere2-MoE MTP graph: logits=%v hidden=%v rope=%d window=%d half=%d scale=%d", logits.Shape, nextHidden.Shape, rope, window, half, scaled)
	}
}
