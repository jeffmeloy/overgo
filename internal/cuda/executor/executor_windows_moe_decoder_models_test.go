//go:build windows

package executor

import (
	"context"

	"overgo/internal/model"

	"overgo/internal/tensor"

	"overgo/internal/tensor/dtype"

	"overgo/internal/tensor/reference"

	"testing"

	cudatest "overgo/internal/cuda/testutil"
)

func TestExecutorGPTJBlockMatchesReference(t *testing.T) {
	cudatest.Require(t)
	builder := tensor.NewBuilder()
	feeds := make(map[*tensor.Tensor]reference.Value)
	seed := 3
	input := func(name string, shape tensor.Shape, scale, offset float32) *tensor.Tensor {
		item := builder.Input(name, dtype.F32, shape)
		feeds[item] = patternedValue(shape, seed, scale, offset)
		seed += 2
		return item
	}
	spec := model.Spec{CommonSpec: model.CommonSpec{Architecture: "gptj", EmbeddingLength: 8,
		FeedForwardLength: 12, LayerNormEpsilon: 1e-5}, AttentionSpec: model.AttentionSpec{
		HeadCount: 2, HeadCountKV: 2, KeyLength: 4, ValueLength: 4,
		RopeDimensionCount: 2, RopeFrequencyBase: 10000}}
	weights := model.LayerGraphWeights{
		AttentionNorm:       input("attn_norm", tensor.MustShape(8), 0.03, 0.9),
		AttentionNormBias:   input("attn_norm_bias", tensor.MustShape(8), 0.02, -0.03),
		AttentionQ:          input("q", tensor.MustShape(8, 8), 0.03, -0.1),
		AttentionK:          input("k", tensor.MustShape(8, 8), 0.03, -0.1),
		AttentionV:          input("v", tensor.MustShape(8, 8), 0.03, -0.1),
		AttentionOutput:     input("attn_out", tensor.MustShape(8, 8), 0.03, -0.1),
		FeedForwardUp:       input("ffn_up", tensor.MustShape(8, 12), 0.03, -0.1),
		FeedForwardUpBias:   input("ffn_up_bias", tensor.MustShape(12), 0.02, -0.03),
		FeedForwardDown:     input("ffn_down", tensor.MustShape(12, 8), 0.03, -0.1),
		FeedForwardDownBias: input("ffn_down_bias", tensor.MustShape(8), 0.02, -0.03),
	}
	current := input("current", tensor.MustShape(8, 3), 0.08, -0.1)
	result, err := model.BuildDenseBlockCachedForLayer(
		builder, current, spec, weights, []uint32{0, 1, 2}, nil, nil, 0,
	)
	if err != nil {
		t.Fatal(err)
	}
	outputs := []*tensor.Tensor{result.Output, result.Key, result.Value}
	want, err := reference.Execute(outputs, feeds)
	if err != nil {
		t.Fatal(err)
	}
	cuda, err := New(0)
	if err != nil {
		t.Fatal(err)
	}
	defer cuda.Close()
	got, err := cuda.Execute(context.Background(), outputs, feeds)
	if err != nil {
		t.Fatal(err)
	}
	compare(t, got[result.Output].Data, want[result.Output].Data, 2e-3)
	compare(t, got[result.Key].Data, want[result.Key].Data, 5e-5)
	compare(t, got[result.Value].Data, want[result.Value].Data, 5e-5)
}

func TestExecutorRND1NonCausalMoEBlockMatchesReference(t *testing.T) {
	cudatest.Require(t)
	builder := tensor.NewBuilder()
	spec := model.Spec{CommonSpec: model.CommonSpec{Architecture: "rnd1", EmbeddingLength: 8, FeedForwardLength: 24,

		RMSNormEpsilon: 1e-6}, AttentionSpec: model.AttentionSpec{HeadCount: 2, HeadCountKV: 1, KeyLength: 4, ValueLength: 4,
		RopeFrequencyBase: 1_000_000, NonCausalAttention: true}, MoESpec: model.MoESpec{ExpertCount: 4, ExpertUsedCount: 2, ExpertFeedForward: 12, ExpertWeightsScale: 1},
	}
	input := builder.Input("input", dtype.F32, tensor.MustShape(8, 3))
	weights := model.LayerGraphWeights{
		AttentionNorm:          builder.Input("attn_norm", dtype.F32, tensor.MustShape(8)),
		AttentionQ:             builder.Input("q", dtype.F32, tensor.MustShape(8, 8)),
		AttentionK:             builder.Input("k", dtype.F32, tensor.MustShape(8, 4)),
		AttentionV:             builder.Input("v", dtype.F32, tensor.MustShape(8, 4)),
		AttentionOutput:        builder.Input("attn_out", dtype.F32, tensor.MustShape(8, 8)),
		AttentionQNorm:         builder.Input("q_norm", dtype.F32, tensor.MustShape(4)),
		AttentionKNorm:         builder.Input("k_norm", dtype.F32, tensor.MustShape(4)),
		FeedForwardNorm:        builder.Input("ffn_norm", dtype.F32, tensor.MustShape(8)),
		FeedForwardRouter:      builder.Input("router", dtype.F32, tensor.MustShape(8, 4)),
		FeedForwardGateExperts: builder.Input("gate_exps", dtype.F32, tensor.MustShape(8, 12, 4)),
		FeedForwardUpExperts:   builder.Input("up_exps", dtype.F32, tensor.MustShape(8, 12, 4)),
		FeedForwardDownExperts: builder.Input("down_exps", dtype.F32, tensor.MustShape(12, 8, 4)),
	}
	result, err := model.BuildDenseBlockCachedForLayer(
		builder, input, spec, weights, []uint32{0, 1, 2}, nil, nil, 0,
	)
	if err != nil {
		t.Fatal(err)
	}
	feeds := map[*tensor.Tensor]reference.Value{input: patternedValue(input.Shape, 3, 0.2, 0)}
	for index, node := range []*tensor.Tensor{
		weights.AttentionQ, weights.AttentionK, weights.AttentionV, weights.AttentionOutput,
		weights.FeedForwardRouter, weights.FeedForwardGateExperts,
		weights.FeedForwardUpExperts, weights.FeedForwardDownExperts,
	} {
		feeds[node] = patternedValue(node.Shape, index+11, 0.06, 0)
	}
	feeds[weights.AttentionNorm] = patternedValue(weights.AttentionNorm.Shape, 31, 0.03, 1)
	feeds[weights.AttentionQNorm] = patternedValue(weights.AttentionQNorm.Shape, 37, 0.03, 1)
	feeds[weights.AttentionKNorm] = patternedValue(weights.AttentionKNorm.Shape, 41, 0.03, 1)
	feeds[weights.FeedForwardNorm] = patternedValue(weights.FeedForwardNorm.Shape, 43, 0.03, 1)
	outputs := []*tensor.Tensor{result.Output, result.Key, result.Value}
	want, err := reference.Execute(outputs, feeds)
	if err != nil {
		t.Fatal(err)
	}
	cuda, err := New(0)
	if err != nil {
		t.Fatal(err)
	}
	defer cuda.Close()
	got, err := cuda.Execute(context.Background(), outputs, feeds)
	if err != nil {
		t.Fatal(err)
	}
	compare(t, got[result.Output].Data, want[result.Output].Data, 7e-4)
	compare(t, got[result.Key].Data, want[result.Key].Data, 5e-5)
	compare(t, got[result.Value].Data, want[result.Value].Data, 5e-5)
}

func TestExecutorLLaDAMoEBlockMatchesReference(t *testing.T) {
	cudatest.Require(t)
	builder := tensor.NewBuilder()
	spec := model.Spec{CommonSpec: model.CommonSpec{Architecture: "llada-moe", EmbeddingLength: 8, FeedForwardLength: 12,

		RMSNormEpsilon: 1e-6}, AttentionSpec: model.AttentionSpec{HeadCount: 2, HeadCountKV: 1, KeyLength: 4, ValueLength: 4,
		RopeFrequencyBase: 10000, NonCausalAttention: true}, MoESpec: model.MoESpec{ExpertCount: 4, ExpertUsedCount: 2, ExpertFeedForward: 6, ExpertWeightsScale: 1},
	}
	input := builder.Input("input", dtype.F32, tensor.MustShape(8, 3))
	weights := model.LayerGraphWeights{
		AttentionNorm:          builder.Input("attn_norm", dtype.F32, tensor.MustShape(8)),
		AttentionQ:             builder.Input("q", dtype.F32, tensor.MustShape(8, 8)),
		AttentionK:             builder.Input("k", dtype.F32, tensor.MustShape(8, 4)),
		AttentionV:             builder.Input("v", dtype.F32, tensor.MustShape(8, 4)),
		AttentionOutput:        builder.Input("attn_out", dtype.F32, tensor.MustShape(8, 8)),
		AttentionQNorm:         builder.Input("q_norm", dtype.F32, tensor.MustShape(4)),
		AttentionKNorm:         builder.Input("k_norm", dtype.F32, tensor.MustShape(4)),
		FeedForwardNorm:        builder.Input("ffn_norm", dtype.F32, tensor.MustShape(8)),
		FeedForwardRouter:      builder.Input("router", dtype.F32, tensor.MustShape(8, 4)),
		FeedForwardGateExperts: builder.Input("gate_exps", dtype.F32, tensor.MustShape(8, 6, 4)),
		FeedForwardUpExperts:   builder.Input("up_exps", dtype.F32, tensor.MustShape(8, 6, 4)),
		FeedForwardDownExperts: builder.Input("down_exps", dtype.F32, tensor.MustShape(6, 8, 4)),
	}
	result, err := model.BuildDenseBlockCachedForLayer(builder, input, spec, weights, []uint32{0, 1, 2}, nil, nil, 0)
	if err != nil {
		t.Fatal(err)
	}
	feeds := map[*tensor.Tensor]reference.Value{input: patternedValue(input.Shape, 3, 0.2, 0)}
	for index, node := range []*tensor.Tensor{
		weights.AttentionQ, weights.AttentionK, weights.AttentionV, weights.AttentionOutput,
		weights.FeedForwardRouter, weights.FeedForwardGateExperts,
		weights.FeedForwardUpExperts, weights.FeedForwardDownExperts,
	} {
		feeds[node] = patternedValue(node.Shape, index+11, 0.06, 0)
	}
	for index, node := range []*tensor.Tensor{
		weights.AttentionNorm, weights.AttentionQNorm, weights.AttentionKNorm, weights.FeedForwardNorm,
	} {
		feeds[node] = patternedValue(node.Shape, index+31, 0.03, 1)
	}
	outputs := []*tensor.Tensor{result.Output, result.Key, result.Value}
	want, err := reference.Execute(outputs, feeds)
	if err != nil {
		t.Fatal(err)
	}
	cuda, err := New(0)
	if err != nil {
		t.Fatal(err)
	}
	defer cuda.Close()
	got, err := cuda.Execute(context.Background(), outputs, feeds)
	if err != nil {
		t.Fatal(err)
	}
	compare(t, got[result.Output].Data, want[result.Output].Data, 7e-4)
	compare(t, got[result.Key].Data, want[result.Key].Data, 5e-5)
	compare(t, got[result.Value].Data, want[result.Value].Data, 5e-5)
}

func TestExecutorLagunaYaRNMoEBlockMatchesReference(t *testing.T) {
	cudatest.Require(t)
	builder := tensor.NewBuilder()
	spec := model.Spec{CommonSpec: model.CommonSpec{Architecture: "laguna", BlockCount: 2, EmbeddingLength: 8,
		FeedForwardLength: 16,

		RMSNormEpsilon: 1e-6}, AttentionSpec: model.AttentionSpec{HeadCount: 2, HeadCountKV: 1, LayerHeadCounts: []uint32{2, 4},
		LayerKVHeadCounts: []uint32{1, 1}, KeyLength: 4, ValueLength: 4,
		RopeFrequencyBase: 500000, RopeDimensionCount: 4, RopeScalingType: "yarn",
		RopeScalingFactor: 4, OriginalContextLength: 2048, YaRNExtFactor: 1,
		YaRNAttentionFactor: 1, YaRNBetaFast: 32, YaRNBetaSlow: 1}, MoESpec: model.MoESpec{LeadingDenseBlocks: 1,
		ExpertCount: 4, ExpertUsedCount: 2, ExpertFeedForward: 12,
		SharedExpertFF: 10, ExpertWeightsScale: 1.25, ExpertWeightsNorm: true},
	}
	input := builder.Input("input", dtype.F32, tensor.MustShape(8, 3))
	weights := model.LayerGraphWeights{
		AttentionNorm:          builder.Input("attn_norm", dtype.F32, tensor.MustShape(8)),
		AttentionQ:             builder.Input("q", dtype.F32, tensor.MustShape(8, 16)),
		AttentionK:             builder.Input("k", dtype.F32, tensor.MustShape(8, 4)),
		AttentionV:             builder.Input("v", dtype.F32, tensor.MustShape(8, 4)),
		AttentionOutput:        builder.Input("attn_out", dtype.F32, tensor.MustShape(16, 8)),
		AttentionQNorm:         builder.Input("q_norm", dtype.F32, tensor.MustShape(4)),
		AttentionKNorm:         builder.Input("k_norm", dtype.F32, tensor.MustShape(4)),
		AttentionOutputGate:    builder.Input("attn_gate", dtype.F32, tensor.MustShape(8, 16)),
		FeedForwardNorm:        builder.Input("ffn_norm", dtype.F32, tensor.MustShape(8)),
		FeedForwardRouter:      builder.Input("router", dtype.F32, tensor.MustShape(8, 4)),
		FeedForwardGateExperts: builder.Input("gate_exps", dtype.F32, tensor.MustShape(8, 12, 4)),
		FeedForwardUpExperts:   builder.Input("up_exps", dtype.F32, tensor.MustShape(8, 12, 4)),
		FeedForwardDownExperts: builder.Input("down_exps", dtype.F32, tensor.MustShape(12, 8, 4)),
		FeedForwardExpertBias:  builder.Input("correction", dtype.F32, tensor.MustShape(4)),
		FeedForwardSharedGate:  builder.Input("shared_gate", dtype.F32, tensor.MustShape(8, 10)),
		FeedForwardSharedUp:    builder.Input("shared_up", dtype.F32, tensor.MustShape(8, 10)),
		FeedForwardSharedDown:  builder.Input("shared_down", dtype.F32, tensor.MustShape(10, 8)),
	}
	result, err := model.BuildDenseBlockCachedForLayer(builder, input, spec, weights, []uint32{0, 1, 2}, nil, nil, 1)
	if err != nil {
		t.Fatal(err)
	}
	feeds := map[*tensor.Tensor]reference.Value{input: patternedValue(input.Shape, 3, 0.2, 0)}
	allWeights := []*tensor.Tensor{
		weights.AttentionQ, weights.AttentionK, weights.AttentionV, weights.AttentionOutput,
		weights.AttentionOutputGate, weights.FeedForwardRouter, weights.FeedForwardGateExperts,
		weights.FeedForwardUpExperts, weights.FeedForwardDownExperts, weights.FeedForwardExpertBias,
		weights.FeedForwardSharedGate, weights.FeedForwardSharedUp, weights.FeedForwardSharedDown,
	}
	for index, node := range allWeights {
		feeds[node] = patternedValue(node.Shape, index+11, 0.05, 0)
	}
	for index, node := range []*tensor.Tensor{weights.AttentionNorm, weights.AttentionQNorm, weights.AttentionKNorm, weights.FeedForwardNorm} {
		feeds[node] = patternedValue(node.Shape, index+37, 0.03, 1)
	}
	outputs := []*tensor.Tensor{result.Output, result.Key, result.Value}
	want, err := reference.Execute(outputs, feeds)
	if err != nil {
		t.Fatal(err)
	}
	cuda, err := New(0)
	if err != nil {
		t.Fatal(err)
	}
	defer cuda.Close()
	got, err := cuda.Execute(context.Background(), outputs, feeds)
	if err != nil {
		t.Fatal(err)
	}
	compare(t, got[result.Output].Data, want[result.Output].Data, 1e-3)
	compare(t, got[result.Key].Data, want[result.Key].Data, 7e-5)
	compare(t, got[result.Value].Data, want[result.Value].Data, 5e-5)
}

func TestExecutorAFMoEBlockMatchesReference(t *testing.T) {
	cudatest.Require(t)
	builder := tensor.NewBuilder()
	spec := model.Spec{CommonSpec: model.CommonSpec{Architecture: "afmoe", BlockCount: 1, EmbeddingLength: 8,
		FeedForwardLength: 16,

		RMSNormEpsilon: 1e-6}, AttentionSpec: model.AttentionSpec{HeadCount: 2, HeadCountKV: 1, KeyLength: 4, ValueLength: 4,
		RopeFrequencyBase: 10000, RopeDimensionCount: 4, NoRopeLayerStep: 4,
		SlidingWindow: 64, SlidingPattern: 4, RopeFrequencySWA: 10000}, MoESpec: model.MoESpec{ExpertCount: 4, ExpertUsedCount: 2,
		ExpertFeedForward: 6, SharedExpertCount: 2, SharedExpertFF: 12,
		ExpertWeightsScale: 2.826, ExpertWeightsNorm: true},
	}
	input := builder.Input("input", dtype.F32, tensor.MustShape(8, 3))
	weights := model.LayerGraphWeights{
		AttentionNorm:          builder.Input("attn_norm", dtype.F32, tensor.MustShape(8)),
		AttentionPostNorm:      builder.Input("attn_post_norm", dtype.F32, tensor.MustShape(8)),
		AttentionQ:             builder.Input("q", dtype.F32, tensor.MustShape(8, 8)),
		AttentionK:             builder.Input("k", dtype.F32, tensor.MustShape(8, 4)),
		AttentionV:             builder.Input("v", dtype.F32, tensor.MustShape(8, 4)),
		AttentionOutput:        builder.Input("attn_out", dtype.F32, tensor.MustShape(8, 8)),
		AttentionQNorm:         builder.Input("q_norm", dtype.F32, tensor.MustShape(4)),
		AttentionKNorm:         builder.Input("k_norm", dtype.F32, tensor.MustShape(4)),
		AttentionOutputGate:    builder.Input("attn_gate", dtype.F32, tensor.MustShape(8, 8)),
		FeedForwardNorm:        builder.Input("ffn_norm", dtype.F32, tensor.MustShape(8)),
		FeedForwardPostNorm:    builder.Input("ffn_post_norm", dtype.F32, tensor.MustShape(8)),
		FeedForwardRouter:      builder.Input("router", dtype.F32, tensor.MustShape(8, 4)),
		FeedForwardGateExperts: builder.Input("gate_exps", dtype.F32, tensor.MustShape(8, 6, 4)),
		FeedForwardUpExperts:   builder.Input("up_exps", dtype.F32, tensor.MustShape(8, 6, 4)),
		FeedForwardDownExperts: builder.Input("down_exps", dtype.F32, tensor.MustShape(6, 8, 4)),
		FeedForwardExpertBias:  builder.Input("correction", dtype.F32, tensor.MustShape(4)),
		FeedForwardSharedGate:  builder.Input("shared_gate", dtype.F32, tensor.MustShape(8, 12)),
		FeedForwardSharedUp:    builder.Input("shared_up", dtype.F32, tensor.MustShape(8, 12)),
		FeedForwardSharedDown:  builder.Input("shared_down", dtype.F32, tensor.MustShape(12, 8)),
	}
	result, err := model.BuildDenseBlockCachedForLayer(builder, input, spec, weights, []uint32{0, 1, 2}, nil, nil, 0)
	if err != nil {
		t.Fatal(err)
	}
	feeds := map[*tensor.Tensor]reference.Value{input: patternedValue(input.Shape, 3, 0.2, 0)}
	for index, node := range []*tensor.Tensor{
		weights.AttentionQ, weights.AttentionK, weights.AttentionV, weights.AttentionOutput,
		weights.AttentionOutputGate, weights.FeedForwardRouter, weights.FeedForwardGateExperts,
		weights.FeedForwardUpExperts, weights.FeedForwardDownExperts, weights.FeedForwardExpertBias,
		weights.FeedForwardSharedGate, weights.FeedForwardSharedUp, weights.FeedForwardSharedDown,
	} {
		feeds[node] = patternedValue(node.Shape, index+11, 0.05, 0)
	}
	for index, node := range []*tensor.Tensor{
		weights.AttentionNorm, weights.AttentionPostNorm, weights.AttentionQNorm,
		weights.AttentionKNorm, weights.FeedForwardNorm, weights.FeedForwardPostNorm,
	} {
		feeds[node] = patternedValue(node.Shape, index+37, 0.03, 1)
	}
	outputs := []*tensor.Tensor{result.Output, result.Key, result.Value}
	want, err := reference.Execute(outputs, feeds)
	if err != nil {
		t.Fatal(err)
	}
	cuda, err := New(0)
	if err != nil {
		t.Fatal(err)
	}
	defer cuda.Close()
	got, err := cuda.Execute(context.Background(), outputs, feeds)
	if err != nil {
		t.Fatal(err)
	}
	compare(t, got[result.Output].Data, want[result.Output].Data, 1e-3)
	compare(t, got[result.Key].Data, want[result.Key].Data, 7e-5)
	compare(t, got[result.Value].Data, want[result.Value].Data, 5e-5)
}

func TestExecutorOLMoEBlockMatchesReference(t *testing.T) {
	cudatest.Require(t)
	builder := tensor.NewBuilder()
	spec := model.Spec{CommonSpec: model.CommonSpec{Architecture: "olmoe", EmbeddingLength: 8, FeedForwardLength: 12,

		RMSNormEpsilon: 1e-6}, AttentionSpec: model.AttentionSpec{HeadCount: 2, HeadCountKV: 2, KeyLength: 4, ValueLength: 4,
		RopeFrequencyBase: 10000}, MoESpec: model.MoESpec{ExpertCount: 4, ExpertUsedCount: 2, ExpertFeedForward: 12, ExpertWeightsScale: 1},
	}
	input := builder.Input("input", dtype.F32, tensor.MustShape(8, 3))
	weights := model.LayerGraphWeights{
		AttentionNorm:          builder.Input("attn_norm", dtype.F32, tensor.MustShape(8)),
		AttentionQ:             builder.Input("q", dtype.F32, tensor.MustShape(8, 8)),
		AttentionK:             builder.Input("k", dtype.F32, tensor.MustShape(8, 8)),
		AttentionV:             builder.Input("v", dtype.F32, tensor.MustShape(8, 8)),
		AttentionOutput:        builder.Input("attn_out", dtype.F32, tensor.MustShape(8, 8)),
		AttentionQNorm:         builder.Input("q_norm", dtype.F32, tensor.MustShape(8)),
		AttentionKNorm:         builder.Input("k_norm", dtype.F32, tensor.MustShape(8)),
		FeedForwardNorm:        builder.Input("ffn_norm", dtype.F32, tensor.MustShape(8)),
		FeedForwardRouter:      builder.Input("router", dtype.F32, tensor.MustShape(8, 4)),
		FeedForwardGateExperts: builder.Input("gate_exps", dtype.F32, tensor.MustShape(8, 12, 4)),
		FeedForwardUpExperts:   builder.Input("up_exps", dtype.F32, tensor.MustShape(8, 12, 4)),
		FeedForwardDownExperts: builder.Input("down_exps", dtype.F32, tensor.MustShape(12, 8, 4)),
	}
	result, err := model.BuildDenseBlockCachedForLayer(
		builder, input, spec, weights, []uint32{0, 1, 2}, nil, nil, 0,
	)
	if err != nil {
		t.Fatal(err)
	}
	feeds := map[*tensor.Tensor]reference.Value{input: patternedValue(input.Shape, 3, 0.2, 0)}
	for index, node := range []*tensor.Tensor{
		weights.AttentionQ, weights.AttentionK, weights.AttentionV, weights.AttentionOutput,
		weights.FeedForwardRouter, weights.FeedForwardGateExperts,
		weights.FeedForwardUpExperts, weights.FeedForwardDownExperts,
	} {
		feeds[node] = patternedValue(node.Shape, index+11, 0.06, 0)
	}
	for index, node := range []*tensor.Tensor{
		weights.AttentionNorm, weights.AttentionQNorm, weights.AttentionKNorm, weights.FeedForwardNorm,
	} {
		feeds[node] = patternedValue(node.Shape, index+37, 0.03, 1)
	}
	outputs := []*tensor.Tensor{result.Output, result.Key, result.Value}
	want, err := reference.Execute(outputs, feeds)
	if err != nil {
		t.Fatal(err)
	}
	cuda, err := New(0)
	if err != nil {
		t.Fatal(err)
	}
	defer cuda.Close()
	got, err := cuda.Execute(context.Background(), outputs, feeds)
	if err != nil {
		t.Fatal(err)
	}
	compare(t, got[result.Output].Data, want[result.Output].Data, 8e-4)
	compare(t, got[result.Key].Data, want[result.Key].Data, 5e-5)
	compare(t, got[result.Value].Data, want[result.Value].Data, 5e-5)
}

func TestExecutorMixtralBlockMatchesReference(t *testing.T) {
	cudatest.Require(t)
	builder := tensor.NewBuilder()
	spec := model.Spec{CommonSpec: model.CommonSpec{Architecture: "llama", EmbeddingLength: 8, FeedForwardLength: 12,

		RMSNormEpsilon: 1e-6}, AttentionSpec: model.AttentionSpec{HeadCount: 2, HeadCountKV: 1, KeyLength: 4, ValueLength: 4,
		RopeFrequencyBase: 10000}, MoESpec: model.MoESpec{ExpertCount: 4, ExpertUsedCount: 2, ExpertFeedForward: 12, ExpertWeightsScale: 1},
	}
	input := builder.Input("input", dtype.F32, tensor.MustShape(8, 3))
	weights := model.LayerGraphWeights{
		AttentionNorm:          builder.Input("attn_norm", dtype.F32, tensor.MustShape(8)),
		AttentionQ:             builder.Input("q", dtype.F32, tensor.MustShape(8, 8)),
		AttentionK:             builder.Input("k", dtype.F32, tensor.MustShape(8, 4)),
		AttentionV:             builder.Input("v", dtype.F32, tensor.MustShape(8, 4)),
		AttentionOutput:        builder.Input("attn_out", dtype.F32, tensor.MustShape(8, 8)),
		FeedForwardNorm:        builder.Input("ffn_norm", dtype.F32, tensor.MustShape(8)),
		FeedForwardRouter:      builder.Input("router", dtype.F32, tensor.MustShape(8, 4)),
		FeedForwardGateExperts: builder.Input("gate_exps", dtype.F32, tensor.MustShape(8, 12, 4)),
		FeedForwardUpExperts:   builder.Input("up_exps", dtype.F32, tensor.MustShape(8, 12, 4)),
		FeedForwardDownExperts: builder.Input("down_exps", dtype.F32, tensor.MustShape(12, 8, 4)),
	}
	result, err := model.BuildDenseBlockCachedForLayer(
		builder, input, spec, weights, []uint32{0, 1, 2}, nil, nil, 0,
	)
	if err != nil {
		t.Fatal(err)
	}
	feeds := map[*tensor.Tensor]reference.Value{input: patternedValue(input.Shape, 3, 0.2, 0)}
	for index, node := range []*tensor.Tensor{
		weights.AttentionQ, weights.AttentionK, weights.AttentionV, weights.AttentionOutput,
		weights.FeedForwardRouter, weights.FeedForwardGateExperts,
		weights.FeedForwardUpExperts, weights.FeedForwardDownExperts,
	} {
		feeds[node] = patternedValue(node.Shape, index+11, 0.06, 0)
	}
	for index, node := range []*tensor.Tensor{
		weights.AttentionNorm, weights.FeedForwardNorm,
	} {
		feeds[node] = patternedValue(node.Shape, index+37, 0.03, 1)
	}
	outputs := []*tensor.Tensor{result.Output, result.Key, result.Value}
	want, err := reference.Execute(outputs, feeds)
	if err != nil {
		t.Fatal(err)
	}
	cuda, err := New(0)
	if err != nil {
		t.Fatal(err)
	}
	defer cuda.Close()
	got, err := cuda.Execute(context.Background(), outputs, feeds)
	if err != nil {
		t.Fatal(err)
	}
	compare(t, got[result.Output].Data, want[result.Output].Data, 8e-4)
	compare(t, got[result.Key].Data, want[result.Key].Data, 5e-5)
	compare(t, got[result.Value].Data, want[result.Value].Data, 5e-5)
}

func TestExecutorPhiMoEBlockMatchesReference(t *testing.T) {
	cudatest.Require(t)
	builder := tensor.NewBuilder()
	spec := model.Spec{CommonSpec: model.CommonSpec{Architecture: "phimoe", BlockCount: 1, ContextLength: 128,
		EmbeddingLength: 8, FeedForwardLength: 12,

		RMSNormEpsilon: 1e-5}, AttentionSpec: model.AttentionSpec{OriginalContextLength: 32,

		HeadCount: 2, HeadCountKV: 2, KeyLength: 4, ValueLength: 4,
		RopeDimensionCount: 4, RopeFrequencyBase: 10000,
		RopeScalingType: "longrope", RopeAttentionFactor: 1.1}, MoESpec: model.MoESpec{ExpertCount: 4, ExpertUsedCount: 2, ExpertFeedForward: 12, ExpertWeightsScale: 1},
	}
	input := builder.Input("input", dtype.F32, tensor.MustShape(8, 3))
	weights := model.LayerGraphWeights{
		AttentionNorm:          builder.Input("attn_norm", dtype.F32, tensor.MustShape(8)),
		AttentionNormBias:      builder.Input("attn_norm_bias", dtype.F32, tensor.MustShape(8)),
		AttentionQ:             builder.Input("q", dtype.F32, tensor.MustShape(8, 8)),
		AttentionK:             builder.Input("k", dtype.F32, tensor.MustShape(8, 8)),
		AttentionV:             builder.Input("v", dtype.F32, tensor.MustShape(8, 8)),
		AttentionOutput:        builder.Input("attn_out", dtype.F32, tensor.MustShape(8, 8)),
		AttentionOutputBias:    builder.Input("attn_out_bias", dtype.F32, tensor.MustShape(8)),
		FeedForwardNorm:        builder.Input("ffn_norm", dtype.F32, tensor.MustShape(8)),
		FeedForwardNormBias:    builder.Input("ffn_norm_bias", dtype.F32, tensor.MustShape(8)),
		FeedForwardRouter:      builder.Input("router", dtype.F32, tensor.MustShape(8, 4)),
		FeedForwardGateExperts: builder.Input("gate_exps", dtype.F32, tensor.MustShape(8, 12, 4)),
		FeedForwardUpExperts:   builder.Input("up_exps", dtype.F32, tensor.MustShape(8, 12, 4)),
		FeedForwardDownExperts: builder.Input("down_exps", dtype.F32, tensor.MustShape(12, 8, 4)),
		RopeFactors:            builder.Input("rope_long", dtype.F32, tensor.MustShape(2)),
	}
	result, err := model.BuildDenseBlockCachedForLayer(
		builder, input, spec, weights, []uint32{0, 1, 2}, nil, nil, 0,
	)
	if err != nil {
		t.Fatal(err)
	}
	feeds := map[*tensor.Tensor]reference.Value{input: patternedValue(input.Shape, 3, 0.2, 0)}
	for index, node := range []*tensor.Tensor{
		weights.AttentionQ, weights.AttentionK, weights.AttentionV, weights.AttentionOutput,
		weights.FeedForwardRouter, weights.FeedForwardGateExperts,
		weights.FeedForwardUpExperts, weights.FeedForwardDownExperts,
	} {
		feeds[node] = patternedValue(node.Shape, index+11, 0.06, 0)
	}
	for index, node := range []*tensor.Tensor{
		weights.AttentionNorm, weights.FeedForwardNorm,
	} {
		feeds[node] = patternedValue(node.Shape, index+37, 0.03, 1)
	}
	for index, node := range []*tensor.Tensor{
		weights.AttentionNormBias, weights.AttentionOutputBias, weights.FeedForwardNormBias,
	} {
		feeds[node] = patternedValue(node.Shape, index+43, 0.01, 0)
	}
	ropeValue, valueErr := reference.NewValue(weights.RopeFactors.Shape, []float32{1, 1.25})
	if valueErr != nil {
		t.Fatal(valueErr)
	}
	feeds[weights.RopeFactors] = ropeValue
	outputs := []*tensor.Tensor{result.Output, result.Key, result.Value}
	want, err := reference.Execute(outputs, feeds)
	if err != nil {
		t.Fatal(err)
	}
	cuda, err := New(0)
	if err != nil {
		t.Fatal(err)
	}
	defer cuda.Close()
	got, err := cuda.Execute(context.Background(), outputs, feeds)
	if err != nil {
		t.Fatal(err)
	}
	compare(t, got[result.Output].Data, want[result.Output].Data, 1e-3)
	compare(t, got[result.Key].Data, want[result.Key].Data, 7e-5)
	compare(t, got[result.Value].Data, want[result.Value].Data, 5e-5)
}

func TestExecutorEXAOneMoEBlockMatchesReference(t *testing.T) {
	cudatest.Require(t)
	builder := tensor.NewBuilder()
	spec := model.Spec{CommonSpec: model.CommonSpec{Architecture: "exaone-moe", BlockCount: 4, EmbeddingLength: 8, FeedForwardLength: 16,

		RMSNormEpsilon: 1e-6}, AttentionSpec: model.AttentionSpec{HeadCount: 2, HeadCountKV: 1, KeyLength: 4, ValueLength: 4,
		RopeDimensionCount: 4, RopeFrequencyBase: 10000, RopeFrequencySWA: 500000,
		SlidingWindow: 128, SlidingPattern: 4}, MoESpec: model.MoESpec{ExpertCount: 4, ExpertUsedCount: 2, ExpertFeedForward: 6, ExpertWeightsScale: 1.5,
		SharedExpertFF: 12, ExpertGatingFunc: 2, ExpertWeightsNorm: true},
	}
	input := builder.Input("input", dtype.F32, tensor.MustShape(8, 3))
	weights := model.LayerGraphWeights{
		AttentionNorm:          builder.Input("attn_norm", dtype.F32, tensor.MustShape(8)),
		AttentionQ:             builder.Input("q", dtype.F32, tensor.MustShape(8, 8)),
		AttentionK:             builder.Input("k", dtype.F32, tensor.MustShape(8, 4)),
		AttentionV:             builder.Input("v", dtype.F32, tensor.MustShape(8, 4)),
		AttentionOutput:        builder.Input("attn_out", dtype.F32, tensor.MustShape(8, 8)),
		AttentionQNorm:         builder.Input("q_norm", dtype.F32, tensor.MustShape(4)),
		AttentionKNorm:         builder.Input("k_norm", dtype.F32, tensor.MustShape(4)),
		FeedForwardNorm:        builder.Input("ffn_norm", dtype.F32, tensor.MustShape(8)),
		FeedForwardRouter:      builder.Input("router", dtype.F32, tensor.MustShape(8, 4)),
		FeedForwardGateExperts: builder.Input("gate_exps", dtype.F32, tensor.MustShape(8, 6, 4)),
		FeedForwardUpExperts:   builder.Input("up_exps", dtype.F32, tensor.MustShape(8, 6, 4)),
		FeedForwardDownExperts: builder.Input("down_exps", dtype.F32, tensor.MustShape(6, 8, 4)),
		FeedForwardExpertBias:  builder.Input("expert_bias", dtype.F32, tensor.MustShape(4)),
		FeedForwardSharedGate:  builder.Input("shared_gate", dtype.F32, tensor.MustShape(8, 12)),
		FeedForwardSharedUp:    builder.Input("shared_up", dtype.F32, tensor.MustShape(8, 12)),
		FeedForwardSharedDown:  builder.Input("shared_down", dtype.F32, tensor.MustShape(12, 8)),
	}
	result, err := model.BuildDenseBlockCachedForLayer(builder, input, spec, weights, []uint32{0, 1, 2}, nil, nil, 0)
	if err != nil {
		t.Fatal(err)
	}
	feeds := map[*tensor.Tensor]reference.Value{input: patternedValue(input.Shape, 3, 0.2, 0)}
	for index, node := range []*tensor.Tensor{
		weights.AttentionQ, weights.AttentionK, weights.AttentionV, weights.AttentionOutput,
		weights.FeedForwardRouter, weights.FeedForwardGateExperts, weights.FeedForwardUpExperts,
		weights.FeedForwardDownExperts, weights.FeedForwardSharedGate,
		weights.FeedForwardSharedUp, weights.FeedForwardSharedDown,
	} {
		feeds[node] = patternedValue(node.Shape, index+11, 0.05, 0)
	}
	for index, node := range []*tensor.Tensor{
		weights.AttentionNorm, weights.AttentionQNorm, weights.AttentionKNorm, weights.FeedForwardNorm,
	} {
		feeds[node] = patternedValue(node.Shape, index+37, 0.03, 1)
	}
	feeds[weights.FeedForwardExpertBias] = patternedValue(weights.FeedForwardExpertBias.Shape, 47, 0.02, 0)
	outputs := []*tensor.Tensor{result.Output, result.Key, result.Value}
	want, err := reference.Execute(outputs, feeds)
	if err != nil {
		t.Fatal(err)
	}
	cuda, err := New(0)
	if err != nil {
		t.Fatal(err)
	}
	defer cuda.Close()
	got, err := cuda.Execute(context.Background(), outputs, feeds)
	if err != nil {
		t.Fatal(err)
	}
	compare(t, got[result.Output].Data, want[result.Output].Data, 1e-3)
	compare(t, got[result.Key].Data, want[result.Key].Data, 7e-5)
	compare(t, got[result.Value].Data, want[result.Value].Data, 5e-5)
}

func TestExecutorBailingMoEBlockMatchesReference(t *testing.T) {
	cudatest.Require(t)
	builder := tensor.NewBuilder()
	spec := model.Spec{CommonSpec: model.CommonSpec{Architecture: "bailingmoe", BlockCount: 1, EmbeddingLength: 8, FeedForwardLength: 16,

		RMSNormEpsilon: 1e-6}, AttentionSpec: model.AttentionSpec{HeadCount: 2, HeadCountKV: 1, KeyLength: 4, ValueLength: 4,
		RopeFrequencyBase: 10000}, MoESpec: model.MoESpec{ExpertCount: 4, ExpertUsedCount: 2, ExpertFeedForward: 6, ExpertWeightsScale: 1.25,
		SharedExpertCount: 2, SharedExpertFF: 12, ExpertWeightsNorm: true},
	}
	input := builder.Input("input", dtype.F32, tensor.MustShape(8, 3))
	weights := model.LayerGraphWeights{
		AttentionNorm:          builder.Input("attn_norm", dtype.F32, tensor.MustShape(8)),
		AttentionQ:             builder.Input("q", dtype.F32, tensor.MustShape(8, 8)),
		AttentionK:             builder.Input("k", dtype.F32, tensor.MustShape(8, 4)),
		AttentionV:             builder.Input("v", dtype.F32, tensor.MustShape(8, 4)),
		AttentionOutput:        builder.Input("attn_out", dtype.F32, tensor.MustShape(8, 8)),
		FeedForwardNorm:        builder.Input("ffn_norm", dtype.F32, tensor.MustShape(8)),
		FeedForwardRouter:      builder.Input("router", dtype.F32, tensor.MustShape(8, 4)),
		FeedForwardGateExperts: builder.Input("gate_exps", dtype.F32, tensor.MustShape(8, 6, 4)),
		FeedForwardUpExperts:   builder.Input("up_exps", dtype.F32, tensor.MustShape(8, 6, 4)),
		FeedForwardDownExperts: builder.Input("down_exps", dtype.F32, tensor.MustShape(6, 8, 4)),
		FeedForwardSharedGate:  builder.Input("shared_gate", dtype.F32, tensor.MustShape(8, 12)),
		FeedForwardSharedUp:    builder.Input("shared_up", dtype.F32, tensor.MustShape(8, 12)),
		FeedForwardSharedDown:  builder.Input("shared_down", dtype.F32, tensor.MustShape(12, 8)),
	}
	result, err := model.BuildDenseBlockCachedForLayer(builder, input, spec, weights, []uint32{0, 1, 2}, nil, nil, 0)
	if err != nil {
		t.Fatal(err)
	}
	feeds := map[*tensor.Tensor]reference.Value{input: patternedValue(input.Shape, 3, 0.2, 0)}
	for index, node := range []*tensor.Tensor{
		weights.AttentionQ, weights.AttentionK, weights.AttentionV, weights.AttentionOutput,
		weights.FeedForwardRouter, weights.FeedForwardGateExperts, weights.FeedForwardUpExperts,
		weights.FeedForwardDownExperts, weights.FeedForwardSharedGate,
		weights.FeedForwardSharedUp, weights.FeedForwardSharedDown,
	} {
		feeds[node] = patternedValue(node.Shape, index+11, 0.05, 0)
	}
	for index, node := range []*tensor.Tensor{weights.AttentionNorm, weights.FeedForwardNorm} {
		feeds[node] = patternedValue(node.Shape, index+37, 0.03, 1)
	}
	outputs := []*tensor.Tensor{result.Output, result.Key, result.Value}
	want, err := reference.Execute(outputs, feeds)
	if err != nil {
		t.Fatal(err)
	}
	cuda, err := New(0)
	if err != nil {
		t.Fatal(err)
	}
	defer cuda.Close()
	got, err := cuda.Execute(context.Background(), outputs, feeds)
	if err != nil {
		t.Fatal(err)
	}
	compare(t, got[result.Output].Data, want[result.Output].Data, 1e-3)
	compare(t, got[result.Key].Data, want[result.Key].Data, 7e-5)
	compare(t, got[result.Value].Data, want[result.Value].Data, 5e-5)
}

func TestExecutorDeepSeekMoEBlockMatchesReference(t *testing.T) {
	cudatest.Require(t)
	builder := tensor.NewBuilder()
	spec := model.Spec{CommonSpec: model.CommonSpec{Architecture: "deepseek", BlockCount: 1, EmbeddingLength: 8, FeedForwardLength: 16,

		RMSNormEpsilon: 1e-6}, AttentionSpec: model.AttentionSpec{HeadCount: 2, HeadCountKV: 1, KeyLength: 4, ValueLength: 4,
		RopeFrequencyBase: 10000}, MoESpec: model.MoESpec{ExpertCount: 4, ExpertUsedCount: 2, ExpertFeedForward: 6, ExpertWeightsScale: 1.3,
		SharedExpertCount: 2, SharedExpertFF: 12},
	}
	input := builder.Input("input", dtype.F32, tensor.MustShape(8, 3))
	weights := model.LayerGraphWeights{
		AttentionNorm:          builder.Input("attn_norm", dtype.F32, tensor.MustShape(8)),
		AttentionQ:             builder.Input("q", dtype.F32, tensor.MustShape(8, 8)),
		AttentionK:             builder.Input("k", dtype.F32, tensor.MustShape(8, 4)),
		AttentionV:             builder.Input("v", dtype.F32, tensor.MustShape(8, 4)),
		AttentionOutput:        builder.Input("attn_out", dtype.F32, tensor.MustShape(8, 8)),
		FeedForwardNorm:        builder.Input("ffn_norm", dtype.F32, tensor.MustShape(8)),
		FeedForwardRouter:      builder.Input("router", dtype.F32, tensor.MustShape(8, 4)),
		FeedForwardGateExperts: builder.Input("gate_exps", dtype.F32, tensor.MustShape(8, 6, 4)),
		FeedForwardUpExperts:   builder.Input("up_exps", dtype.F32, tensor.MustShape(8, 6, 4)),
		FeedForwardDownExperts: builder.Input("down_exps", dtype.F32, tensor.MustShape(6, 8, 4)),
		FeedForwardSharedGate:  builder.Input("shared_gate", dtype.F32, tensor.MustShape(8, 12)),
		FeedForwardSharedUp:    builder.Input("shared_up", dtype.F32, tensor.MustShape(8, 12)),
		FeedForwardSharedDown:  builder.Input("shared_down", dtype.F32, tensor.MustShape(12, 8)),
	}
	result, err := model.BuildDenseBlockCachedForLayer(builder, input, spec, weights, []uint32{0, 1, 2}, nil, nil, 0)
	if err != nil {
		t.Fatal(err)
	}
	feeds := map[*tensor.Tensor]reference.Value{input: patternedValue(input.Shape, 3, 0.2, 0)}
	for index, node := range []*tensor.Tensor{
		weights.AttentionQ, weights.AttentionK, weights.AttentionV, weights.AttentionOutput,
		weights.FeedForwardRouter, weights.FeedForwardGateExperts, weights.FeedForwardUpExperts,
		weights.FeedForwardDownExperts, weights.FeedForwardSharedGate,
		weights.FeedForwardSharedUp, weights.FeedForwardSharedDown,
	} {
		feeds[node] = patternedValue(node.Shape, index+11, 0.05, 0)
	}
	for index, node := range []*tensor.Tensor{weights.AttentionNorm, weights.FeedForwardNorm} {
		feeds[node] = patternedValue(node.Shape, index+37, 0.03, 1)
	}
	outputs := []*tensor.Tensor{result.Output, result.Key, result.Value}
	want, err := reference.Execute(outputs, feeds)
	if err != nil {
		t.Fatal(err)
	}
	cuda, err := New(0)
	if err != nil {
		t.Fatal(err)
	}
	defer cuda.Close()
	got, err := cuda.Execute(context.Background(), outputs, feeds)
	if err != nil {
		t.Fatal(err)
	}
	compare(t, got[result.Output].Data, want[result.Output].Data, 1e-3)
	compare(t, got[result.Key].Data, want[result.Key].Data, 7e-5)
	compare(t, got[result.Value].Data, want[result.Value].Data, 5e-5)
}

func TestExecutorGraniteMoEUngatedBlockMatchesReference(t *testing.T) {
	cudatest.Require(t)
	for _, architecture := range []string{"granitemoe", "granite"} {
		t.Run(architecture, func(t *testing.T) {
			builder := tensor.NewBuilder()
			spec := model.Spec{CommonSpec: model.CommonSpec{Architecture: architecture, BlockCount: 1, EmbeddingLength: 8, FeedForwardLength: 6,

				ResidualScale: 0.5,

				RMSNormEpsilon: 1e-6}, AttentionSpec: model.AttentionSpec{HeadCount: 2, HeadCountKV: 1, KeyLength: 4, ValueLength: 4,
				RopeFrequencyBase: 10000}, MoESpec: model.MoESpec{ExpertCount: 4, ExpertUsedCount: 2, ExpertFeedForward: 6, ExpertWeightsScale: 1,
				ExpertWeightsNorm: true, SharedExpertFF: 5},
			}
			input := builder.Input("input", dtype.F32, tensor.MustShape(8, 3))
			weights := model.LayerGraphWeights{
				AttentionNorm:          builder.Input("attn_norm", dtype.F32, tensor.MustShape(8)),
				AttentionQ:             builder.Input("q", dtype.F32, tensor.MustShape(8, 8)),
				AttentionK:             builder.Input("k", dtype.F32, tensor.MustShape(8, 4)),
				AttentionV:             builder.Input("v", dtype.F32, tensor.MustShape(8, 4)),
				AttentionOutput:        builder.Input("attn_out", dtype.F32, tensor.MustShape(8, 8)),
				FeedForwardNorm:        builder.Input("ffn_norm", dtype.F32, tensor.MustShape(8)),
				FeedForwardRouter:      builder.Input("router", dtype.F32, tensor.MustShape(8, 4)),
				FeedForwardUpExperts:   builder.Input("up_exps", dtype.F32, tensor.MustShape(8, 6, 4)),
				FeedForwardDownExperts: builder.Input("down_exps", dtype.F32, tensor.MustShape(6, 8, 4)),
				FeedForwardSharedGate:  builder.Input("shared_gate", dtype.F32, tensor.MustShape(8, 5)),
				FeedForwardSharedUp:    builder.Input("shared_up", dtype.F32, tensor.MustShape(8, 5)),
				FeedForwardSharedDown:  builder.Input("shared_down", dtype.F32, tensor.MustShape(5, 8)),
			}
			layerInput := input
			var deepstack *tensor.Tensor
			if architecture == "granite" {
				deepstack = builder.Input("deepstack_input", dtype.F32, input.Shape)
				layerInput = builder.Add(layerInput, deepstack)
			}
			result, err := model.BuildDenseBlockCachedForLayer(builder, layerInput, spec, weights, []uint32{0, 1, 2}, nil, nil, 0)
			if err != nil {
				t.Fatal(err)
			}
			feeds := map[*tensor.Tensor]reference.Value{input: patternedValue(input.Shape, 3, 0.2, 0)}
			if deepstack != nil {
				feeds[deepstack] = patternedValue(deepstack.Shape, 7, 0.04, 0)
			}
			for index, node := range []*tensor.Tensor{
				weights.AttentionQ, weights.AttentionK, weights.AttentionV, weights.AttentionOutput,
				weights.FeedForwardRouter, weights.FeedForwardUpExperts, weights.FeedForwardDownExperts,
				weights.FeedForwardSharedGate, weights.FeedForwardSharedUp, weights.FeedForwardSharedDown,
			} {
				feeds[node] = patternedValue(node.Shape, index+11, 0.05, 0)
			}
			for index, node := range []*tensor.Tensor{weights.AttentionNorm, weights.FeedForwardNorm} {
				feeds[node] = patternedValue(node.Shape, index+37, 0.03, 1)
			}
			outputs := []*tensor.Tensor{result.Output, result.Key, result.Value}
			want, err := reference.Execute(outputs, feeds)
			if err != nil {
				t.Fatal(err)
			}
			cuda, err := New(0)
			if err != nil {
				t.Fatal(err)
			}
			defer cuda.Close()
			got, err := cuda.Execute(context.Background(), outputs, feeds)
			if err != nil {
				t.Fatal(err)
			}
			compare(t, got[result.Output].Data, want[result.Output].Data, 1e-3)
			compare(t, got[result.Key].Data, want[result.Key].Data, 7e-5)
			compare(t, got[result.Value].Data, want[result.Value].Data, 5e-5)
		})
	}
}

func TestExecutorSmallThinkerBlockMatchesReference(t *testing.T) {
	cudatest.Require(t)
	builder := tensor.NewBuilder()
	spec := model.Spec{CommonSpec: model.CommonSpec{Architecture: "smallthinker", BlockCount: 4, EmbeddingLength: 8, FeedForwardLength: 6,

		RMSNormEpsilon: 1e-6}, AttentionSpec: model.AttentionSpec{HeadCount: 2, HeadCountKV: 1, KeyLength: 4, ValueLength: 4,
		RopeDimensionCount: 4, RopeFrequencyBase: 10000, RopeFrequencySWA: 20000,
		SlidingWindow: 128, SlidingPattern: 4, NoRopeLayerStep: 4}, MoESpec: model.MoESpec{ExpertCount: 4, ExpertUsedCount: 2, ExpertFeedForward: 6,
		ExpertWeightsScale: 1.25, ExpertWeightsNorm: true, ExpertGatingFunc: 2},
	}
	input := builder.Input("input", dtype.F32, tensor.MustShape(8, 3))
	weights := model.LayerGraphWeights{
		AttentionNorm:          builder.Input("attn_norm", dtype.F32, tensor.MustShape(8)),
		AttentionQKV:           builder.Input("qkv", dtype.F32, tensor.MustShape(8, 16)),
		AttentionOutput:        builder.Input("attn_out", dtype.F32, tensor.MustShape(8, 8)),
		FeedForwardNorm:        builder.Input("ffn_norm", dtype.F32, tensor.MustShape(8)),
		FeedForwardRouter:      builder.Input("router", dtype.F32, tensor.MustShape(8, 4)),
		FeedForwardGateExperts: builder.Input("gate_exps", dtype.F32, tensor.MustShape(8, 6, 4)),
		FeedForwardUpExperts:   builder.Input("up_exps", dtype.F32, tensor.MustShape(8, 6, 4)),
		FeedForwardDownExperts: builder.Input("down_exps", dtype.F32, tensor.MustShape(6, 8, 4)),
	}
	result, err := model.BuildDenseBlockCachedForLayer(
		builder, input, spec, weights, []uint32{0, 1, 2}, nil, nil, 1,
	)
	if err != nil {
		t.Fatal(err)
	}
	feeds := map[*tensor.Tensor]reference.Value{input: patternedValue(input.Shape, 3, 0.2, 0)}
	for index, node := range []*tensor.Tensor{
		weights.AttentionQKV, weights.AttentionOutput, weights.FeedForwardRouter,
		weights.FeedForwardGateExperts, weights.FeedForwardUpExperts, weights.FeedForwardDownExperts,
	} {
		feeds[node] = patternedValue(node.Shape, index+11, 0.05, 0)
	}
	for index, node := range []*tensor.Tensor{weights.AttentionNorm, weights.FeedForwardNorm} {
		feeds[node] = patternedValue(node.Shape, index+37, 0.03, 1)
	}
	outputs := []*tensor.Tensor{result.Output, result.Key, result.Value}
	want, err := reference.Execute(outputs, feeds)
	if err != nil {
		t.Fatal(err)
	}
	cuda, err := New(0)
	if err != nil {
		t.Fatal(err)
	}
	defer cuda.Close()
	got, err := cuda.Execute(context.Background(), outputs, feeds)
	if err != nil {
		t.Fatal(err)
	}
	compare(t, got[result.Output].Data, want[result.Output].Data, 1e-3)
	compare(t, got[result.Key].Data, want[result.Key].Data, 7e-5)
	compare(t, got[result.Value].Data, want[result.Value].Data, 5e-5)
}

func TestExecutorDOTS1BlockMatchesReference(t *testing.T) {
	cudatest.Require(t)
	builder := tensor.NewBuilder()
	spec := model.Spec{CommonSpec: model.CommonSpec{Architecture: "dots1", BlockCount: 2,
		EmbeddingLength: 8, FeedForwardLength: 12,

		RMSNormEpsilon: 1e-6}, AttentionSpec: model.AttentionSpec{HeadCount: 2, HeadCountKV: 2, KeyLength: 4, ValueLength: 4,
		RopeFrequencyBase: 10000}, MoESpec: model.MoESpec{LeadingDenseBlocks: 1,
		ExpertCount: 4, ExpertUsedCount: 2,
		ExpertFeedForward: 6, SharedExpertCount: 2, SharedExpertFF: 12,
		ExpertWeightsScale: 1.25, ExpertWeightsNorm: true, ExpertGatingFunc: 2},
	}
	input := builder.Input("input", dtype.F32, tensor.MustShape(8, 3))
	weights := model.LayerGraphWeights{
		AttentionNorm:          builder.Input("attn_norm", dtype.F32, tensor.MustShape(8)),
		AttentionQKV:           builder.Input("qkv", dtype.F32, tensor.MustShape(8, 24)),
		AttentionOutput:        builder.Input("attn_out", dtype.F32, tensor.MustShape(8, 8)),
		AttentionQNorm:         builder.Input("q_norm", dtype.F32, tensor.MustShape(4)),
		AttentionKNorm:         builder.Input("k_norm", dtype.F32, tensor.MustShape(4)),
		FeedForwardNorm:        builder.Input("ffn_norm", dtype.F32, tensor.MustShape(8)),
		FeedForwardRouter:      builder.Input("router", dtype.F32, tensor.MustShape(8, 4)),
		FeedForwardGateExperts: builder.Input("gate_exps", dtype.F32, tensor.MustShape(8, 6, 4)),
		FeedForwardUpExperts:   builder.Input("up_exps", dtype.F32, tensor.MustShape(8, 6, 4)),
		FeedForwardDownExperts: builder.Input("down_exps", dtype.F32, tensor.MustShape(6, 8, 4)),
		FeedForwardExpertBias:  builder.Input("expert_bias", dtype.F32, tensor.MustShape(4)),
		FeedForwardSharedGate:  builder.Input("shared_gate", dtype.F32, tensor.MustShape(8, 12)),
		FeedForwardSharedUp:    builder.Input("shared_up", dtype.F32, tensor.MustShape(8, 12)),
		FeedForwardSharedDown:  builder.Input("shared_down", dtype.F32, tensor.MustShape(12, 8)),
	}
	result, err := model.BuildDenseBlockCachedForLayer(
		builder, input, spec, weights, []uint32{0, 1, 2}, nil, nil, 1,
	)
	if err != nil {
		t.Fatal(err)
	}
	feeds := map[*tensor.Tensor]reference.Value{input: patternedValue(input.Shape, 3, 0.2, 0)}
	for index, node := range []*tensor.Tensor{
		weights.AttentionQKV, weights.AttentionOutput, weights.FeedForwardRouter,
		weights.FeedForwardGateExperts, weights.FeedForwardUpExperts,
		weights.FeedForwardDownExperts, weights.FeedForwardExpertBias,
		weights.FeedForwardSharedGate, weights.FeedForwardSharedUp,
		weights.FeedForwardSharedDown,
	} {
		feeds[node] = patternedValue(node.Shape, index+11, 0.05, 0)
	}
	for index, node := range []*tensor.Tensor{
		weights.AttentionNorm, weights.AttentionQNorm, weights.AttentionKNorm,
		weights.FeedForwardNorm,
	} {
		feeds[node] = patternedValue(node.Shape, index+37, 0.03, 1)
	}
	outputs := []*tensor.Tensor{result.Output, result.Key, result.Value}
	want, err := reference.Execute(outputs, feeds)
	if err != nil {
		t.Fatal(err)
	}
	cuda, err := New(0)
	if err != nil {
		t.Fatal(err)
	}
	defer cuda.Close()
	got, err := cuda.Execute(context.Background(), outputs, feeds)
	if err != nil {
		t.Fatal(err)
	}
	compare(t, got[result.Output].Data, want[result.Output].Data, 1e-3)
	compare(t, got[result.Key].Data, want[result.Key].Data, 7e-5)
	compare(t, got[result.Value].Data, want[result.Value].Data, 5e-5)
}

func TestExecutorMiniMaxM2BlockMatchesReference(t *testing.T) {
	cudatest.Require(t)
	builder := tensor.NewBuilder()
	spec := model.Spec{CommonSpec: model.CommonSpec{Architecture: "minimax-m2", BlockCount: 1, EmbeddingLength: 8,
		FeedForwardLength: 6,

		RMSNormEpsilon: 1e-6}, AttentionSpec: model.AttentionSpec{HeadCount: 2, HeadCountKV: 1, KeyLength: 4, ValueLength: 4,
		RopeDimensionCount: 2, RopeFrequencyBase: 10000}, MoESpec: model.MoESpec{ExpertCount: 4, ExpertUsedCount: 2,
		ExpertFeedForward: 6, ExpertWeightsScale: 1.25, ExpertGatingFunc: 2},
	}
	input := builder.Input("input", dtype.F32, tensor.MustShape(8, 3))
	weights := model.LayerGraphWeights{
		AttentionNorm:          builder.Input("attn_norm", dtype.F32, tensor.MustShape(8)),
		AttentionQKV:           builder.Input("qkv", dtype.F32, tensor.MustShape(8, 16)),
		AttentionOutput:        builder.Input("attn_out", dtype.F32, tensor.MustShape(8, 8)),
		AttentionQNorm:         builder.Input("q_norm", dtype.F32, tensor.MustShape(8)),
		AttentionKNorm:         builder.Input("k_norm", dtype.F32, tensor.MustShape(4)),
		FeedForwardNorm:        builder.Input("ffn_norm", dtype.F32, tensor.MustShape(8)),
		FeedForwardRouter:      builder.Input("router", dtype.F32, tensor.MustShape(8, 4)),
		FeedForwardGateExperts: builder.Input("gate_exps", dtype.F32, tensor.MustShape(8, 6, 4)),
		FeedForwardUpExperts:   builder.Input("up_exps", dtype.F32, tensor.MustShape(8, 6, 4)),
		FeedForwardDownExperts: builder.Input("down_exps", dtype.F32, tensor.MustShape(6, 8, 4)),
		FeedForwardExpertBias:  builder.Input("expert_bias", dtype.F32, tensor.MustShape(4)),
	}
	result, err := model.BuildDenseBlockCachedForLayer(
		builder, input, spec, weights, []uint32{0, 1, 2}, nil, nil, 0,
	)
	if err != nil {
		t.Fatal(err)
	}
	feeds := map[*tensor.Tensor]reference.Value{input: patternedValue(input.Shape, 3, 0.2, 0)}
	for index, node := range []*tensor.Tensor{
		weights.AttentionQKV, weights.AttentionOutput, weights.FeedForwardRouter,
		weights.FeedForwardGateExperts, weights.FeedForwardUpExperts,
		weights.FeedForwardDownExperts, weights.FeedForwardExpertBias,
	} {
		feeds[node] = patternedValue(node.Shape, index+11, 0.05, 0)
	}
	for index, node := range []*tensor.Tensor{
		weights.AttentionNorm, weights.AttentionQNorm, weights.AttentionKNorm,
		weights.FeedForwardNorm,
	} {
		feeds[node] = patternedValue(node.Shape, index+37, 0.03, 1)
	}
	outputs := []*tensor.Tensor{result.Output, result.Key, result.Value}
	want, err := reference.Execute(outputs, feeds)
	if err != nil {
		t.Fatal(err)
	}
	cuda, err := New(0)
	if err != nil {
		t.Fatal(err)
	}
	defer cuda.Close()
	got, err := cuda.Execute(context.Background(), outputs, feeds)
	if err != nil {
		t.Fatal(err)
	}
	compare(t, got[result.Output].Data, want[result.Output].Data, 1e-3)
	compare(t, got[result.Key].Data, want[result.Key].Data, 7e-5)
	compare(t, got[result.Value].Data, want[result.Value].Data, 5e-5)
}

func TestExecutorGrokBlockMatchesReference(t *testing.T) {
	cudatest.Require(t)
	for _, test := range []struct {
		name  string
		gated bool
		dense bool
	}{
		{name: "ungated"},
		{name: "gated-dense", gated: true, dense: true},
	} {
		t.Run(test.name, func(t *testing.T) {
			builder := tensor.NewBuilder()
			spec := model.Spec{CommonSpec: model.CommonSpec{Architecture: "grok", BlockCount: 1, EmbeddingLength: 8, FeedForwardLength: 12,

				RMSNormEpsilon: 1e-6}, AttentionSpec: model.AttentionSpec{HeadCount: 2, HeadCountKV: 1, KeyLength: 4, ValueLength: 4,
				RopeDimensionCount: 4, RopeFrequencyBase: 10000, RopeScalingType: "yarn",
				RopeScalingFactor: 4, OriginalContextLength: 2048, YaRNExtFactor: 1,
				YaRNAttentionFactor: 1.25, YaRNBetaFast: 8, YaRNBetaSlow: 1,
				AttentionScale: 0.25, AttentionSoftcap: 30}, MoESpec: model.MoESpec{ExpertCount: 4, ExpertUsedCount: 2, ExpertFeedForward: 6,
				ExpertWeightsScale: 1.25, ExpertWeightsNorm: true},
			}
			input := builder.Input("input", dtype.F32, tensor.MustShape(8, 3))
			weights := model.LayerGraphWeights{
				AttentionNorm:          builder.Input("attn_norm", dtype.F32, tensor.MustShape(8)),
				AttentionQKV:           builder.Input("qkv", dtype.F32, tensor.MustShape(8, 16)),
				AttentionOutput:        builder.Input("attn_out", dtype.F32, tensor.MustShape(8, 8)),
				AttentionPostNorm:      builder.Input("attn_post_norm", dtype.F32, tensor.MustShape(8)),
				FeedForwardNorm:        builder.Input("ffn_norm", dtype.F32, tensor.MustShape(8)),
				FeedForwardRouter:      builder.Input("router", dtype.F32, tensor.MustShape(8, 4)),
				FeedForwardUpExperts:   builder.Input("up_exps", dtype.F32, tensor.MustShape(8, 6, 4)),
				FeedForwardDownExperts: builder.Input("down_exps", dtype.F32, tensor.MustShape(6, 8, 4)),
				FeedForwardPostNorm:    builder.Input("ffn_post_norm", dtype.F32, tensor.MustShape(8)),
			}
			if test.gated {
				weights.FeedForwardGateExperts = builder.Input(
					"gate_exps", dtype.F32, tensor.MustShape(8, 6, 4),
				)
			}
			if test.dense {
				weights.FeedForwardGate = builder.Input("dense_gate", dtype.F32, tensor.MustShape(8, 12))
				weights.FeedForwardUp = builder.Input("dense_up", dtype.F32, tensor.MustShape(8, 12))
				weights.FeedForwardDown = builder.Input("dense_down", dtype.F32, tensor.MustShape(12, 8))
			}
			result, err := model.BuildDenseBlockCachedForLayer(
				builder, input, spec, weights, []uint32{0, 1, 2}, nil, nil, 0,
			)
			if err != nil {
				t.Fatal(err)
			}
			feeds := map[*tensor.Tensor]reference.Value{
				input: patternedValue(input.Shape, 3, 0.2, 0),
			}
			matrices := []*tensor.Tensor{
				weights.AttentionQKV, weights.AttentionOutput, weights.FeedForwardRouter,
				weights.FeedForwardGateExperts, weights.FeedForwardUpExperts,
				weights.FeedForwardDownExperts, weights.FeedForwardGate,
				weights.FeedForwardUp, weights.FeedForwardDown,
			}
			for index, node := range matrices {
				if node != nil {
					feeds[node] = patternedValue(node.Shape, index+11, 0.05, 0)
				}
			}
			for index, node := range []*tensor.Tensor{
				weights.AttentionNorm, weights.AttentionPostNorm, weights.FeedForwardNorm,
				weights.FeedForwardPostNorm,
			} {
				feeds[node] = patternedValue(node.Shape, index+37, 0.03, 1)
			}
			outputs := []*tensor.Tensor{result.Output, result.Key, result.Value}
			want, err := reference.Execute(outputs, feeds)
			if err != nil {
				t.Fatal(err)
			}
			cuda, err := New(0)
			if err != nil {
				t.Fatal(err)
			}
			defer cuda.Close()
			got, err := cuda.Execute(context.Background(), outputs, feeds)
			if err != nil {
				t.Fatal(err)
			}
			compare(t, got[result.Output].Data, want[result.Output].Data, 2e-3)
			compare(t, got[result.Key].Data, want[result.Key].Data, 7e-5)
			compare(t, got[result.Value].Data, want[result.Value].Data, 5e-5)
		})
	}
}

func TestExecutorMellumBlockMatchesReference(t *testing.T) {
	cudatest.Require(t)
	builder := tensor.NewBuilder()
	spec := model.Spec{CommonSpec: model.CommonSpec{Architecture: "mellum", BlockCount: 4, EmbeddingLength: 8, FeedForwardLength: 12,

		RMSNormEpsilon: 1e-6}, AttentionSpec: model.AttentionSpec{HeadCount: 2, HeadCountKV: 1, KeyLength: 4, ValueLength: 4,
		RopeDimensionCount: 4, RopeFrequencyBase: 10000, RopeScalingType: "yarn",
		RopeScalingFactor: 4, OriginalContextLength: 2048, YaRNExtFactor: 1,
		YaRNAttentionFactor: 1.25, YaRNBetaFast: 32, YaRNBetaSlow: 1,
		SlidingWindow: 128, SlidingPattern: 4}, MoESpec: model.MoESpec{ExpertCount: 4, ExpertUsedCount: 2, ExpertFeedForward: 6,
		ExpertWeightsScale: 1, ExpertWeightsNorm: true},
	}
	input := builder.Input("input", dtype.F32, tensor.MustShape(8, 3))
	weights := model.LayerGraphWeights{
		AttentionNorm:          builder.Input("attn_norm", dtype.F32, tensor.MustShape(8)),
		AttentionQ:             builder.Input("q", dtype.F32, tensor.MustShape(8, 8)),
		AttentionK:             builder.Input("k", dtype.F32, tensor.MustShape(8, 4)),
		AttentionV:             builder.Input("v", dtype.F32, tensor.MustShape(8, 4)),
		AttentionOutput:        builder.Input("attn_out", dtype.F32, tensor.MustShape(8, 8)),
		AttentionQNorm:         builder.Input("q_norm", dtype.F32, tensor.MustShape(4)),
		AttentionKNorm:         builder.Input("k_norm", dtype.F32, tensor.MustShape(4)),
		FeedForwardNorm:        builder.Input("ffn_norm", dtype.F32, tensor.MustShape(8)),
		FeedForwardRouter:      builder.Input("router", dtype.F32, tensor.MustShape(8, 4)),
		FeedForwardGateExperts: builder.Input("gate_exps", dtype.F32, tensor.MustShape(8, 6, 4)),
		FeedForwardUpExperts:   builder.Input("up_exps", dtype.F32, tensor.MustShape(8, 6, 4)),
		FeedForwardDownExperts: builder.Input("down_exps", dtype.F32, tensor.MustShape(6, 8, 4)),
	}
	result, err := model.BuildDenseBlockCachedForLayer(
		builder, input, spec, weights, []uint32{0, 1, 2}, nil, nil, 3,
	)
	if err != nil {
		t.Fatal(err)
	}
	feeds := map[*tensor.Tensor]reference.Value{input: patternedValue(input.Shape, 3, 0.2, 0)}
	for index, node := range []*tensor.Tensor{
		weights.AttentionQ, weights.AttentionK, weights.AttentionV, weights.AttentionOutput,
		weights.FeedForwardRouter, weights.FeedForwardGateExperts,
		weights.FeedForwardUpExperts, weights.FeedForwardDownExperts,
	} {
		feeds[node] = patternedValue(node.Shape, index+11, 0.05, 0)
	}
	for index, node := range []*tensor.Tensor{
		weights.AttentionNorm, weights.AttentionQNorm, weights.AttentionKNorm, weights.FeedForwardNorm,
	} {
		feeds[node] = patternedValue(node.Shape, index+37, 0.03, 1)
	}
	outputs := []*tensor.Tensor{result.Output, result.Key, result.Value}
	want, err := reference.Execute(outputs, feeds)
	if err != nil {
		t.Fatal(err)
	}
	cuda, err := New(0)
	if err != nil {
		t.Fatal(err)
	}
	defer cuda.Close()
	got, err := cuda.Execute(context.Background(), outputs, feeds)
	if err != nil {
		t.Fatal(err)
	}
	compare(t, got[result.Output].Data, want[result.Output].Data, 1e-3)
	compare(t, got[result.Key].Data, want[result.Key].Data, 7e-5)
	compare(t, got[result.Value].Data, want[result.Value].Data, 5e-5)
}

func TestExecutorQwenBlockMatchesReference(t *testing.T) {
	cudatest.Require(t)
	b := tensor.NewBuilder()
	s := model.Spec{CommonSpec: model.CommonSpec{Architecture: "qwen", EmbeddingLength: 8, FeedForwardLength: 12,

		RMSNormEpsilon: 1e-6}, AttentionSpec: model.AttentionSpec{HeadCount: 2, HeadCountKV: 2, KeyLength: 4, ValueLength: 4,
		RopeDimensionCount: 4, RopeFrequencyBase: 10000}}
	in := b.Input("input", dtype.F32, tensor.MustShape(8, 3))
	w := model.LayerGraphWeights{
		AttentionNorm: b.Input("an", dtype.F32, tensor.MustShape(8)), AttentionQKV: b.Input("qkv", dtype.F32, tensor.MustShape(8, 24)), AttentionQKVBias: b.Input("qkvb", dtype.F32, tensor.MustShape(24)), AttentionOutput: b.Input("o", dtype.F32, tensor.MustShape(8, 8)), FeedForwardNorm: b.Input("fn", dtype.F32, tensor.MustShape(8)), FeedForwardGate: b.Input("fg", dtype.F32, tensor.MustShape(8, 12)), FeedForwardUp: b.Input("fu", dtype.F32, tensor.MustShape(8, 12)), FeedForwardDown: b.Input("fd", dtype.F32, tensor.MustShape(12, 8)),
	}
	r, err := model.BuildDenseBlockCachedForLayer(b, in, s, w, []uint32{0, 1, 2}, nil, nil, 0)
	if err != nil {
		t.Fatal(err)
	}
	feeds := map[*tensor.Tensor]reference.Value{in: patternedValue(in.Shape, 3, 0.2, 0)}
	for i, n := range []*tensor.Tensor{w.AttentionQKV, w.AttentionQKVBias, w.AttentionOutput, w.FeedForwardGate, w.FeedForwardUp, w.FeedForwardDown} {
		feeds[n] = patternedValue(n.Shape, i+11, 0.05, 0)
	}
	for i, n := range []*tensor.Tensor{w.AttentionNorm, w.FeedForwardNorm} {
		feeds[n] = patternedValue(n.Shape, i+31, 0.03, 1)
	}
	outputs := []*tensor.Tensor{r.Output, r.Key, r.Value}
	want, err := reference.Execute(outputs, feeds)
	if err != nil {
		t.Fatal(err)
	}
	cuda, err := New(0)
	if err != nil {
		t.Fatal(err)
	}
	defer cuda.Close()
	got, err := cuda.Execute(context.Background(), outputs, feeds)
	if err != nil {
		t.Fatal(err)
	}
	compare(t, got[r.Output].Data, want[r.Output].Data, 1e-3)
	compare(t, got[r.Key].Data, want[r.Key].Data, 7e-5)
	compare(t, got[r.Value].Data, want[r.Value].Data, 5e-5)
}

func TestExecutorChatGLMBlockMatchesReference(t *testing.T) {
	cudatest.Require(t)
	b := tensor.NewBuilder()
	s := model.Spec{CommonSpec: model.CommonSpec{Architecture: "chatglm", EmbeddingLength: 8, FeedForwardLength: 12, RMSNormEpsilon: 1e-6}, AttentionSpec: model.AttentionSpec{HeadCount: 2, HeadCountKV: 1, KeyLength: 4, ValueLength: 4, RopeDimensionCount: 4, RopeFrequencyBase: 10000}}
	in := b.Input("input", dtype.F32, tensor.MustShape(8, 3))
	w := model.LayerGraphWeights{AttentionNorm: b.Input("an", dtype.F32, tensor.MustShape(8)), AttentionQKV: b.Input("qkv", dtype.F32, tensor.MustShape(8, 16)), AttentionOutput: b.Input("o", dtype.F32, tensor.MustShape(8, 8)), FeedForwardNorm: b.Input("fn", dtype.F32, tensor.MustShape(8)), FeedForwardUp: b.Input("fu", dtype.F32, tensor.MustShape(8, 24)), FeedForwardDown: b.Input("fd", dtype.F32, tensor.MustShape(12, 8))}
	r, err := model.BuildDenseBlockCachedForLayer(b, in, s, w, []uint32{0, 1, 2}, nil, nil, 0)
	if err != nil {
		t.Fatal(err)
	}
	feeds := map[*tensor.Tensor]reference.Value{in: patternedValue(in.Shape, 3, 0.2, 0)}
	for i, n := range []*tensor.Tensor{w.AttentionQKV, w.AttentionOutput, w.FeedForwardUp, w.FeedForwardDown} {
		feeds[n] = patternedValue(n.Shape, i+11, 0.05, 0)
	}
	for i, n := range []*tensor.Tensor{w.AttentionNorm, w.FeedForwardNorm} {
		feeds[n] = patternedValue(n.Shape, i+31, 0.03, 1)
	}
	outputs := []*tensor.Tensor{r.Output, r.Key, r.Value}
	want, err := reference.Execute(outputs, feeds)
	if err != nil {
		t.Fatal(err)
	}
	cuda, err := New(0)
	if err != nil {
		t.Fatal(err)
	}
	defer cuda.Close()
	got, err := cuda.Execute(context.Background(), outputs, feeds)
	if err != nil {
		t.Fatal(err)
	}
	compare(t, got[r.Output].Data, want[r.Output].Data, 1e-3)
	compare(t, got[r.Key].Data, want[r.Key].Data, 7e-5)
	compare(t, got[r.Value].Data, want[r.Value].Data, 5e-5)
}

func TestExecutorHunyuanDenseBlockMatchesReference(t *testing.T) {
	testExecutorHunyuanBlockMatchesReference(t, "hunyuan-dense", [4]int32{1, 1, 0, 0})
}

func TestExecutorHunyuanVLBlockMatchesReference(t *testing.T) {
	testExecutorHunyuanBlockMatchesReference(t, "hunyuan_vl", [4]int32{1, 1, 0, 0})
}

func testExecutorHunyuanBlockMatchesReference(t *testing.T, architecture string, sections [4]int32) {
	cudatest.Require(t)
	b := tensor.NewBuilder()
	s := model.Spec{CommonSpec: model.CommonSpec{Architecture: architecture, EmbeddingLength: 8, FeedForwardLength: 12, RMSNormEpsilon: 1e-6}, AttentionSpec: model.AttentionSpec{HeadCount: 2, HeadCountKV: 1, KeyLength: 4, ValueLength: 4, RopeDimensionCount: 4, RopeSections: sections, RopeFrequencyBase: 40000}}
	in := b.Input("input", dtype.F32, tensor.MustShape(8, 3))
	w := model.LayerGraphWeights{AttentionNorm: b.Input("an", dtype.F32, tensor.MustShape(8)), AttentionQKV: b.Input("qkv", dtype.F32, tensor.MustShape(8, 16)), AttentionOutput: b.Input("o", dtype.F32, tensor.MustShape(8, 8)), AttentionQNorm: b.Input("qn", dtype.F32, tensor.MustShape(4)), AttentionKNorm: b.Input("kn", dtype.F32, tensor.MustShape(4)), FeedForwardNorm: b.Input("fn", dtype.F32, tensor.MustShape(8)), FeedForwardGate: b.Input("fg", dtype.F32, tensor.MustShape(8, 12)), FeedForwardUp: b.Input("fu", dtype.F32, tensor.MustShape(8, 12)), FeedForwardDown: b.Input("fd", dtype.F32, tensor.MustShape(12, 8))}
	r, err := model.BuildDenseBlockCachedForLayer(b, in, s, w, []uint32{0, 1, 2}, nil, nil, 0)
	if err != nil {
		t.Fatal(err)
	}
	feeds := map[*tensor.Tensor]reference.Value{in: patternedValue(in.Shape, 3, 0.2, 0)}
	for i, n := range []*tensor.Tensor{w.AttentionQKV, w.AttentionOutput, w.FeedForwardGate, w.FeedForwardUp, w.FeedForwardDown} {
		feeds[n] = patternedValue(n.Shape, i+11, 0.05, 0)
	}
	for i, n := range []*tensor.Tensor{w.AttentionNorm, w.AttentionQNorm, w.AttentionKNorm, w.FeedForwardNorm} {
		feeds[n] = patternedValue(n.Shape, i+31, 0.03, 1)
	}
	outputs := []*tensor.Tensor{r.Output, r.Key, r.Value}
	want, err := reference.Execute(outputs, feeds)
	if err != nil {
		t.Fatal(err)
	}
	cuda, err := New(0)
	if err != nil {
		t.Fatal(err)
	}
	defer cuda.Close()
	got, err := cuda.Execute(context.Background(), outputs, feeds)
	if err != nil {
		t.Fatal(err)
	}
	compare(t, got[r.Output].Data, want[r.Output].Data, 1e-3)
	compare(t, got[r.Key].Data, want[r.Key].Data, 7e-5)
	compare(t, got[r.Value].Data, want[r.Value].Data, 5e-5)
}

func TestExecutorCogVLMTokenBlockMatchesReference(t *testing.T) {
	cudatest.Require(t)
	b := tensor.NewBuilder()
	s := model.Spec{CommonSpec: model.CommonSpec{Architecture: "cogvlm", EmbeddingLength: 8, FeedForwardLength: 12, RMSNormEpsilon: 1e-6}, AttentionSpec: model.AttentionSpec{HeadCount: 2, HeadCountKV: 2, KeyLength: 4, ValueLength: 4, RopeDimensionCount: 4, RopeFrequencyBase: 10000}}
	in := b.Input("input", dtype.F32, tensor.MustShape(8, 3))
	w := model.LayerGraphWeights{AttentionNorm: b.Input("an", dtype.F32, tensor.MustShape(8)), AttentionQKV: b.Input("qkv", dtype.F32, tensor.MustShape(8, 24)), AttentionOutput: b.Input("o", dtype.F32, tensor.MustShape(8, 8)), FeedForwardNorm: b.Input("fn", dtype.F32, tensor.MustShape(8)), FeedForwardGate: b.Input("fg", dtype.F32, tensor.MustShape(8, 12)), FeedForwardUp: b.Input("fu", dtype.F32, tensor.MustShape(8, 12)), FeedForwardDown: b.Input("fd", dtype.F32, tensor.MustShape(12, 8))}
	r, err := model.BuildDenseBlockCachedForLayer(b, in, s, w, []uint32{0, 1, 2}, nil, nil, 0)
	if err != nil {
		t.Fatal(err)
	}
	feeds := map[*tensor.Tensor]reference.Value{in: patternedValue(in.Shape, 3, 0.2, 0)}
	for i, node := range []*tensor.Tensor{w.AttentionQKV, w.AttentionOutput, w.FeedForwardGate, w.FeedForwardUp, w.FeedForwardDown} {
		feeds[node] = patternedValue(node.Shape, i+11, 0.05, 0)
	}
	for i, node := range []*tensor.Tensor{w.AttentionNorm, w.FeedForwardNorm} {
		feeds[node] = patternedValue(node.Shape, i+31, 0.03, 1)
	}
	outputs := []*tensor.Tensor{r.Output, r.Key, r.Value}
	want, err := reference.Execute(outputs, feeds)
	if err != nil {
		t.Fatal(err)
	}
	cuda, err := New(0)
	if err != nil {
		t.Fatal(err)
	}
	defer cuda.Close()
	got, err := cuda.Execute(context.Background(), outputs, feeds)
	if err != nil {
		t.Fatal(err)
	}
	compare(t, got[r.Output].Data, want[r.Output].Data, 1e-3)
	compare(t, got[r.Key].Data, want[r.Key].Data, 7e-5)
	compare(t, got[r.Value].Data, want[r.Value].Data, 5e-5)
}

func TestExecutorDBRXBlockMatchesReference(t *testing.T) {
	cudatest.Require(t)
	builder := tensor.NewBuilder()
	spec := model.Spec{CommonSpec: model.CommonSpec{Architecture: "dbrx", BlockCount: 1, EmbeddingLength: 8, FeedForwardLength: 6,

		LayerNormEpsilon: 1e-5}, AttentionSpec: model.AttentionSpec{HeadCount: 2, HeadCountKV: 1, KeyLength: 4, ValueLength: 4,
		AttentionClamp: 0.35, RopeFrequencyBase: 10000}, MoESpec: model.MoESpec{ExpertCount: 4, ExpertUsedCount: 2, ExpertFeedForward: 6,
		ExpertWeightsScale: 1.25, ExpertWeightsNorm: true},
	}
	input := builder.Input("input", dtype.F32, tensor.MustShape(8, 3))
	weights := model.LayerGraphWeights{
		AttentionNorm:          builder.Input("attn_norm", dtype.F32, tensor.MustShape(8)),
		AttentionQKV:           builder.Input("qkv", dtype.F32, tensor.MustShape(8, 16)),
		AttentionOutput:        builder.Input("attn_out", dtype.F32, tensor.MustShape(8, 8)),
		FeedForwardNorm:        builder.Input("attn_out_norm", dtype.F32, tensor.MustShape(8)),
		FeedForwardRouter:      builder.Input("router", dtype.F32, tensor.MustShape(8, 4)),
		FeedForwardGateExperts: builder.Input("gate_exps", dtype.F32, tensor.MustShape(8, 6, 4)),
		FeedForwardUpExperts:   builder.Input("up_exps", dtype.F32, tensor.MustShape(8, 6, 4)),
		FeedForwardDownExperts: builder.Input("down_exps", dtype.F32, tensor.MustShape(6, 8, 4)),
	}
	result, err := model.BuildDenseBlockCachedForLayer(builder, input, spec, weights, []uint32{0, 1, 2}, nil, nil, 0)
	if err != nil {
		t.Fatal(err)
	}
	feeds := map[*tensor.Tensor]reference.Value{input: patternedValue(input.Shape, 3, 0.2, 0)}
	for index, node := range []*tensor.Tensor{
		weights.AttentionQKV, weights.AttentionOutput, weights.FeedForwardRouter,
		weights.FeedForwardGateExperts, weights.FeedForwardUpExperts, weights.FeedForwardDownExperts,
	} {
		feeds[node] = patternedValue(node.Shape, index+11, 0.05, 0)
	}
	for index, node := range []*tensor.Tensor{weights.AttentionNorm, weights.FeedForwardNorm} {
		feeds[node] = patternedValue(node.Shape, index+37, 0.03, 1)
	}
	outputs := []*tensor.Tensor{result.Output, result.Key, result.Value}
	want, err := reference.Execute(outputs, feeds)
	if err != nil {
		t.Fatal(err)
	}
	cuda, err := New(0)
	if err != nil {
		t.Fatal(err)
	}
	defer cuda.Close()
	got, err := cuda.Execute(context.Background(), outputs, feeds)
	if err != nil {
		t.Fatal(err)
	}
	compare(t, got[result.Output].Data, want[result.Output].Data, 1e-3)
	compare(t, got[result.Key].Data, want[result.Key].Data, 7e-5)
	compare(t, got[result.Value].Data, want[result.Value].Data, 5e-5)
}

func TestExecutorArcticBlockMatchesReference(t *testing.T) {
	cudatest.Require(t)
	builder := tensor.NewBuilder()
	spec := model.Spec{CommonSpec: model.CommonSpec{Architecture: "arctic", BlockCount: 1, EmbeddingLength: 8, FeedForwardLength: 12,

		RMSNormEpsilon: 1e-5}, AttentionSpec: model.AttentionSpec{HeadCount: 2, HeadCountKV: 1, KeyLength: 4, ValueLength: 4,
		RopeFrequencyBase: 10000}, MoESpec: model.MoESpec{ExpertCount: 4, ExpertUsedCount: 2, ExpertFeedForward: 12, ExpertWeightsScale: 1},
	}
	input := builder.Input("input", dtype.F32, tensor.MustShape(8, 3))
	weights := model.LayerGraphWeights{
		AttentionNorm:          builder.Input("attn_norm", dtype.F32, tensor.MustShape(8)),
		AttentionQ:             builder.Input("q", dtype.F32, tensor.MustShape(8, 8)),
		AttentionK:             builder.Input("k", dtype.F32, tensor.MustShape(8, 4)),
		AttentionV:             builder.Input("v", dtype.F32, tensor.MustShape(8, 4)),
		AttentionOutput:        builder.Input("attn_out", dtype.F32, tensor.MustShape(8, 8)),
		FeedForwardNorm:        builder.Input("ffn_norm", dtype.F32, tensor.MustShape(8)),
		FeedForwardGate:        builder.Input("dense_gate", dtype.F32, tensor.MustShape(8, 8)),
		FeedForwardUp:          builder.Input("dense_up", dtype.F32, tensor.MustShape(8, 8)),
		FeedForwardDown:        builder.Input("dense_down", dtype.F32, tensor.MustShape(8, 8)),
		FeedForwardExpertNorm:  builder.Input("expert_norm", dtype.F32, tensor.MustShape(8)),
		FeedForwardRouter:      builder.Input("router", dtype.F32, tensor.MustShape(8, 4)),
		FeedForwardGateExperts: builder.Input("gate_exps", dtype.F32, tensor.MustShape(8, 12, 4)),
		FeedForwardUpExperts:   builder.Input("up_exps", dtype.F32, tensor.MustShape(8, 12, 4)),
		FeedForwardDownExperts: builder.Input("down_exps", dtype.F32, tensor.MustShape(12, 8, 4)),
	}
	result, err := model.BuildDenseBlockCachedForLayer(builder, input, spec, weights, []uint32{0, 1, 2}, nil, nil, 0)
	if err != nil {
		t.Fatal(err)
	}
	feeds := map[*tensor.Tensor]reference.Value{input: patternedValue(input.Shape, 3, 0.2, 0)}
	for index, node := range []*tensor.Tensor{
		weights.AttentionQ, weights.AttentionK, weights.AttentionV, weights.AttentionOutput,
		weights.FeedForwardGate, weights.FeedForwardUp, weights.FeedForwardDown,
		weights.FeedForwardRouter, weights.FeedForwardGateExperts,
		weights.FeedForwardUpExperts, weights.FeedForwardDownExperts,
	} {
		feeds[node] = patternedValue(node.Shape, index+11, 0.05, 0)
	}
	for index, node := range []*tensor.Tensor{
		weights.AttentionNorm, weights.FeedForwardNorm, weights.FeedForwardExpertNorm,
	} {
		feeds[node] = patternedValue(node.Shape, index+37, 0.03, 1)
	}
	outputs := []*tensor.Tensor{result.Output, result.Key, result.Value}
	want, err := reference.Execute(outputs, feeds)
	if err != nil {
		t.Fatal(err)
	}
	cuda, err := New(0)
	if err != nil {
		t.Fatal(err)
	}
	defer cuda.Close()
	got, err := cuda.Execute(context.Background(), outputs, feeds)
	if err != nil {
		t.Fatal(err)
	}
	compare(t, got[result.Output].Data, want[result.Output].Data, 1e-3)
	compare(t, got[result.Key].Data, want[result.Key].Data, 7e-5)
	compare(t, got[result.Value].Data, want[result.Value].Data, 5e-5)
}
