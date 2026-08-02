//go:build windows

package executor

import (
	"context"

	"fmt"

	"llamacpp2go/internal/model"

	"llamacpp2go/internal/tensor"

	"llamacpp2go/internal/tensor/dtype"

	"llamacpp2go/internal/tensor/reference"

	"testing"

	cudatest "llamacpp2go/internal/cuda/testutil"
)

func TestExecutorJaisBlockMatchesReference(t *testing.T) {
	cudatest.Require(t)
	builder := tensor.NewBuilder()
	spec := model.Spec{CommonSpec: model.CommonSpec{Architecture: "jais", BlockCount: 1, EmbeddingLength: 8, FeedForwardLength: 12,

		LayerNormEpsilon: 1e-5}, AttentionSpec: model.AttentionSpec{HeadCount: 2, HeadCountKV: 2, KeyLength: 4, ValueLength: 4,
		RopeDisabled: true, AttentionScale: 0.25, MaxALiBiBias: 8},
	}
	input := builder.Input("input", dtype.F32, tensor.MustShape(8, 3))
	weights := model.LayerGraphWeights{
		AttentionNorm:       builder.Input("attn_norm", dtype.F32, tensor.MustShape(8)),
		AttentionNormBias:   builder.Input("attn_norm_bias", dtype.F32, tensor.MustShape(8)),
		AttentionQKV:        builder.Input("qkv", dtype.F32, tensor.MustShape(8, 24)),
		AttentionQKVBias:    builder.Input("qkv_bias", dtype.F32, tensor.MustShape(24)),
		AttentionOutput:     builder.Input("attn_out", dtype.F32, tensor.MustShape(8, 8)),
		AttentionOutputBias: builder.Input("attn_out_bias", dtype.F32, tensor.MustShape(8)),
		FeedForwardNorm:     builder.Input("ffn_norm", dtype.F32, tensor.MustShape(8)),
		FeedForwardNormBias: builder.Input("ffn_norm_bias", dtype.F32, tensor.MustShape(8)),
		FeedForwardGate:     builder.Input("ffn_gate", dtype.F32, tensor.MustShape(8, 12)),
		FeedForwardGateBias: builder.Input("ffn_gate_bias", dtype.F32, tensor.MustShape(12)),
		FeedForwardUp:       builder.Input("ffn_up", dtype.F32, tensor.MustShape(8, 12)),
		FeedForwardUpBias:   builder.Input("ffn_up_bias", dtype.F32, tensor.MustShape(12)),
		FeedForwardDown:     builder.Input("ffn_down", dtype.F32, tensor.MustShape(12, 8)),
		FeedForwardDownBias: builder.Input("ffn_down_bias", dtype.F32, tensor.MustShape(8)),
	}
	result, err := model.BuildDenseBlockCachedForLayer(builder, input, spec, weights, []uint32{0, 1, 2}, nil, nil, 0)
	if err != nil {
		t.Fatal(err)
	}
	feeds := map[*tensor.Tensor]reference.Value{input: patternedValue(input.Shape, 3, 0.2, 0)}
	for index, node := range []*tensor.Tensor{
		weights.AttentionQKV, weights.AttentionOutput,
		weights.FeedForwardGate, weights.FeedForwardUp, weights.FeedForwardDown,
	} {
		feeds[node] = patternedValue(node.Shape, index+11, 0.05, 0)
	}
	for index, node := range []*tensor.Tensor{
		weights.AttentionNorm, weights.FeedForwardNorm,
	} {
		feeds[node] = patternedValue(node.Shape, index+31, 0.03, 1)
	}
	for index, node := range []*tensor.Tensor{
		weights.AttentionNormBias, weights.AttentionQKVBias, weights.AttentionOutputBias,
		weights.FeedForwardNormBias, weights.FeedForwardGateBias,
		weights.FeedForwardUpBias, weights.FeedForwardDownBias,
	} {
		feeds[node] = patternedValue(node.Shape, index+41, 0.02, 0)
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
	compare(t, got[result.Key].Data, want[result.Key].Data, 5e-5)
	compare(t, got[result.Value].Data, want[result.Value].Data, 5e-5)
}

func TestExecutorOpenELMBlockMatchesReference(t *testing.T) {
	cudatest.Require(t)
	builder := tensor.NewBuilder()
	spec := model.Spec{CommonSpec: model.CommonSpec{Architecture: "openelm", BlockCount: 2, EmbeddingLength: 8,
		FeedForwardLength: 12,

		RMSNormEpsilon: 1e-6}, AttentionSpec: model.AttentionSpec{HeadCount: 2, HeadCountKV: 1, LayerHeadCounts: []uint32{2, 4},
		LayerKVHeadCounts: []uint32{1, 2}, KeyLength: 4, ValueLength: 4,
		RopeFrequencyBase: 10000}, MoESpec: model.MoESpec{LayerFeedForward: []uint32{12, 16}},
	}
	input := builder.Input("input", dtype.F32, tensor.MustShape(8, 3))
	weights := model.LayerGraphWeights{
		AttentionNorm:   builder.Input("attn_norm", dtype.F32, tensor.MustShape(8)),
		AttentionQKV:    builder.Input("qkv", dtype.F32, tensor.MustShape(8, 32)),
		AttentionQNorm:  builder.Input("q_norm", dtype.F32, tensor.MustShape(4)),
		AttentionKNorm:  builder.Input("k_norm", dtype.F32, tensor.MustShape(4)),
		AttentionOutput: builder.Input("attn_out", dtype.F32, tensor.MustShape(16, 8)),
		FeedForwardNorm: builder.Input("ffn_norm", dtype.F32, tensor.MustShape(8)),
		FeedForwardGate: builder.Input("ffn_gate", dtype.F32, tensor.MustShape(8, 16)),
		FeedForwardUp:   builder.Input("ffn_up", dtype.F32, tensor.MustShape(8, 16)),
		FeedForwardDown: builder.Input("ffn_down", dtype.F32, tensor.MustShape(16, 8)),
	}
	result, err := model.BuildDenseBlockCachedForLayer(builder, input, spec, weights, []uint32{0, 1, 2}, nil, nil, 1)
	if err != nil {
		t.Fatal(err)
	}
	feeds := map[*tensor.Tensor]reference.Value{input: patternedValue(input.Shape, 3, 0.2, 0)}
	for index, node := range []*tensor.Tensor{
		weights.AttentionQKV, weights.AttentionOutput,
		weights.FeedForwardGate, weights.FeedForwardUp, weights.FeedForwardDown,
	} {
		feeds[node] = patternedValue(node.Shape, index+11, 0.05, 0)
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
	compare(t, got[result.Output].Data, want[result.Output].Data, 1e-3)
	compare(t, got[result.Key].Data, want[result.Key].Data, 7e-5)
	compare(t, got[result.Value].Data, want[result.Value].Data, 5e-5)
}

func TestExecutorDeciMixedLayersMatchReference(t *testing.T) {
	cudatest.Require(t)
	builder := tensor.NewBuilder()
	spec := model.Spec{CommonSpec: model.CommonSpec{Architecture: "deci", BlockCount: 4, EmbeddingLength: 8,
		FeedForwardLength: 12,

		RMSNormEpsilon: 1e-6}, AttentionSpec: model.AttentionSpec{HeadCount: 2, HeadCountKV: 1, LayerHeadCounts: []uint32{2, 2, 0, 0},
		LayerKVHeadCounts: []uint32{1, 0, 0, 0}, KeyLength: 4, ValueLength: 4,
		RopeDimensionCount: 4, RopeFrequencyBase: 10000}, MoESpec: model.MoESpec{LayerFeedForward: []uint32{12, 12, 12, 0}},
	}
	input := builder.Input("input", dtype.F32, tensor.MustShape(8, 3))
	current := input
	feeds := map[*tensor.Tensor]reference.Value{input: patternedValue(input.Shape, 3, 0.2, 0)}
	keys := make([]*tensor.Tensor, 4)
	values := make([]*tensor.Tensor, 4)
	for layer := uint32(0); layer < 4; layer++ {
		weights := model.LayerGraphWeights{}
		weighted := make([]*tensor.Tensor, 0, 7)
		if layer == 0 {
			weights.AttentionNorm = builder.Input("full_attn_norm", dtype.F32, tensor.MustShape(8))
			weights.AttentionQKV = builder.Input("full_qkv", dtype.F32, tensor.MustShape(8, 16))
			weights.AttentionOutput = builder.Input("full_attn_out", dtype.F32, tensor.MustShape(8, 8))
			weighted = append(weighted, weights.AttentionNorm)
			feeds[weights.AttentionQKV] = patternedValue(weights.AttentionQKV.Shape, 11, 0.05, 0)
			feeds[weights.AttentionOutput] = patternedValue(weights.AttentionOutput.Shape, 13, 0.05, 0)
		}
		if layer == 1 {
			weights.AttentionNorm = builder.Input("linear_attn_norm", dtype.F32, tensor.MustShape(8))
			weights.AttentionOutput = builder.Input("linear_attn_out", dtype.F32, tensor.MustShape(8, 8))
			weights.AttentionOutputBias = builder.Input("linear_attn_bias", dtype.F32, tensor.MustShape(8))
			weighted = append(weighted, weights.AttentionNorm)
			feeds[weights.AttentionOutput] = patternedValue(weights.AttentionOutput.Shape, 17, 0.05, 0)
			feeds[weights.AttentionOutputBias] = patternedValue(weights.AttentionOutputBias.Shape, 19, 0.02, 0)
		}
		if layer < 3 {
			weights.FeedForwardNorm = builder.Input(fmt.Sprintf("ffn_norm_%d", layer), dtype.F32, tensor.MustShape(8))
			weights.FeedForwardGate = builder.Input(fmt.Sprintf("ffn_gate_%d", layer), dtype.F32, tensor.MustShape(8, 12))
			weights.FeedForwardUp = builder.Input(fmt.Sprintf("ffn_up_%d", layer), dtype.F32, tensor.MustShape(8, 12))
			weights.FeedForwardDown = builder.Input(fmt.Sprintf("ffn_down_%d", layer), dtype.F32, tensor.MustShape(12, 8))
			weights.FeedForwardGateBias = builder.Input(fmt.Sprintf("ffn_gate_bias_%d", layer), dtype.F32, tensor.MustShape(12))
			weights.FeedForwardUpBias = builder.Input(fmt.Sprintf("ffn_up_bias_%d", layer), dtype.F32, tensor.MustShape(12))
			weights.FeedForwardDownBias = builder.Input(fmt.Sprintf("ffn_down_bias_%d", layer), dtype.F32, tensor.MustShape(8))
			weighted = append(weighted, weights.FeedForwardNorm)
			for index, node := range []*tensor.Tensor{
				weights.FeedForwardGate, weights.FeedForwardUp, weights.FeedForwardDown,
				weights.FeedForwardGateBias, weights.FeedForwardUpBias, weights.FeedForwardDownBias,
			} {
				feeds[node] = patternedValue(node.Shape, int(layer)*20+index+23, 0.04, 0)
			}
		}
		for index, node := range weighted {
			feeds[node] = patternedValue(node.Shape, int(layer)*10+index+41, 0.03, 1)
		}
		result, err := model.BuildDenseBlockCachedForLayer(
			builder, current, spec, weights, []uint32{0, 1, 2}, nil, nil, layer,
		)
		if err != nil {
			t.Fatal(err)
		}
		current, keys[layer], values[layer] = result.Output, result.Key, result.Value
	}
	outputs := []*tensor.Tensor{current}
	for layer := range keys {
		outputs = append(outputs, keys[layer], values[layer])
	}
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
	compare(t, got[current].Data, want[current].Data, 1e-3)
	for layer := range keys {
		compare(t, got[keys[layer]].Data, want[keys[layer]].Data, 7e-5)
		compare(t, got[values[layer]].Data, want[values[layer]].Data, 7e-5)
	}
}

func TestExecutorBailingMoE2BlockMatchesReference(t *testing.T) {
	cudatest.Require(t)
	builder := tensor.NewBuilder()
	spec := model.Spec{CommonSpec: model.CommonSpec{Architecture: "bailingmoe2", BlockCount: 2,
		EmbeddingLength: 8, FeedForwardLength: 16,

		RMSNormEpsilon: 1e-6}, AttentionSpec: model.AttentionSpec{HeadCount: 2, HeadCountKV: 1, KeyLength: 4, ValueLength: 4,
		RopeDimensionCount: 4, RopeFrequencyBase: 10000}, MoESpec: model.MoESpec{LeadingDenseBlocks: 1,

		ExpertCount: 4, ExpertUsedCount: 2, ExpertFeedForward: 6,
		SharedExpertCount: 2, SharedExpertFF: 10, ExpertWeightsScale: 1.25,
		ExpertWeightsNorm: true, ExpertGatingFunc: 2},
	}
	input := builder.Input("input", dtype.F32, tensor.MustShape(8, 3))
	weights := model.LayerGraphWeights{
		AttentionNorm:          builder.Input("attn_norm", dtype.F32, tensor.MustShape(8)),
		AttentionQKV:           builder.Input("qkv", dtype.F32, tensor.MustShape(8, 16)),
		AttentionOutput:        builder.Input("attn_out", dtype.F32, tensor.MustShape(8, 8)),
		AttentionQNorm:         builder.Input("q_norm", dtype.F32, tensor.MustShape(4)),
		AttentionKNorm:         builder.Input("k_norm", dtype.F32, tensor.MustShape(4)),
		FeedForwardNorm:        builder.Input("ffn_norm", dtype.F32, tensor.MustShape(8)),
		FeedForwardRouter:      builder.Input("router", dtype.F32, tensor.MustShape(8, 4)),
		FeedForwardGateExperts: builder.Input("gate_exps", dtype.F32, tensor.MustShape(8, 6, 4)),
		FeedForwardUpExperts:   builder.Input("up_exps", dtype.F32, tensor.MustShape(8, 6, 4)),
		FeedForwardDownExperts: builder.Input("down_exps", dtype.F32, tensor.MustShape(6, 8, 4)),
		FeedForwardExpertBias:  builder.Input("expert_bias", dtype.F32, tensor.MustShape(4)),
		FeedForwardSharedGate:  builder.Input("shared_gate", dtype.F32, tensor.MustShape(8, 10)),
		FeedForwardSharedUp:    builder.Input("shared_up", dtype.F32, tensor.MustShape(8, 10)),
		FeedForwardSharedDown:  builder.Input("shared_down", dtype.F32, tensor.MustShape(10, 8)),
	}
	result, err := model.BuildDenseBlockCachedForLayer(builder, input, spec, weights, []uint32{0, 1, 2}, nil, nil, 1)
	if err != nil {
		t.Fatal(err)
	}
	feeds := map[*tensor.Tensor]reference.Value{input: patternedValue(input.Shape, 3, 0.2, 0)}
	for index, node := range []*tensor.Tensor{
		weights.AttentionQKV, weights.AttentionOutput,
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

func TestExecutorChameleonSandwichBlockMatchesReference(t *testing.T) {
	cudatest.Require(t)
	builder := tensor.NewBuilder()
	spec := model.Spec{CommonSpec: model.CommonSpec{Architecture: "chameleon", EmbeddingLength: 8, FeedForwardLength: 12,

		RMSNormEpsilon: 1e-5,
		SandwichNorm:   true}, AttentionSpec: model.AttentionSpec{HeadCount: 2, HeadCountKV: 1, KeyLength: 4, ValueLength: 4,
		RopeFrequencyBase: 10000, QKNormEpsilon: 1e-5},
	}
	input := builder.Input("input", dtype.F32, tensor.MustShape(8, 2))
	weights := model.LayerGraphWeights{
		AttentionNorm:      builder.Input("attn_norm", dtype.F32, tensor.MustShape(8)),
		AttentionQ:         builder.Input("q", dtype.F32, tensor.MustShape(8, 8)),
		AttentionK:         builder.Input("k", dtype.F32, tensor.MustShape(8, 4)),
		AttentionV:         builder.Input("v", dtype.F32, tensor.MustShape(8, 4)),
		AttentionOutput:    builder.Input("attn_out", dtype.F32, tensor.MustShape(8, 8)),
		AttentionQNorm:     builder.Input("q_norm", dtype.F32, tensor.MustShape(4, 2)),
		AttentionKNorm:     builder.Input("k_norm", dtype.F32, tensor.MustShape(4, 1)),
		AttentionQNormBias: builder.Input("q_norm_bias", dtype.F32, tensor.MustShape(4, 2)),
		AttentionKNormBias: builder.Input("k_norm_bias", dtype.F32, tensor.MustShape(4, 1)),
		FeedForwardNorm:    builder.Input("ffn_norm", dtype.F32, tensor.MustShape(8)),
		FeedForwardGate:    builder.Input("ffn_gate", dtype.F32, tensor.MustShape(8, 12)),
		FeedForwardUp:      builder.Input("ffn_up", dtype.F32, tensor.MustShape(8, 12)),
		FeedForwardDown:    builder.Input("ffn_down", dtype.F32, tensor.MustShape(12, 8)),
	}
	result, err := model.BuildDenseBlockCachedForLayer(
		builder, input, spec, weights, []uint32{0, 1}, nil, nil, 0,
	)
	if err != nil {
		t.Fatal(err)
	}
	feeds := map[*tensor.Tensor]reference.Value{input: patternedValue(input.Shape, 3, 0.2, 0)}
	for index, node := range []*tensor.Tensor{
		weights.AttentionQ, weights.AttentionK, weights.AttentionV, weights.AttentionOutput,
		weights.AttentionQNormBias, weights.AttentionKNormBias,
		weights.FeedForwardGate, weights.FeedForwardUp, weights.FeedForwardDown,
	} {
		feeds[node] = patternedValue(node.Shape, index+11, 0.08, 0)
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
	compare(t, got[result.Output].Data, want[result.Output].Data, 5e-4)
	compare(t, got[result.Key].Data, want[result.Key].Data, 5e-5)
	compare(t, got[result.Value].Data, want[result.Value].Data, 5e-5)
}

func TestExecutorDeepSeek2AbsorbedMLABlockMatchesReference(t *testing.T) {
	cudatest.Require(t)
	builder := tensor.NewBuilder()
	spec := model.Spec{CommonSpec: model.CommonSpec{Architecture: "deepseek2", BlockCount: 27, EmbeddingLength: 8,
		FeedForwardLength: 12,

		RMSNormEpsilon: 1e-6}, AttentionSpec: model.AttentionSpec{HeadCount: 2, HeadCountKV: 2, KeyLength: 6, ValueLength: 4,
		KVLoRARank: 3, RopeDimensionCount: 2, RopeFrequencyBase: 10000}, MoESpec: model.MoESpec{LeadingDenseBlocks: 1}}
	input := builder.Input("input", dtype.F32, tensor.MustShape(8, 2))
	weights := model.LayerGraphWeights{
		AttentionNorm:             builder.Input("attn_norm", dtype.F32, tensor.MustShape(8)),
		AttentionQ:                builder.Input("q", dtype.F32, tensor.MustShape(8, 12)),
		AttentionKVAMQA:           builder.Input("kv_a", dtype.F32, tensor.MustShape(8, 5)),
		AttentionKVANorm:          builder.Input("kv_norm", dtype.F32, tensor.MustShape(3)),
		AttentionKB:               builder.Input("k_b", dtype.F32, tensor.MustShape(4, 3, 2)),
		AttentionVB:               builder.Input("v_b", dtype.F32, tensor.MustShape(3, 4, 2)),
		AttentionOutput:           builder.Input("attn_out", dtype.F32, tensor.MustShape(8, 8)),
		AttentionTemperatureScale: builder.Input("temperature", dtype.F32, tensor.MustShape(1, 1, 2)),
		FeedForwardNorm:           builder.Input("ffn_norm", dtype.F32, tensor.MustShape(8)),
		FeedForwardGate:           builder.Input("ffn_gate", dtype.F32, tensor.MustShape(8, 12)),
		FeedForwardUp:             builder.Input("ffn_up", dtype.F32, tensor.MustShape(8, 12)),
		FeedForwardDown:           builder.Input("ffn_down", dtype.F32, tensor.MustShape(12, 8)),
	}
	result, err := model.BuildMLABlockCachedForLayer(builder, input, spec, weights, []uint32{0, 1}, nil, nil, 0)
	if err != nil {
		t.Fatal(err)
	}
	feeds := map[*tensor.Tensor]reference.Value{input: patternedValue(input.Shape, 3, 0.05, -0.1)}
	for index, node := range []*tensor.Tensor{weights.AttentionNorm, weights.AttentionQ,
		weights.AttentionKVAMQA, weights.AttentionKVANorm, weights.AttentionKB, weights.AttentionVB,
		weights.AttentionOutput, weights.FeedForwardNorm, weights.FeedForwardGate,
		weights.FeedForwardUp, weights.FeedForwardDown} {
		feeds[node] = patternedValue(node.Shape, index+5, 0.025, 0.1)
	}
	feeds[weights.AttentionTemperatureScale] = reference.Value{Shape: weights.AttentionTemperatureScale.Shape, Data: []float32{1, 1.1}}
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
	compare(t, got[result.Output].Data, want[result.Output].Data, 5e-4)
	compare(t, got[result.Key].Data, want[result.Key].Data, 5e-5)
	compare(t, got[result.Value].Data, want[result.Value].Data, 5e-5)
}

func TestExecutorPLMMLABlockMatchesReference(t *testing.T) {
	cudatest.Require(t)
	builder := tensor.NewBuilder()
	spec := model.Spec{CommonSpec: model.CommonSpec{Architecture: "plm", EmbeddingLength: 8, FeedForwardLength: 12,

		RMSNormEpsilon: 1e-6}, AttentionSpec: model.AttentionSpec{HeadCount: 2, HeadCountKV: 2, KeyLength: 6, ValueLength: 4,
		KVLoRARank: 3, RopeDimensionCount: 2, RopeFrequencyBase: 10000},
	}
	input := builder.Input("input", dtype.F32, tensor.MustShape(8, 2))
	weights := model.LayerGraphWeights{
		AttentionNorm:    builder.Input("attn_norm", dtype.F32, tensor.MustShape(8)),
		AttentionQ:       builder.Input("q", dtype.F32, tensor.MustShape(8, 12)),
		AttentionKVAMQA:  builder.Input("kv_a", dtype.F32, tensor.MustShape(8, 5)),
		AttentionKVANorm: builder.Input("kv_a_norm", dtype.F32, tensor.MustShape(3)),
		AttentionKVB:     builder.Input("kv_b", dtype.F32, tensor.MustShape(3, 16)),
		AttentionOutput:  builder.Input("attn_out", dtype.F32, tensor.MustShape(8, 8)),
		FeedForwardNorm:  builder.Input("ffn_norm", dtype.F32, tensor.MustShape(8)),
		FeedForwardUp:    builder.Input("ffn_up", dtype.F32, tensor.MustShape(8, 12)),
		FeedForwardDown:  builder.Input("ffn_down", dtype.F32, tensor.MustShape(12, 8)),
	}
	result, err := model.BuildPLMBlockCached(builder, input, spec, weights, []uint32{0, 1}, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	feeds := map[*tensor.Tensor]reference.Value{
		input: patternedValue(input.Shape, 3, 0.2, 0),
	}
	for index, node := range []*tensor.Tensor{
		weights.AttentionQ, weights.AttentionKVAMQA, weights.AttentionKVB,
		weights.AttentionOutput, weights.FeedForwardUp, weights.FeedForwardDown,
	} {
		feeds[node] = patternedValue(node.Shape, index+11, 0.08, 0)
	}
	feeds[weights.AttentionNorm] = patternedValue(weights.AttentionNorm.Shape, 23, 0.03, 1)
	feeds[weights.AttentionKVANorm] = patternedValue(weights.AttentionKVANorm.Shape, 29, 0.03, 1)
	feeds[weights.FeedForwardNorm] = patternedValue(weights.FeedForwardNorm.Shape, 31, 0.03, 1)
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
	compare(t, got[result.Output].Data, want[result.Output].Data, 5e-4)
	compare(t, got[result.Key].Data, want[result.Key].Data, 5e-5)
	compare(t, got[result.Value].Data, want[result.Value].Data, 5e-5)
}

func TestExecutorMiniCPM3MLABlockMatchesReference(t *testing.T) {
	cudatest.Require(t)
	builder := tensor.NewBuilder()
	spec := model.Spec{CommonSpec: model.CommonSpec{Architecture: "minicpm3", EmbeddingLength: 8, FeedForwardLength: 12,

		ResidualScale: 0.7, RMSNormEpsilon: 1e-6}, AttentionSpec: model.AttentionSpec{HeadCount: 2, HeadCountKV: 2, KeyLength: 6, ValueLength: 4,
		QLoRARank: 3, KVLoRARank: 3, RopeDimensionCount: 2, RopeFrequencyBase: 10000,
		RopeAttentionFactor: 1}}
	input := builder.Input("input", dtype.F32, tensor.MustShape(8, 2))
	weights := model.LayerGraphWeights{
		AttentionNorm:    builder.Input("attn_norm", dtype.F32, tensor.MustShape(8)),
		AttentionQ:       builder.Input("q_a", dtype.F32, tensor.MustShape(8, 3)),
		AttentionQNorm:   builder.Input("q_a_norm", dtype.F32, tensor.MustShape(3)),
		AttentionQB:      builder.Input("q_b", dtype.F32, tensor.MustShape(3, 12)),
		AttentionKVAMQA:  builder.Input("kv_a", dtype.F32, tensor.MustShape(8, 5)),
		AttentionKVANorm: builder.Input("kv_a_norm", dtype.F32, tensor.MustShape(3)),
		AttentionKVB:     builder.Input("kv_b", dtype.F32, tensor.MustShape(3, 16)),
		AttentionOutput:  builder.Input("attn_out", dtype.F32, tensor.MustShape(8, 8)),
		RopeFactors:      builder.Input("rope", dtype.F32, tensor.MustShape(1)),
		FeedForwardNorm:  builder.Input("ffn_norm", dtype.F32, tensor.MustShape(8)),
		FeedForwardGate:  builder.Input("ffn_gate", dtype.F32, tensor.MustShape(8, 12)),
		FeedForwardUp:    builder.Input("ffn_up", dtype.F32, tensor.MustShape(8, 12)),
		FeedForwardDown:  builder.Input("ffn_down", dtype.F32, tensor.MustShape(12, 8)),
	}
	result, err := model.BuildMLABlockCached(builder, input, spec, weights, []uint32{0, 1}, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	feeds := map[*tensor.Tensor]reference.Value{input: patternedValue(input.Shape, 3, 0.2, 0)}
	for index, node := range []*tensor.Tensor{
		weights.AttentionQ, weights.AttentionQB, weights.AttentionKVAMQA, weights.AttentionKVB,
		weights.AttentionOutput, weights.FeedForwardGate, weights.FeedForwardUp, weights.FeedForwardDown,
	} {
		feeds[node] = patternedValue(node.Shape, index+11, 0.08, 0)
	}
	for index, node := range []*tensor.Tensor{
		weights.AttentionNorm, weights.AttentionQNorm, weights.AttentionKVANorm, weights.FeedForwardNorm,
	} {
		feeds[node] = patternedValue(node.Shape, index+31, 0.03, 1)
	}
	feeds[weights.RopeFactors] = patternedValue(weights.RopeFactors.Shape, 47, 0.05, 1)
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
	compare(t, got[result.Output].Data, want[result.Output].Data, 5e-4)
	compare(t, got[result.Key].Data, want[result.Key].Data, 5e-5)
	compare(t, got[result.Value].Data, want[result.Value].Data, 5e-5)
}

func TestExecutorCohere2MoEBlockMatchesReference(t *testing.T) {
	cudatest.Require(t)
	builder := tensor.NewBuilder()
	spec := model.Spec{CommonSpec: model.CommonSpec{Architecture: "cohere2moe", BlockCount: 2, EmbeddingLength: 8,
		FeedForwardLength: 12,

		RMSNormEpsilon: 1e-5}, AttentionSpec: model.AttentionSpec{HeadCount: 2, HeadCountKV: 1,
		KeyLength: 4, ValueLength: 4, RopeDimensionCount: 4,
		RopeFrequencyBase: 10000, RopeFrequencySWA: 20000,
		SlidingWindow: 128,
		SlidingLayers: []bool{false, true}}, MoESpec: model.MoESpec{LeadingDenseBlocks: 1,
		ExpertCount: 4, ExpertUsedCount: 2, ExpertFeedForward: 6,
		ExpertGatingFunc: 2, ExpertWeightsNorm: true, ExpertWeightsScale: 1.25,
		SharedExpertCount: 1, SharedExpertFF: 6},
	}
	input := builder.Input("input", dtype.F32, tensor.MustShape(8, 2))
	weights := model.LayerGraphWeights{
		AttentionNorm:            builder.Input("attn_norm", dtype.F32, tensor.MustShape(8)),
		AttentionQKV:             builder.Input("attn_qkv", dtype.F32, tensor.MustShape(8, 16)),
		AttentionOutput:          builder.Input("attn_output", dtype.F32, tensor.MustShape(8, 8)),
		FeedForwardRouter:        builder.Input("router", dtype.F32, tensor.MustShape(8, 4)),
		FeedForwardGateUpExperts: builder.Input("gate_up_exps", dtype.F32, tensor.MustShape(8, 12, 4)),
		FeedForwardDownExperts:   builder.Input("down_exps", dtype.F32, tensor.MustShape(6, 8, 4)),
		FeedForwardSharedGate:    builder.Input("shared_gate", dtype.F32, tensor.MustShape(8, 6)),
		FeedForwardSharedUp:      builder.Input("shared_up", dtype.F32, tensor.MustShape(8, 6)),
		FeedForwardSharedDown:    builder.Input("shared_down", dtype.F32, tensor.MustShape(6, 8)),
	}
	result, err := model.BuildDenseBlockCachedForLayer(
		builder, input, spec, weights, []uint32{0, 1}, nil, nil, 1,
	)
	if err != nil {
		t.Fatal(err)
	}
	feeds := map[*tensor.Tensor]reference.Value{
		input:                 patternedValue(input.Shape, 3, 0.2, 0),
		weights.AttentionNorm: patternedValue(weights.AttentionNorm.Shape, 5, 0.03, 1),
	}
	for index, node := range []*tensor.Tensor{
		weights.AttentionQKV, weights.AttentionOutput, weights.FeedForwardRouter,
		weights.FeedForwardGateUpExperts, weights.FeedForwardDownExperts,
		weights.FeedForwardSharedGate, weights.FeedForwardSharedUp, weights.FeedForwardSharedDown,
	} {
		feeds[node] = patternedValue(node.Shape, index+11, 0.07, 0)
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

func TestExecutorHYV3MoEBlockMatchesReference(t *testing.T) {
	cudatest.Require(t)
	builder := tensor.NewBuilder()
	spec := model.Spec{CommonSpec: model.CommonSpec{Architecture: "hy_v3", BlockCount: 2, EmbeddingLength: 8,
		FeedForwardLength: 12,

		RMSNormEpsilon: 1e-5}, AttentionSpec: model.AttentionSpec{HeadCount: 2, HeadCountKV: 1,
		KeyLength: 4, ValueLength: 4, RopeDimensionCount: 4,
		RopeFrequencyBase: 10000}, MoESpec: model.MoESpec{ExpertCount: 4, ExpertUsedCount: 2, ExpertFeedForward: 6,
		ExpertGatingFunc: 1, ExpertWeightsNorm: true, ExpertWeightsScale: 1.25,
		SharedExpertFF: 6},
	}
	input := builder.Input("input", dtype.F32, tensor.MustShape(8, 2))
	weights := model.LayerGraphWeights{
		AttentionNorm:            builder.Input("attn_norm", dtype.F32, tensor.MustShape(8)),
		AttentionQKV:             builder.Input("attn_qkv", dtype.F32, tensor.MustShape(8, 16)),
		AttentionQNorm:           builder.Input("attn_q_norm", dtype.F32, tensor.MustShape(4)),
		AttentionKNorm:           builder.Input("attn_k_norm", dtype.F32, tensor.MustShape(4)),
		AttentionOutput:          builder.Input("attn_output", dtype.F32, tensor.MustShape(8, 8)),
		FeedForwardNorm:          builder.Input("ffn_norm", dtype.F32, tensor.MustShape(8)),
		FeedForwardRouter:        builder.Input("router", dtype.F32, tensor.MustShape(8, 4)),
		FeedForwardGateUpExperts: builder.Input("gate_up_exps", dtype.F32, tensor.MustShape(8, 12, 4)),
		FeedForwardDownExperts:   builder.Input("down_exps", dtype.F32, tensor.MustShape(6, 8, 4)),
		FeedForwardExpertBias:    builder.Input("expert_bias", dtype.F32, tensor.MustShape(4)),
		FeedForwardSharedGate:    builder.Input("shared_gate", dtype.F32, tensor.MustShape(8, 6)),
		FeedForwardSharedUp:      builder.Input("shared_up", dtype.F32, tensor.MustShape(8, 6)),
		FeedForwardSharedDown:    builder.Input("shared_down", dtype.F32, tensor.MustShape(6, 8)),
	}
	result, err := model.BuildDenseBlockCachedForLayer(
		builder, input, spec, weights, []uint32{0, 1}, nil, nil, 1,
	)
	if err != nil {
		t.Fatal(err)
	}
	feeds := map[*tensor.Tensor]reference.Value{
		input:                         patternedValue(input.Shape, 3, 0.2, 0),
		weights.AttentionNorm:         patternedValue(weights.AttentionNorm.Shape, 5, 0.03, 1),
		weights.AttentionQNorm:        patternedValue(weights.AttentionQNorm.Shape, 7, 0.02, 1),
		weights.AttentionKNorm:        patternedValue(weights.AttentionKNorm.Shape, 9, 0.02, 1),
		weights.FeedForwardNorm:       patternedValue(weights.FeedForwardNorm.Shape, 11, 0.03, 1),
		weights.FeedForwardExpertBias: {Shape: weights.FeedForwardExpertBias.Shape, Data: []float32{0.1, -0.2, 0.3, -0.1}},
	}
	for index, node := range []*tensor.Tensor{
		weights.AttentionQKV, weights.AttentionOutput, weights.FeedForwardRouter,
		weights.FeedForwardGateUpExperts, weights.FeedForwardDownExperts,
		weights.FeedForwardSharedGate, weights.FeedForwardSharedUp, weights.FeedForwardSharedDown,
	} {
		feeds[node] = patternedValue(node.Shape, index+13, 0.07, 0)
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

func TestExecutorDeepSeek2OCRMoEBlockMatchesReference(t *testing.T) {
	cudatest.Require(t)
	builder := tensor.NewBuilder()
	spec := model.Spec{CommonSpec: model.CommonSpec{Architecture: "deepseek2-ocr", BlockCount: 2, EmbeddingLength: 8,
		FeedForwardLength: 12,

		RMSNormEpsilon: 1e-6}, AttentionSpec: model.AttentionSpec{HeadCount: 2, HeadCountKV: 2,
		KeyLength: 4, ValueLength: 4, RopeDimensionCount: 4,
		RopeFrequencyBase: 10000}, MoESpec: model.MoESpec{ExpertCount: 4, ExpertUsedCount: 2, ExpertFeedForward: 6,
		ExpertGatingFunc: 1, ExpertWeightsScale: 1, SharedExpertFF: 12},
	}
	input := builder.Input("input", dtype.F32, tensor.MustShape(8, 2))
	weights := model.LayerGraphWeights{
		AttentionNorm:            builder.Input("attn_norm", dtype.F32, tensor.MustShape(8)),
		AttentionQ:               builder.Input("attn_q", dtype.F32, tensor.MustShape(8, 8)),
		AttentionK:               builder.Input("attn_k", dtype.F32, tensor.MustShape(8, 8)),
		AttentionV:               builder.Input("attn_v", dtype.F32, tensor.MustShape(8, 8)),
		AttentionOutput:          builder.Input("attn_output", dtype.F32, tensor.MustShape(8, 8)),
		FeedForwardNorm:          builder.Input("ffn_norm", dtype.F32, tensor.MustShape(8)),
		FeedForwardRouter:        builder.Input("router", dtype.F32, tensor.MustShape(8, 4)),
		FeedForwardGateUpExperts: builder.Input("gate_up_exps", dtype.F32, tensor.MustShape(8, 12, 4)),
		FeedForwardDownExperts:   builder.Input("down_exps", dtype.F32, tensor.MustShape(6, 8, 4)),
		FeedForwardExpertBias:    builder.Input("expert_bias", dtype.F32, tensor.MustShape(4)),
		FeedForwardSharedGate:    builder.Input("shared_gate", dtype.F32, tensor.MustShape(8, 12)),
		FeedForwardSharedUp:      builder.Input("shared_up", dtype.F32, tensor.MustShape(8, 12)),
		FeedForwardSharedDown:    builder.Input("shared_down", dtype.F32, tensor.MustShape(12, 8)),
	}
	result, err := model.BuildDenseBlockCachedForLayer(
		builder, input, spec, weights, []uint32{0, 1}, nil, nil, 1,
	)
	if err != nil {
		t.Fatal(err)
	}
	feeds := map[*tensor.Tensor]reference.Value{
		input:                         patternedValue(input.Shape, 3, 0.2, 0),
		weights.AttentionNorm:         patternedValue(weights.AttentionNorm.Shape, 5, 0.03, 1),
		weights.FeedForwardNorm:       patternedValue(weights.FeedForwardNorm.Shape, 7, 0.03, 1),
		weights.FeedForwardExpertBias: {Shape: weights.FeedForwardExpertBias.Shape, Data: []float32{0.1, -0.2, 0.3, -0.1}},
	}
	for index, node := range []*tensor.Tensor{
		weights.AttentionQ, weights.AttentionK, weights.AttentionV, weights.AttentionOutput,
		weights.FeedForwardRouter, weights.FeedForwardGateUpExperts, weights.FeedForwardDownExperts,
		weights.FeedForwardSharedGate, weights.FeedForwardSharedUp, weights.FeedForwardSharedDown,
	} {
		feeds[node] = patternedValue(node.Shape, index+11, 0.07, 0)
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

func TestExecutorErnie45MoEBlockMatchesReference(t *testing.T) {
	cudatest.Require(t)
	builder := tensor.NewBuilder()
	spec := model.Spec{CommonSpec: model.CommonSpec{Architecture: "ernie4_5-moe", BlockCount: 4, EmbeddingLength: 8,
		FeedForwardLength: 12,

		RMSNormEpsilon: 1e-6}, AttentionSpec: model.AttentionSpec{HeadCount: 2, HeadCountKV: 1,
		KeyLength: 4, ValueLength: 4, RopeDimensionCount: 4,
		RopeFrequencyBase: 10000}, MoESpec: model.MoESpec{ExpertCount: 4, ExpertUsedCount: 2, ExpertFeedForward: 6,
		ExpertWeightsScale: 1.25, ExpertWeightsNorm: true,
		LeadingDenseBlocks: 1, MoELayerStep: 2, SharedExpertFF: 5},
	}
	input := builder.Input("input", dtype.F32, tensor.MustShape(8, 2))
	weights := model.LayerGraphWeights{
		AttentionNorm:          builder.Input("attn_norm", dtype.F32, tensor.MustShape(8)),
		AttentionQKV:           builder.Input("attn_qkv", dtype.F32, tensor.MustShape(8, 16)),
		AttentionOutput:        builder.Input("attn_output", dtype.F32, tensor.MustShape(8, 8)),
		FeedForwardNorm:        builder.Input("ffn_norm", dtype.F32, tensor.MustShape(8)),
		FeedForwardRouter:      builder.Input("router", dtype.F32, tensor.MustShape(8, 4)),
		FeedForwardUpExperts:   builder.Input("up_exps", dtype.F32, tensor.MustShape(8, 6, 4)),
		FeedForwardDownExperts: builder.Input("down_exps", dtype.F32, tensor.MustShape(6, 8, 4)),
		FeedForwardExpertBias:  builder.Input("expert_bias", dtype.F32, tensor.MustShape(4)),
		FeedForwardSharedGate:  builder.Input("shared_gate", dtype.F32, tensor.MustShape(8, 5)),
		FeedForwardSharedUp:    builder.Input("shared_up", dtype.F32, tensor.MustShape(8, 5)),
		FeedForwardSharedDown:  builder.Input("shared_down", dtype.F32, tensor.MustShape(5, 8)),
	}
	result, err := model.BuildDenseBlockCachedForLayer(
		builder, input, spec, weights, []uint32{0, 1}, nil, nil, 1,
	)
	if err != nil {
		t.Fatal(err)
	}
	feeds := map[*tensor.Tensor]reference.Value{
		input:                 patternedValue(input.Shape, 3, 0.2, 0),
		weights.AttentionNorm: patternedValue(weights.AttentionNorm.Shape, 5, 0.03, 1),
		weights.FeedForwardNorm: patternedValue(
			weights.FeedForwardNorm.Shape, 7, 0.03, 1,
		),
		weights.FeedForwardExpertBias: patternedValue(
			weights.FeedForwardExpertBias.Shape, 2, 0.1, 0,
		),
	}
	for index, node := range []*tensor.Tensor{
		weights.AttentionQKV, weights.AttentionOutput, weights.FeedForwardRouter,
		weights.FeedForwardUpExperts, weights.FeedForwardDownExperts,
		weights.FeedForwardSharedGate, weights.FeedForwardSharedUp, weights.FeedForwardSharedDown,
	} {
		feeds[node] = patternedValue(node.Shape, index+11, 0.06, 0)
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

func TestExecutorPLaMo3BlockMatchesReference(t *testing.T) {
	cudatest.Require(t)
	builder := tensor.NewBuilder()
	spec := model.Spec{CommonSpec: model.CommonSpec{Architecture: "plamo3", BlockCount: 2, EmbeddingLength: 8,
		FeedForwardLength: 12,

		RMSNormEpsilon: 1e-6}, AttentionSpec: model.AttentionSpec{HeadCount: 2, HeadCountKV: 1,
		KeyLength: 2, ValueLength: 2, RopeDimensionCount: 2,
		RopeFrequencyBase: 10000, RopeFrequencySWA: 20000,
		SlidingWindow: 128, SlidingPattern: 2,
		LayerHeadCounts: []uint32{2, 4}, LayerKVHeadCounts: []uint32{1, 2}}, MoESpec: model.MoESpec{LayerFeedForward: []uint32{12, 16}},
	}
	input := builder.Input("input", dtype.F32, tensor.MustShape(8, 2))
	weights := model.LayerGraphWeights{
		AttentionNorm:       builder.Input("attn_norm", dtype.F32, tensor.MustShape(8)),
		AttentionQKV:        builder.Input("attn_qkv", dtype.F32, tensor.MustShape(8, 8)),
		AttentionQNorm:      builder.Input("attn_q_norm", dtype.F32, tensor.MustShape(2)),
		AttentionKNorm:      builder.Input("attn_k_norm", dtype.F32, tensor.MustShape(2)),
		AttentionOutput:     builder.Input("attn_output", dtype.F32, tensor.MustShape(4, 8)),
		AttentionPostNorm:   builder.Input("attn_post_norm", dtype.F32, tensor.MustShape(8)),
		FeedForwardNorm:     builder.Input("ffn_norm", dtype.F32, tensor.MustShape(8)),
		FeedForwardUp:       builder.Input("ffn_up", dtype.F32, tensor.MustShape(8, 24)),
		FeedForwardDown:     builder.Input("ffn_down", dtype.F32, tensor.MustShape(12, 8)),
		FeedForwardPostNorm: builder.Input("ffn_post_norm", dtype.F32, tensor.MustShape(8)),
	}
	result, err := model.BuildDenseBlockCachedForLayer(
		builder, input, spec, weights, []uint32{0, 1}, nil, nil, 0,
	)
	if err != nil {
		t.Fatal(err)
	}
	feeds := map[*tensor.Tensor]reference.Value{
		input: patternedValue(input.Shape, 3, 0.2, 0),
	}
	for index, node := range []*tensor.Tensor{
		weights.AttentionNorm, weights.AttentionQKV, weights.AttentionQNorm,
		weights.AttentionKNorm, weights.AttentionOutput, weights.AttentionPostNorm,
		weights.FeedForwardNorm, weights.FeedForwardUp, weights.FeedForwardDown,
		weights.FeedForwardPostNorm,
	} {
		offset := float32(0)
		if node.Shape.Rank == 1 {
			offset = 1
		}
		feeds[node] = patternedValue(node.Shape, index+5, 0.05, offset)
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
	compare(t, got[result.Output].Data, want[result.Output].Data, 6e-4)
	compare(t, got[result.Key].Data, want[result.Key].Data, 5e-5)
	compare(t, got[result.Value].Data, want[result.Value].Data, 5e-5)
}

func TestExecutorPaddleOCRBlockMatchesReference(t *testing.T) {
	testExecutorMRoPETextDecoderBlockMatchesReference(t, "paddleocr")
}

func TestExecutorQwen2VLBlockMatchesReference(t *testing.T) {
	testExecutorMRoPETextDecoderBlockMatchesReference(t, "qwen2vl")
}

func TestExecutorQwen3VLBlockMatchesReference(t *testing.T) {
	testExecutorMRoPETextDecoderBlockMatchesReference(t, "qwen3vl")
}

func TestExecutorQwen3VLMoEBlockMatchesReference(t *testing.T) {
	testExecutorMRoPETextDecoderBlockMatchesReference(t, "qwen3vlmoe")
}

func testExecutorMRoPETextDecoderBlockMatchesReference(t *testing.T, architecture string) {
	cudatest.Require(t)
	builder := tensor.NewBuilder()
	spec := model.Spec{CommonSpec: model.CommonSpec{Architecture: architecture, EmbeddingLength: 8, FeedForwardLength: 12,

		RMSNormEpsilon: 1e-6}, AttentionSpec: model.AttentionSpec{HeadCount: 2, HeadCountKV: 1, KeyLength: 4, ValueLength: 4,
		RopeDimensionCount: 4, RopeSections: [4]int32{1, 1, 0, 0},
		RopeFrequencyBase: 10000, RopeScalingType: "linear", RopeScalingFactor: 4},
	}
	if architecture == "qwen3vlmoe" {
		spec.FeedForwardLength = 24
		spec.ExpertCount = 4
		spec.ExpertUsedCount = 2
		spec.ExpertFeedForward = 12
		spec.ExpertWeightsScale = 1.25
	}
	input := builder.Input("input", dtype.F32, tensor.MustShape(8, 2))
	weights := model.LayerGraphWeights{
		AttentionNorm:       builder.Input("attn_norm", dtype.F32, tensor.MustShape(8)),
		AttentionQKV:        builder.Input("attn_qkv", dtype.F32, tensor.MustShape(8, 16)),
		AttentionOutput:     builder.Input("attn_output", dtype.F32, tensor.MustShape(8, 8)),
		AttentionOutputBias: builder.Input("attn_output_bias", dtype.F32, tensor.MustShape(8)),
		FeedForwardNorm:     builder.Input("ffn_norm", dtype.F32, tensor.MustShape(8)),
		FeedForwardGate:     builder.Input("ffn_gate", dtype.F32, tensor.MustShape(8, 12)),
		FeedForwardUp:       builder.Input("ffn_up", dtype.F32, tensor.MustShape(8, 12)),
		FeedForwardDown:     builder.Input("ffn_down", dtype.F32, tensor.MustShape(12, 8)),
	}
	if architecture == "qwen3vl" || architecture == "qwen3vlmoe" {
		weights.AttentionQNorm = builder.Input("attn_q_norm", dtype.F32, tensor.MustShape(4))
		weights.AttentionKNorm = builder.Input("attn_k_norm", dtype.F32, tensor.MustShape(4))
	}
	if architecture == "qwen3vlmoe" {
		weights.AttentionOutputBias = nil
		weights.FeedForwardGate, weights.FeedForwardUp, weights.FeedForwardDown = nil, nil, nil
		weights.FeedForwardRouter = builder.Input("ffn_router", dtype.F32, tensor.MustShape(8, 4))
		weights.FeedForwardGateExperts = builder.Input("ffn_gate_exps", dtype.F32, tensor.MustShape(8, 12, 4))
		weights.FeedForwardUpExperts = builder.Input("ffn_up_exps", dtype.F32, tensor.MustShape(8, 12, 4))
		weights.FeedForwardDownExperts = builder.Input("ffn_down_exps", dtype.F32, tensor.MustShape(12, 8, 4))
	}
	positions := [4][]uint32{{10, 11}, {20, 21}, {30, 31}, {40, 41}}
	result, err := model.BuildDenseBlockCachedForLayerWithMultiPositions(
		builder, input, spec, weights, positions, nil, nil, 0,
	)
	if err != nil {
		t.Fatal(err)
	}
	feeds := map[*tensor.Tensor]reference.Value{input: patternedValue(input.Shape, 3, 0.2, 0)}
	feedNodes := []*tensor.Tensor{
		weights.AttentionNorm, weights.AttentionQKV, weights.AttentionOutput,
		weights.AttentionOutputBias, weights.FeedForwardNorm, weights.FeedForwardGate,
		weights.FeedForwardUp, weights.FeedForwardDown,
		weights.FeedForwardRouter, weights.FeedForwardGateExperts,
		weights.FeedForwardUpExperts, weights.FeedForwardDownExperts,
	}
	for index, node := range feedNodes {
		if node == nil {
			continue
		}
		offset := float32(0)
		if node.Shape.Rank == 1 && node != weights.AttentionOutputBias {
			offset = 1
		}
		feeds[node] = patternedValue(node.Shape, index+5, 0.05, offset)
	}
	if weights.AttentionQNorm != nil {
		feeds[weights.AttentionQNorm] = patternedValue(weights.AttentionQNorm.Shape, 31, 0.03, 1)
		feeds[weights.AttentionKNorm] = patternedValue(weights.AttentionKNorm.Shape, 37, 0.03, 1)
	}
	output := result.Output
	if architecture == "qwen3vl" || architecture == "qwen3vlmoe" {
		deepstack := builder.Input("deepstack_output", dtype.F32, result.Output.Shape)
		feeds[deepstack] = patternedValue(deepstack.Shape, 43, 0.04, 0)
		output = builder.Add(output, deepstack)
	}
	outputs := []*tensor.Tensor{output, result.Key, result.Value}
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
	compare(t, got[output].Data, want[output].Data, 6e-4)
	compare(t, got[result.Key].Data, want[result.Key].Data, 5e-5)
	compare(t, got[result.Value].Data, want[result.Value].Data, 5e-5)
}

func TestExecutorLFM2ShortConvolutionBlockMatchesReference(t *testing.T) {
	cudatest.Require(t)
	builder := tensor.NewBuilder()
	spec := model.Spec{CommonSpec: model.CommonSpec{Architecture: "lfm2", EmbeddingLength: 4, FeedForwardLength: 6,

		RMSNormEpsilon: 1e-6}, AttentionSpec: model.AttentionSpec{HeadCount: 1, HeadCountKV: 1, KeyLength: 4, ValueLength: 4}, RecurrentSpec: model.RecurrentSpec{ShortConvCacheLength: 3},
	}
	input := builder.Input("input", dtype.F32, tensor.MustShape(4, 2))
	weights := model.LayerGraphWeights{
		AttentionNorm:   builder.Input("operator_norm", dtype.F32, tensor.MustShape(4)),
		ShortConvInput:  builder.Input("conv_in", dtype.F32, tensor.MustShape(4, 12)),
		ShortConvKernel: builder.Input("conv_kernel", dtype.F32, tensor.MustShape(3, 4)),
		ShortConvOutput: builder.Input("conv_out", dtype.F32, tensor.MustShape(4, 4)),
		FeedForwardNorm: builder.Input("ffn_norm", dtype.F32, tensor.MustShape(4)),
		FeedForwardGate: builder.Input("ffn_gate", dtype.F32, tensor.MustShape(4, 6)),
		FeedForwardUp:   builder.Input("ffn_up", dtype.F32, tensor.MustShape(4, 6)),
		FeedForwardDown: builder.Input("ffn_down", dtype.F32, tensor.MustShape(6, 4)),
	}
	state := builder.Input("conv_state", dtype.F32, tensor.MustShape(2, 4))
	reserved := builder.Input("reserved", dtype.F32, tensor.MustShape(1))
	result, err := model.BuildLFM2BlockCached(
		builder, input, spec, weights, []uint32{0, 1}, true, state, reserved, 0,
	)
	if err != nil {
		t.Fatal(err)
	}
	feeds := map[*tensor.Tensor]reference.Value{
		input:    patternedValue(input.Shape, 3, 0.2, 0),
		state:    patternedValue(state.Shape, 5, 0.1, 0),
		reserved: patternedValue(reserved.Shape, 7, 0.1, 0),
	}
	for index, node := range []*tensor.Tensor{
		weights.ShortConvInput, weights.ShortConvKernel, weights.ShortConvOutput,
		weights.FeedForwardGate, weights.FeedForwardUp, weights.FeedForwardDown,
	} {
		feeds[node] = patternedValue(node.Shape, index+11, 0.08, 0)
	}
	feeds[weights.AttentionNorm] = patternedValue(weights.AttentionNorm.Shape, 21, 0.03, 1)
	feeds[weights.FeedForwardNorm] = patternedValue(weights.FeedForwardNorm.Shape, 23, 0.03, 1)
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
	compare(t, got[result.Output].Data, want[result.Output].Data, 5e-4)
	compare(t, got[result.Key].Data, want[result.Key].Data, 1e-6)
	compare(t, got[result.Value].Data, want[result.Value].Data, 0)
}

func TestExecutorLFM2CenteredShortConvolutionBlockMatchesReference(t *testing.T) {
	cudatest.Require(t)
	builder := tensor.NewBuilder()
	spec := model.Spec{CommonSpec: model.CommonSpec{Architecture: "lfm2", EmbeddingLength: 4, FeedForwardLength: 6,

		RMSNormEpsilon: 1e-6}, AttentionSpec: model.AttentionSpec{HeadCount: 1, HeadCountKV: 1, KeyLength: 4, ValueLength: 4,
		NonCausalAttention: true}, RecurrentSpec: model.RecurrentSpec{ShortConvCacheLength: 3},
	}
	input := builder.Input("input", dtype.F32, tensor.MustShape(4, 3))
	weights := model.LayerGraphWeights{
		AttentionNorm:   builder.Input("operator_norm", dtype.F32, tensor.MustShape(4)),
		ShortConvInput:  builder.Input("conv_in", dtype.F32, tensor.MustShape(4, 12)),
		ShortConvKernel: builder.Input("conv_kernel", dtype.F32, tensor.MustShape(3, 4)),
		ShortConvOutput: builder.Input("conv_out", dtype.F32, tensor.MustShape(4, 4)),
		FeedForwardNorm: builder.Input("ffn_norm", dtype.F32, tensor.MustShape(4)),
		FeedForwardGate: builder.Input("ffn_gate", dtype.F32, tensor.MustShape(4, 6)),
		FeedForwardUp:   builder.Input("ffn_up", dtype.F32, tensor.MustShape(4, 6)),
		FeedForwardDown: builder.Input("ffn_down", dtype.F32, tensor.MustShape(6, 4)),
	}
	state := builder.Input("conv_state", dtype.F32, tensor.MustShape(2, 4))
	reserved := builder.Input("reserved", dtype.F32, tensor.MustShape(1))
	result, err := model.BuildLFM2BlockCached(
		builder, input, spec, weights, []uint32{0, 1, 2}, true, state, reserved, 0,
	)
	if err != nil {
		t.Fatal(err)
	}
	feeds := map[*tensor.Tensor]reference.Value{
		input:    patternedValue(input.Shape, 3, 0.2, 0),
		state:    patternedValue(state.Shape, 5, 0.1, 0),
		reserved: patternedValue(reserved.Shape, 7, 0.1, 0),
	}
	for index, node := range []*tensor.Tensor{
		weights.ShortConvInput, weights.ShortConvKernel, weights.ShortConvOutput,
		weights.FeedForwardGate, weights.FeedForwardUp, weights.FeedForwardDown,
	} {
		feeds[node] = patternedValue(node.Shape, index+11, 0.08, 0)
	}
	feeds[weights.AttentionNorm] = patternedValue(weights.AttentionNorm.Shape, 21, 0.03, 1)
	feeds[weights.FeedForwardNorm] = patternedValue(weights.FeedForwardNorm.Shape, 23, 0.03, 1)
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
	compare(t, got[result.Output].Data, want[result.Output].Data, 5e-4)
	compare(t, got[result.Key].Data, want[result.Key].Data, 1e-6)
	compare(t, got[result.Value].Data, want[result.Value].Data, 0)
}

func TestExecutorLFM2MoEShortConvolutionBlockMatchesReference(t *testing.T) {
	cudatest.Require(t)
	builder := tensor.NewBuilder()
	spec := model.Spec{CommonSpec: model.CommonSpec{Architecture: "lfm2moe", BlockCount: 3,
		EmbeddingLength: 4, FeedForwardLength: 6,

		RMSNormEpsilon: 1e-6}, AttentionSpec: model.AttentionSpec{HeadCount: 1, HeadCountKV: 1, KeyLength: 4, ValueLength: 4}, MoESpec: model.MoESpec{LeadingDenseBlocks: 1,

		ExpertCount: 4, ExpertUsedCount: 2, ExpertFeedForward: 6,
		ExpertWeightsScale: 1.25, ExpertGatingFunc: 2}, RecurrentSpec: model.RecurrentSpec{ShortConvCacheLength: 3},
	}
	input := builder.Input("input", dtype.F32, tensor.MustShape(4, 2))
	weights := model.LayerGraphWeights{
		AttentionNorm:          builder.Input("operator_norm", dtype.F32, tensor.MustShape(4)),
		ShortConvInput:         builder.Input("conv_in", dtype.F32, tensor.MustShape(4, 12)),
		ShortConvKernel:        builder.Input("conv_kernel", dtype.F32, tensor.MustShape(3, 4)),
		ShortConvOutput:        builder.Input("conv_out", dtype.F32, tensor.MustShape(4, 4)),
		FeedForwardNorm:        builder.Input("ffn_norm", dtype.F32, tensor.MustShape(4)),
		FeedForwardRouter:      builder.Input("router", dtype.F32, tensor.MustShape(4, 4)),
		FeedForwardGateExperts: builder.Input("gate_exps", dtype.F32, tensor.MustShape(4, 6, 4)),
		FeedForwardUpExperts:   builder.Input("up_exps", dtype.F32, tensor.MustShape(4, 6, 4)),
		FeedForwardDownExperts: builder.Input("down_exps", dtype.F32, tensor.MustShape(6, 4, 4)),
		FeedForwardExpertBias:  builder.Input("correction", dtype.F32, tensor.MustShape(4)),
	}
	state := builder.Input("conv_state", dtype.F32, tensor.MustShape(2, 4))
	reserved := builder.Input("reserved", dtype.F32, tensor.MustShape(1))
	result, err := model.BuildLFM2BlockCached(
		builder, input, spec, weights, []uint32{0, 1}, true, state, reserved, 2,
	)
	if err != nil {
		t.Fatal(err)
	}
	feeds := map[*tensor.Tensor]reference.Value{
		input:    patternedValue(input.Shape, 3, 0.2, 0),
		state:    patternedValue(state.Shape, 5, 0.1, 0),
		reserved: patternedValue(reserved.Shape, 7, 0.1, 0),
	}
	for index, node := range []*tensor.Tensor{
		weights.ShortConvInput, weights.ShortConvKernel, weights.ShortConvOutput,
		weights.FeedForwardRouter, weights.FeedForwardGateExperts,
		weights.FeedForwardUpExperts, weights.FeedForwardDownExperts,
	} {
		feeds[node] = patternedValue(node.Shape, index+11, 0.08, 0)
	}
	feeds[weights.AttentionNorm] = patternedValue(weights.AttentionNorm.Shape, 21, 0.03, 1)
	feeds[weights.FeedForwardNorm] = patternedValue(weights.FeedForwardNorm.Shape, 23, 0.03, 1)
	feeds[weights.FeedForwardExpertBias] = patternedValue(weights.FeedForwardExpertBias.Shape, 29, 0.02, 0)
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
	compare(t, got[result.Key].Data, want[result.Key].Data, 1e-6)
	compare(t, got[result.Value].Data, want[result.Value].Data, 0)
}

func TestExecutorSoftcappedWindowAttentionMatchesReference(t *testing.T) {
	cudatest.Require(t)
	builder := tensor.NewBuilder()
	query := builder.Input("query", dtype.F32, tensor.MustShape(4, 2, 2))
	key := builder.Input("key", dtype.F32, tensor.MustShape(4, 1, 5))
	value := builder.Input("value", dtype.F32, tensor.MustShape(4, 1, 5))
	output := builder.AttentionWindowSoftcappedWithOffset(
		query, key, value, 0.5, 1.75, true, 3, 3,
	)
	if err := builder.Err(); err != nil {
		t.Fatal(err)
	}
	feeds := map[*tensor.Tensor]reference.Value{
		query: patternedValue(query.Shape, 13, 0.8, 0),
		key:   patternedValue(key.Shape, 17, 0.7, 0),
		value: patternedValue(value.Shape, 19, 0.2, 0),
	}
	want, err := reference.Execute([]*tensor.Tensor{output}, feeds)
	if err != nil {
		t.Fatal(err)
	}
	cuda, err := New(0)
	if err != nil {
		t.Fatal(err)
	}
	defer cuda.Close()
	got, err := cuda.Execute(context.Background(), []*tensor.Tensor{output}, feeds)
	if err != nil {
		t.Fatal(err)
	}
	compare(t, got[output].Data, want[output].Data, 3e-5)
}

func TestExecutorSymmetricWindowAttentionMatchesReference(t *testing.T) {
	cudatest.Require(t)
	builder := tensor.NewBuilder()
	query := builder.Input("query", dtype.F32, tensor.MustShape(4, 2, 7))
	key := builder.Input("key", dtype.F32, tensor.MustShape(4, 1, 7))
	value := builder.Input("value", dtype.F32, tensor.MustShape(3, 1, 7))
	output := builder.AttentionSymmetricWindow(query, key, value, 0.5, 4)
	if err := builder.Err(); err != nil {
		t.Fatal(err)
	}
	feeds := map[*tensor.Tensor]reference.Value{
		query: patternedValue(query.Shape, 13, 0.8, 0),
		key:   patternedValue(key.Shape, 17, 0.7, 0),
		value: patternedValue(value.Shape, 19, 0.2, 0),
	}
	want, err := reference.Execute([]*tensor.Tensor{output}, feeds)
	if err != nil {
		t.Fatal(err)
	}
	cuda, err := New(0)
	if err != nil {
		t.Fatal(err)
	}
	defer cuda.Close()
	got, err := cuda.Execute(context.Background(), []*tensor.Tensor{output}, feeds)
	if err != nil {
		t.Fatal(err)
	}
	compare(t, got[output].Data, want[output].Data, 3e-5)
}

func TestExecutorChunkedWindowAttentionMatchesReference(t *testing.T) {
	cudatest.Require(t)
	builder := tensor.NewBuilder()
	query := builder.Input("query", dtype.F32, tensor.MustShape(4, 2, 4))
	key := builder.Input("key", dtype.F32, tensor.MustShape(4, 1, 10))
	value := builder.Input("value", dtype.F32, tensor.MustShape(3, 1, 10))
	output := builder.AttentionChunkedWindowWithOffset(query, key, value, 0.5, true, 6, 4)
	feeds := map[*tensor.Tensor]reference.Value{
		query: patternedValue(query.Shape, 13, 0.8, 0),
		key:   patternedValue(key.Shape, 17, 0.7, 0),
		value: patternedValue(value.Shape, 19, 0.2, 0),
	}
	want, err := reference.Execute([]*tensor.Tensor{output}, feeds)
	if err != nil {
		t.Fatal(err)
	}
	cuda, err := New(0)
	if err != nil {
		t.Fatal(err)
	}
	defer cuda.Close()
	got, err := cuda.Execute(context.Background(), []*tensor.Tensor{output}, feeds)
	if err != nil {
		t.Fatal(err)
	}
	compare(t, got[output].Data, want[output].Data, 3e-5)
}

func TestExecutorT5RelativeBiasAttentionMatchesReference(t *testing.T) {
	cudatest.Require(t)
	builder := tensor.NewBuilder()
	query := builder.Input("query", dtype.F32, tensor.MustShape(3, 2, 4))
	key := builder.Input("key", dtype.F32, tensor.MustShape(3, 2, 4))
	value := builder.Input("value", dtype.F32, tensor.MustShape(3, 2, 4))
	bias := builder.Input("bias", dtype.F32, tensor.MustShape(2, 32))
	output := builder.AttentionWithRelativeBias(query, key, value, bias, 1)
	causal := builder.AttentionWithRelativeBiasAndOffset(query, key, value, bias, 1, 0)
	relu := builder.ReLU(value)
	if err := builder.Err(); err != nil {
		t.Fatal(err)
	}
	feeds := map[*tensor.Tensor]reference.Value{
		query: patternedValue(query.Shape, 3, 0.08, 0),
		key:   patternedValue(key.Shape, 5, 0.07, 0),
		value: patternedValue(value.Shape, 7, 0.1, 0),
		bias:  patternedValue(bias.Shape, 11, 0.04, -0.2),
	}
	outputs := []*tensor.Tensor{output, causal, relu}
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
	for _, item := range outputs {
		compare(t, got[item].Data, want[item].Data, 3e-5)
	}
}

func TestExecutorT5DecoderBlockMatchesReference(t *testing.T) {
	cudatest.Require(t)
	builder := tensor.NewBuilder()
	spec := model.Spec{CommonSpec: model.CommonSpec{Architecture: "t5", EmbeddingLength: 4, FeedForwardLength: 6,

		RMSNormEpsilon: 1e-5}, AttentionSpec: model.AttentionSpec{HeadCount: 2, HeadCountKV: 2, KeyLength: 2, ValueLength: 2}, EncoderSpec: model.EncoderSpec{RelativeBuckets: 4},
	}
	input := builder.Input("input", dtype.F32, tensor.MustShape(4, 2))
	encoder := builder.Input("encoder", dtype.F32, tensor.MustShape(4, 3))
	weights := model.LayerGraphWeights{
		AttentionNorm:         builder.Input("attn_norm", dtype.F32, tensor.MustShape(4)),
		AttentionQ:            builder.Input("attn_q", dtype.F32, tensor.MustShape(4, 4)),
		AttentionK:            builder.Input("attn_k", dtype.F32, tensor.MustShape(4, 4)),
		AttentionV:            builder.Input("attn_v", dtype.F32, tensor.MustShape(4, 4)),
		AttentionOutput:       builder.Input("attn_o", dtype.F32, tensor.MustShape(4, 4)),
		AttentionRelativeBias: builder.Input("attn_rel_b", dtype.F32, tensor.MustShape(2, 4)),
		CrossAttentionNorm:    builder.Input("cross_norm", dtype.F32, tensor.MustShape(4)),
		CrossAttentionQ:       builder.Input("cross_q", dtype.F32, tensor.MustShape(4, 4)),
		CrossAttentionK:       builder.Input("cross_k", dtype.F32, tensor.MustShape(4, 4)),
		CrossAttentionV:       builder.Input("cross_v", dtype.F32, tensor.MustShape(4, 4)),
		CrossAttentionOutput:  builder.Input("cross_o", dtype.F32, tensor.MustShape(4, 4)),
		FeedForwardNorm:       builder.Input("ffn_norm", dtype.F32, tensor.MustShape(4)),
		FeedForwardUp:         builder.Input("ffn_up", dtype.F32, tensor.MustShape(4, 6)),
		FeedForwardDown:       builder.Input("ffn_down", dtype.F32, tensor.MustShape(6, 4)),
	}
	result, err := model.BuildT5DecoderBlockCached(builder, input, encoder, spec, weights, nil, nil, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	feeds := map[*tensor.Tensor]reference.Value{
		input:                         patternedValue(input.Shape, 3, 0.1, -0.2),
		encoder:                       patternedValue(encoder.Shape, 5, 0.08, -0.1),
		weights.AttentionNorm:         patternedValue(weights.AttentionNorm.Shape, 7, 0.03, 0.9),
		weights.AttentionQ:            patternedValue(weights.AttentionQ.Shape, 11, 0.04, -0.2),
		weights.AttentionK:            patternedValue(weights.AttentionK.Shape, 13, 0.04, -0.2),
		weights.AttentionV:            patternedValue(weights.AttentionV.Shape, 17, 0.04, -0.2),
		weights.AttentionOutput:       patternedValue(weights.AttentionOutput.Shape, 19, 0.04, -0.2),
		weights.AttentionRelativeBias: patternedValue(weights.AttentionRelativeBias.Shape, 23, 0.03, -0.1),
		weights.CrossAttentionNorm:    patternedValue(weights.CrossAttentionNorm.Shape, 29, 0.03, 0.9),
		weights.CrossAttentionQ:       patternedValue(weights.CrossAttentionQ.Shape, 31, 0.04, -0.2),
		weights.CrossAttentionK:       patternedValue(weights.CrossAttentionK.Shape, 37, 0.04, -0.2),
		weights.CrossAttentionV:       patternedValue(weights.CrossAttentionV.Shape, 41, 0.04, -0.2),
		weights.CrossAttentionOutput:  patternedValue(weights.CrossAttentionOutput.Shape, 43, 0.04, -0.2),
		weights.FeedForwardNorm:       patternedValue(weights.FeedForwardNorm.Shape, 47, 0.03, 0.9),
		weights.FeedForwardUp:         patternedValue(weights.FeedForwardUp.Shape, 53, 0.04, -0.2),
		weights.FeedForwardDown:       patternedValue(weights.FeedForwardDown.Shape, 59, 0.04, -0.2),
	}
	outputs := []*tensor.Tensor{
		result.Output, result.Key, result.Value,
		result.FixedStates["cross_key"], result.FixedStates["cross_value"],
	}
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
	for _, output := range outputs {
		compare(t, got[output].Data, want[output].Data, 8e-4)
	}
}

func TestExecutorAudioConvolutionPrimitivesMatchReference(t *testing.T) {
	cudatest.Require(t)
	builder := tensor.NewBuilder()
	input := builder.Input("input", dtype.F32, tensor.MustShape(4, 5))
	denseWeight := builder.Input("dense_weight", dtype.F32, tensor.MustShape(3, 4, 6))
	denseBias := builder.Input("dense_bias", dtype.F32, tensor.MustShape(1, 6))
	dense := builder.Conv1DSame(input, denseWeight, denseBias, false)
	depthWeight := builder.Input("depth_weight", dtype.F32, tensor.MustShape(3, 1, 4))
	depthBias := builder.Input("depth_bias", dtype.F32, tensor.MustShape(1, 4))
	depthwise := builder.Conv1DSame(input, depthWeight, depthBias, true)
	normWeight := builder.Input("norm_weight", dtype.F32, tensor.MustShape(1, 4))
	normBias := builder.Input("norm_bias", dtype.F32, tensor.MustShape(1, 4))
	normalized := builder.GroupNorm(input, normWeight, normBias, 2, 1e-5)
	feeds := map[*tensor.Tensor]reference.Value{
		input:       patternedValue(input.Shape, 3, 0.1, -0.4),
		denseWeight: patternedValue(denseWeight.Shape, 5, 0.04, -0.2),
		denseBias:   patternedValue(denseBias.Shape, 7, 0.03, -0.1),
		depthWeight: patternedValue(depthWeight.Shape, 11, 0.04, -0.2),
		depthBias:   patternedValue(depthBias.Shape, 13, 0.03, -0.1),
		normWeight:  patternedValue(normWeight.Shape, 17, 0.03, 0.9),
		normBias:    patternedValue(normBias.Shape, 19, 0.02, -0.05),
	}
	outputs := []*tensor.Tensor{dense, depthwise, normalized}
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
	for _, output := range outputs {
		compare(t, got[output].Data, want[output].Data, 3e-5)
	}
}
