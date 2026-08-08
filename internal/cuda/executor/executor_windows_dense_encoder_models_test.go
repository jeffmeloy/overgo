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

func TestExecutorDenseSmolLM3NoRoPEBlockMatchesReference(t *testing.T) {
	cudatest.Require(t)
	builder := tensor.NewBuilder()
	spec := model.Spec{CommonSpec: model.CommonSpec{Architecture: "smollm3",
		BlockCount:        4,
		EmbeddingLength:   8,
		FeedForwardLength: 12,

		RMSNormEpsilon: 1e-6}, AttentionSpec: model.AttentionSpec{HeadCount: 2,
		HeadCountKV:       1,
		KeyLength:         4,
		ValueLength:       4,
		RopeFrequencyBase: 10000,

		AttentionScale:  0.25,
		NoRopeLayerStep: 4},
	}
	input := builder.Input("input", dtype.F32, tensor.MustShape(8, 2))
	weights := model.LayerGraphWeights{
		AttentionNorm:   builder.Input("attn_norm", dtype.F32, tensor.MustShape(8)),
		AttentionQ:      builder.Input("attn_q", dtype.F32, tensor.MustShape(8, 8)),
		AttentionK:      builder.Input("attn_k", dtype.F32, tensor.MustShape(8, 4)),
		AttentionV:      builder.Input("attn_v", dtype.F32, tensor.MustShape(8, 4)),
		AttentionOutput: builder.Input("attn_output", dtype.F32, tensor.MustShape(8, 8)),
		FeedForwardNorm: builder.Input("ffn_norm", dtype.F32, tensor.MustShape(8)),
		FeedForwardGate: builder.Input("ffn_gate", dtype.F32, tensor.MustShape(8, 12)),
		FeedForwardUp:   builder.Input("ffn_up", dtype.F32, tensor.MustShape(8, 12)),
		FeedForwardDown: builder.Input("ffn_down", dtype.F32, tensor.MustShape(12, 8)),
	}
	result, err := model.BuildDenseBlockCachedForLayer(
		builder,
		input,
		spec,
		weights,
		[]uint32{0, 1},
		nil,
		nil,
		3,
	)
	if err != nil {
		t.Fatal(err)
	}
	feeds := map[*tensor.Tensor]reference.Value{
		input: patternedValue(input.Shape, 1, 0.2, 0),
	}
	for index, node := range []*tensor.Tensor{
		weights.AttentionQ,
		weights.AttentionK,
		weights.AttentionV,
		weights.AttentionOutput,
		weights.FeedForwardGate,
		weights.FeedForwardUp,
		weights.FeedForwardDown,
	} {
		feeds[node] = patternedValue(node.Shape, index+2, 0.06, 0)
	}
	for index, node := range []*tensor.Tensor{
		weights.AttentionNorm,
		weights.FeedForwardNorm,
	} {
		feeds[node] = patternedValue(node.Shape, index+11, 0.03, 1)
	}
	want, err := reference.Execute([]*tensor.Tensor{result.Output}, feeds)
	if err != nil {
		t.Fatal(err)
	}
	cuda := newFixtureExecutor(t)
	got, err := cuda.Execute(context.Background(), []*tensor.Tensor{result.Output}, feeds)
	if err != nil {
		t.Fatal(err)
	}
	compare(t, got[result.Output].Data, want[result.Output].Data, 3e-4)
}

func TestExecutorDenseMiniCPMBlockMatchesReference(t *testing.T) {
	cudatest.Require(t)
	builder := tensor.NewBuilder()
	spec := model.Spec{CommonSpec: model.CommonSpec{Architecture: "minicpm",
		EmbeddingLength:   8,
		FeedForwardLength: 12,

		RMSNormEpsilon: 1e-6,
		ResidualScale:  0.25}, AttentionSpec: model.AttentionSpec{HeadCount: 2,
		HeadCountKV:       1,
		KeyLength:         4,
		ValueLength:       4,
		RopeFrequencyBase: 10000},
	}
	input := builder.Input("input", dtype.F32, tensor.MustShape(8, 2))
	weights := model.LayerGraphWeights{
		AttentionNorm:       builder.Input("attn_norm", dtype.F32, tensor.MustShape(8)),
		AttentionQ:          builder.Input("attn_q", dtype.F32, tensor.MustShape(8, 8)),
		AttentionK:          builder.Input("attn_k", dtype.F32, tensor.MustShape(8, 4)),
		AttentionV:          builder.Input("attn_v", dtype.F32, tensor.MustShape(8, 4)),
		AttentionOutput:     builder.Input("attn_output", dtype.F32, tensor.MustShape(8, 8)),
		AttentionOutputBias: builder.Input("attn_output_bias", dtype.F32, tensor.MustShape(8)),
		FeedForwardNorm:     builder.Input("ffn_norm", dtype.F32, tensor.MustShape(8)),
		FeedForwardGate:     builder.Input("ffn_gate", dtype.F32, tensor.MustShape(8, 12)),
		FeedForwardUp:       builder.Input("ffn_up", dtype.F32, tensor.MustShape(8, 12)),
		FeedForwardDown:     builder.Input("ffn_down", dtype.F32, tensor.MustShape(12, 8)),
		FeedForwardGateBias: builder.Input("ffn_gate_bias", dtype.F32, tensor.MustShape(12)),
		FeedForwardUpBias:   builder.Input("ffn_up_bias", dtype.F32, tensor.MustShape(12)),
		FeedForwardDownBias: builder.Input("ffn_down_bias", dtype.F32, tensor.MustShape(8)),
	}
	result, err := model.BuildDenseBlockCachedForLayer(
		builder,
		input,
		spec,
		weights,
		[]uint32{0, 1},
		nil,
		nil,
		0,
	)
	if err != nil {
		t.Fatal(err)
	}
	feeds := map[*tensor.Tensor]reference.Value{
		input: patternedValue(input.Shape, 1, 0.2, 0),
	}
	for index, node := range []*tensor.Tensor{
		weights.AttentionQ,
		weights.AttentionK,
		weights.AttentionV,
		weights.AttentionOutput,
		weights.FeedForwardGate,
		weights.FeedForwardUp,
		weights.FeedForwardDown,
	} {
		feeds[node] = patternedValue(node.Shape, index+2, 0.06, 0)
	}
	for index, node := range []*tensor.Tensor{
		weights.AttentionNorm,
		weights.FeedForwardNorm,
	} {
		feeds[node] = patternedValue(node.Shape, index+11, 0.03, 1)
	}
	for index, node := range []*tensor.Tensor{
		weights.AttentionOutputBias,
		weights.FeedForwardGateBias,
		weights.FeedForwardUpBias,
		weights.FeedForwardDownBias,
	} {
		feeds[node] = patternedValue(node.Shape, index+20, 0.02, -0.01)
	}
	want, err := reference.Execute([]*tensor.Tensor{result.Output}, feeds)
	if err != nil {
		t.Fatal(err)
	}
	cuda := newFixtureExecutor(t)
	got, err := cuda.Execute(context.Background(), []*tensor.Tensor{result.Output}, feeds)
	if err != nil {
		t.Fatal(err)
	}
	compare(t, got[result.Output].Data, want[result.Output].Data, 3e-4)
}

func TestExecutorDenseGraniteNoRoPEBlockMatchesReference(t *testing.T) {
	cudatest.Require(t)
	builder := tensor.NewBuilder()
	spec := model.Spec{CommonSpec: model.CommonSpec{Architecture: "granite",
		EmbeddingLength:   8,
		FeedForwardLength: 12,

		RMSNormEpsilon: 1e-6,

		ResidualScale: 0.5}, AttentionSpec: model.AttentionSpec{HeadCount: 2,
		HeadCountKV:       1,
		KeyLength:         4,
		ValueLength:       4,
		RopeFrequencyBase: 10000,

		AttentionScale: 0.25,

		RopeDisabled: true},
	}
	input := builder.Input("input", dtype.F32, tensor.MustShape(8, 2))
	weights := model.LayerGraphWeights{
		AttentionNorm:   builder.Input("attn_norm", dtype.F32, tensor.MustShape(8)),
		AttentionQ:      builder.Input("attn_q", dtype.F32, tensor.MustShape(8, 8)),
		AttentionK:      builder.Input("attn_k", dtype.F32, tensor.MustShape(8, 4)),
		AttentionV:      builder.Input("attn_v", dtype.F32, tensor.MustShape(8, 4)),
		AttentionOutput: builder.Input("attn_output", dtype.F32, tensor.MustShape(8, 8)),
		FeedForwardNorm: builder.Input("ffn_norm", dtype.F32, tensor.MustShape(8)),
		FeedForwardGate: builder.Input("ffn_gate", dtype.F32, tensor.MustShape(8, 12)),
		FeedForwardUp:   builder.Input("ffn_up", dtype.F32, tensor.MustShape(8, 12)),
		FeedForwardDown: builder.Input("ffn_down", dtype.F32, tensor.MustShape(12, 8)),
	}
	result, err := model.BuildDenseBlockCachedForLayer(
		builder,
		input,
		spec,
		weights,
		[]uint32{0, 1},
		nil,
		nil,
		0,
	)
	if err != nil {
		t.Fatal(err)
	}
	feeds := map[*tensor.Tensor]reference.Value{
		input: patternedValue(input.Shape, 1, 0.2, 0),
	}
	for index, node := range []*tensor.Tensor{
		weights.AttentionQ,
		weights.AttentionK,
		weights.AttentionV,
		weights.AttentionOutput,
		weights.FeedForwardGate,
		weights.FeedForwardUp,
		weights.FeedForwardDown,
	} {
		feeds[node] = patternedValue(node.Shape, index+2, 0.06, 0)
	}
	for index, node := range []*tensor.Tensor{
		weights.AttentionNorm,
		weights.FeedForwardNorm,
	} {
		feeds[node] = patternedValue(node.Shape, index+11, 0.03, 1)
	}
	want, err := reference.Execute([]*tensor.Tensor{result.Output}, feeds)
	if err != nil {
		t.Fatal(err)
	}
	cuda := newFixtureExecutor(t)
	got, err := cuda.Execute(context.Background(), []*tensor.Tensor{result.Output}, feeds)
	if err != nil {
		t.Fatal(err)
	}
	compare(t, got[result.Output].Data, want[result.Output].Data, 3e-4)
}

func TestExecutorDenseMaincoderBlockMatchesReference(t *testing.T) {
	cudatest.Require(t)
	builder := tensor.NewBuilder()
	spec := model.Spec{CommonSpec: model.CommonSpec{Architecture: "maincoder",
		EmbeddingLength:   8,
		FeedForwardLength: 12,

		RMSNormEpsilon: 1e-6}, AttentionSpec: model.AttentionSpec{HeadCount: 2,
		HeadCountKV:       1,
		KeyLength:         4,
		ValueLength:       4,
		RopeFrequencyBase: 10000},
	}
	input := builder.Input("input", dtype.F32, tensor.MustShape(8, 2))
	weights := model.LayerGraphWeights{
		AttentionNorm:   builder.Input("attn_norm", dtype.F32, tensor.MustShape(8)),
		AttentionQ:      builder.Input("attn_q", dtype.F32, tensor.MustShape(8, 8)),
		AttentionK:      builder.Input("attn_k", dtype.F32, tensor.MustShape(8, 4)),
		AttentionV:      builder.Input("attn_v", dtype.F32, tensor.MustShape(8, 4)),
		AttentionOutput: builder.Input("attn_output", dtype.F32, tensor.MustShape(8, 8)),
		AttentionQNorm:  builder.Input("attn_q_norm", dtype.F32, tensor.MustShape(4)),
		AttentionKNorm:  builder.Input("attn_k_norm", dtype.F32, tensor.MustShape(4)),
		FeedForwardNorm: builder.Input("ffn_norm", dtype.F32, tensor.MustShape(8)),
		FeedForwardGate: builder.Input("ffn_gate", dtype.F32, tensor.MustShape(8, 12)),
		FeedForwardUp:   builder.Input("ffn_up", dtype.F32, tensor.MustShape(8, 12)),
		FeedForwardDown: builder.Input("ffn_down", dtype.F32, tensor.MustShape(12, 8)),
	}
	result, err := model.BuildDenseBlockCachedForLayer(
		builder,
		input,
		spec,
		weights,
		[]uint32{0, 1},
		nil,
		nil,
		0,
	)
	if err != nil {
		t.Fatal(err)
	}
	feeds := map[*tensor.Tensor]reference.Value{
		input: patternedValue(input.Shape, 1, 0.2, 0),
	}
	for index, node := range []*tensor.Tensor{
		weights.AttentionQ,
		weights.AttentionK,
		weights.AttentionV,
		weights.AttentionOutput,
		weights.FeedForwardGate,
		weights.FeedForwardUp,
		weights.FeedForwardDown,
	} {
		feeds[node] = patternedValue(node.Shape, index+2, 0.06, 0)
	}
	for index, node := range []*tensor.Tensor{
		weights.AttentionNorm,
		weights.AttentionQNorm,
		weights.AttentionKNorm,
		weights.FeedForwardNorm,
	} {
		feeds[node] = patternedValue(node.Shape, index+11, 0.03, 1)
	}
	want, err := reference.Execute([]*tensor.Tensor{result.Output}, feeds)
	if err != nil {
		t.Fatal(err)
	}
	cuda := newFixtureExecutor(t)
	got, err := cuda.Execute(context.Background(), []*tensor.Tensor{result.Output}, feeds)
	if err != nil {
		t.Fatal(err)
	}
	compare(t, got[result.Output].Data, want[result.Output].Data, 3e-4)
}

func TestExecutorDenseMistral3BlockMatchesReference(t *testing.T) {
	cudatest.Require(t)
	builder := tensor.NewBuilder()
	spec := model.Spec{CommonSpec: model.CommonSpec{Architecture: "mistral3",
		EmbeddingLength:   8,
		FeedForwardLength: 12,

		RMSNormEpsilon: 1e-6}, AttentionSpec: model.AttentionSpec{HeadCount: 2,
		HeadCountKV:       1,
		KeyLength:         4,
		ValueLength:       4,
		RopeFrequencyBase: 10000},
	}
	input := builder.Input("input", dtype.F32, tensor.MustShape(8, 2))
	weights := model.LayerGraphWeights{
		AttentionNorm:       builder.Input("attn_norm", dtype.F32, tensor.MustShape(8)),
		AttentionQ:          builder.Input("attn_q", dtype.F32, tensor.MustShape(8, 8)),
		AttentionK:          builder.Input("attn_k", dtype.F32, tensor.MustShape(8, 4)),
		AttentionV:          builder.Input("attn_v", dtype.F32, tensor.MustShape(8, 4)),
		AttentionOutput:     builder.Input("attn_output", dtype.F32, tensor.MustShape(8, 8)),
		AttentionOutputBias: builder.Input("attn_output_bias", dtype.F32, tensor.MustShape(8)),
		FeedForwardNorm:     builder.Input("ffn_norm", dtype.F32, tensor.MustShape(8)),
		FeedForwardGate:     builder.Input("ffn_gate", dtype.F32, tensor.MustShape(8, 12)),
		FeedForwardUp:       builder.Input("ffn_up", dtype.F32, tensor.MustShape(8, 12)),
		FeedForwardDown:     builder.Input("ffn_down", dtype.F32, tensor.MustShape(12, 8)),
		FeedForwardGateBias: builder.Input("ffn_gate_bias", dtype.F32, tensor.MustShape(12)),
		FeedForwardUpBias:   builder.Input("ffn_up_bias", dtype.F32, tensor.MustShape(12)),
		FeedForwardDownBias: builder.Input("ffn_down_bias", dtype.F32, tensor.MustShape(8)),
	}
	result, err := model.BuildDenseBlockCachedForLayer(
		builder,
		input,
		spec,
		weights,
		[]uint32{0, 1},
		nil,
		nil,
		0,
	)
	if err != nil {
		t.Fatal(err)
	}
	feeds := map[*tensor.Tensor]reference.Value{
		input: patternedValue(input.Shape, 1, 0.2, 0),
	}
	for index, node := range []*tensor.Tensor{
		weights.AttentionQ,
		weights.AttentionK,
		weights.AttentionV,
		weights.AttentionOutput,
		weights.FeedForwardGate,
		weights.FeedForwardUp,
		weights.FeedForwardDown,
	} {
		feeds[node] = patternedValue(node.Shape, index+2, 0.06, 0)
	}
	for index, node := range []*tensor.Tensor{
		weights.AttentionNorm,
		weights.FeedForwardNorm,
	} {
		feeds[node] = patternedValue(node.Shape, index+11, 0.03, 1)
	}
	for index, node := range []*tensor.Tensor{
		weights.AttentionOutputBias,
		weights.FeedForwardGateBias,
		weights.FeedForwardUpBias,
		weights.FeedForwardDownBias,
	} {
		feeds[node] = patternedValue(node.Shape, index+20, 0.02, -0.01)
	}
	want, err := reference.Execute([]*tensor.Tensor{result.Output}, feeds)
	if err != nil {
		t.Fatal(err)
	}
	cuda := newFixtureExecutor(t)
	got, err := cuda.Execute(context.Background(), []*tensor.Tensor{result.Output}, feeds)
	if err != nil {
		t.Fatal(err)
	}
	compare(t, got[result.Output].Data, want[result.Output].Data, 3e-4)
}

func TestExecutorDenseOrionBlockMatchesReference(t *testing.T) {
	cudatest.Require(t)
	builder := tensor.NewBuilder()
	spec := model.Spec{CommonSpec: model.CommonSpec{Architecture: "orion",
		EmbeddingLength:   8,
		FeedForwardLength: 12,

		LayerNormEpsilon: 1e-5}, AttentionSpec: model.AttentionSpec{HeadCount: 2,
		HeadCountKV:       1,
		KeyLength:         4,
		ValueLength:       4,
		RopeFrequencyBase: 10000},
	}
	input := builder.Input("input", dtype.F32, tensor.MustShape(8, 2))
	weights := model.LayerGraphWeights{
		AttentionNorm:       builder.Input("attn_norm", dtype.F32, tensor.MustShape(8)),
		AttentionNormBias:   builder.Input("attn_norm_bias", dtype.F32, tensor.MustShape(8)),
		AttentionQ:          builder.Input("attn_q", dtype.F32, tensor.MustShape(8, 8)),
		AttentionK:          builder.Input("attn_k", dtype.F32, tensor.MustShape(8, 4)),
		AttentionV:          builder.Input("attn_v", dtype.F32, tensor.MustShape(8, 4)),
		AttentionOutput:     builder.Input("attn_output", dtype.F32, tensor.MustShape(8, 8)),
		FeedForwardNorm:     builder.Input("ffn_norm", dtype.F32, tensor.MustShape(8)),
		FeedForwardNormBias: builder.Input("ffn_norm_bias", dtype.F32, tensor.MustShape(8)),
		FeedForwardGate:     builder.Input("ffn_gate", dtype.F32, tensor.MustShape(8, 12)),
		FeedForwardUp:       builder.Input("ffn_up", dtype.F32, tensor.MustShape(8, 12)),
		FeedForwardDown:     builder.Input("ffn_down", dtype.F32, tensor.MustShape(12, 8)),
	}
	result, err := model.BuildDenseBlockCachedForLayer(
		builder,
		input,
		spec,
		weights,
		[]uint32{0, 1},
		nil,
		nil,
		0,
	)
	if err != nil {
		t.Fatal(err)
	}
	feeds := map[*tensor.Tensor]reference.Value{
		input: patternedValue(input.Shape, 1, 0.2, 0),
	}
	for index, node := range []*tensor.Tensor{
		weights.AttentionQ,
		weights.AttentionK,
		weights.AttentionV,
		weights.AttentionOutput,
		weights.FeedForwardGate,
		weights.FeedForwardUp,
		weights.FeedForwardDown,
	} {
		feeds[node] = patternedValue(node.Shape, index+2, 0.06, 0)
	}
	for index, node := range []*tensor.Tensor{
		weights.AttentionNorm,
		weights.FeedForwardNorm,
	} {
		feeds[node] = patternedValue(node.Shape, index+11, 0.03, 1)
	}
	for index, node := range []*tensor.Tensor{
		weights.AttentionNormBias,
		weights.FeedForwardNormBias,
	} {
		feeds[node] = patternedValue(node.Shape, index+20, 0.02, -0.01)
	}
	want, err := reference.Execute([]*tensor.Tensor{result.Output}, feeds)
	if err != nil {
		t.Fatal(err)
	}
	cuda := newFixtureExecutor(t)
	got, err := cuda.Execute(context.Background(), []*tensor.Tensor{result.Output}, feeds)
	if err != nil {
		t.Fatal(err)
	}
	compare(t, got[result.Output].Data, want[result.Output].Data, 5e-4)
}

func TestExecutorDenseStarCoder2BlockMatchesReference(t *testing.T) {
	cudatest.Require(t)
	builder := tensor.NewBuilder()
	spec := model.Spec{CommonSpec: model.CommonSpec{Architecture: "starcoder2",
		EmbeddingLength:   8,
		FeedForwardLength: 12,

		LayerNormEpsilon: 1e-5}, AttentionSpec: model.AttentionSpec{HeadCount: 2,
		HeadCountKV:       1,
		KeyLength:         4,
		ValueLength:       4,
		RopeFrequencyBase: 10000},
	}
	input := builder.Input("input", dtype.F32, tensor.MustShape(8, 2))
	weights := model.LayerGraphWeights{
		AttentionNorm:       builder.Input("attn_norm", dtype.F32, tensor.MustShape(8)),
		AttentionNormBias:   builder.Input("attn_norm_bias", dtype.F32, tensor.MustShape(8)),
		AttentionQ:          builder.Input("attn_q", dtype.F32, tensor.MustShape(8, 8)),
		AttentionK:          builder.Input("attn_k", dtype.F32, tensor.MustShape(8, 4)),
		AttentionV:          builder.Input("attn_v", dtype.F32, tensor.MustShape(8, 4)),
		AttentionOutput:     builder.Input("attn_output", dtype.F32, tensor.MustShape(8, 8)),
		AttentionOutputBias: builder.Input("attn_output_bias", dtype.F32, tensor.MustShape(8)),
		FeedForwardNorm:     builder.Input("ffn_norm", dtype.F32, tensor.MustShape(8)),
		FeedForwardNormBias: builder.Input("ffn_norm_bias", dtype.F32, tensor.MustShape(8)),
		FeedForwardUp:       builder.Input("ffn_up", dtype.F32, tensor.MustShape(8, 12)),
		FeedForwardUpBias:   builder.Input("ffn_up_bias", dtype.F32, tensor.MustShape(12)),
		FeedForwardDown:     builder.Input("ffn_down", dtype.F32, tensor.MustShape(12, 8)),
		FeedForwardDownBias: builder.Input("ffn_down_bias", dtype.F32, tensor.MustShape(8)),
	}
	result, err := model.BuildDenseBlockCachedForLayer(
		builder,
		input,
		spec,
		weights,
		[]uint32{0, 1},
		nil,
		nil,
		0,
	)
	if err != nil {
		t.Fatal(err)
	}
	feeds := map[*tensor.Tensor]reference.Value{
		input: patternedValue(input.Shape, 1, 0.2, 0),
	}
	for index, node := range []*tensor.Tensor{
		weights.AttentionQ,
		weights.AttentionK,
		weights.AttentionV,
		weights.AttentionOutput,
		weights.FeedForwardUp,
		weights.FeedForwardDown,
	} {
		feeds[node] = patternedValue(node.Shape, index+2, 0.06, 0)
	}
	for index, node := range []*tensor.Tensor{
		weights.AttentionNorm,
		weights.FeedForwardNorm,
	} {
		feeds[node] = patternedValue(node.Shape, index+11, 0.03, 1)
	}
	for index, node := range []*tensor.Tensor{
		weights.AttentionNormBias,
		weights.AttentionOutputBias,
		weights.FeedForwardNormBias,
		weights.FeedForwardUpBias,
		weights.FeedForwardDownBias,
	} {
		feeds[node] = patternedValue(node.Shape, index+20, 0.02, -0.01)
	}
	want, err := reference.Execute([]*tensor.Tensor{result.Output}, feeds)
	if err != nil {
		t.Fatal(err)
	}
	cuda := newFixtureExecutor(t)
	got, err := cuda.Execute(context.Background(), []*tensor.Tensor{result.Output}, feeds)
	if err != nil {
		t.Fatal(err)
	}
	compare(t, got[result.Output].Data, want[result.Output].Data, 5e-4)
}

func TestExecutorCachedAttentionMatchesReference(t *testing.T) {
	cudatest.Require(t)
	builder := tensor.NewBuilder()
	query := builder.Input("query", dtype.F32, tensor.MustShape(4, 2, 1))
	pastKey := builder.Input("past_key", dtype.F32, tensor.MustShape(4, 1, 3))
	newKey := builder.Input("new_key", dtype.F32, tensor.MustShape(4, 1, 1))
	pastValue := builder.Input("past_value", dtype.F32, tensor.MustShape(4, 1, 3))
	newValue := builder.Input("new_value", dtype.F32, tensor.MustShape(4, 1, 1))
	key := builder.Concat(pastKey, newKey, 2)
	value := builder.Concat(pastValue, newValue, 2)
	output := builder.AttentionWithOffset(query, key, value, 0.5, true, 3)
	if err := builder.Err(); err != nil {
		t.Fatal(err)
	}
	feeds := map[*tensor.Tensor]reference.Value{
		query:     patternedValue(query.Shape, 1, 0.2, 0),
		pastKey:   patternedValue(pastKey.Shape, 2, 0.15, 0),
		newKey:    patternedValue(newKey.Shape, 3, 0.15, 0),
		pastValue: patternedValue(pastValue.Shape, 4, 0.2, 0),
		newValue:  patternedValue(newValue.Shape, 5, 0.2, 0),
	}
	outputs := []*tensor.Tensor{key, value, output}
	want, err := reference.Execute(outputs, feeds)
	if err != nil {
		t.Fatal(err)
	}
	cuda := newFixtureExecutor(t)
	got, err := cuda.Execute(context.Background(), outputs, feeds)
	if err != nil {
		t.Fatal(err)
	}
	compare(t, got[key].Data, want[key].Data, 0)
	compare(t, got[value].Data, want[value].Data, 0)
	compare(t, got[output].Data, want[output].Data, 3e-5)
}

func TestExecutorBatchedCachedAttentionMatchesReference(t *testing.T) {
	cudatest.Require(t)
	const (
		width      = 4
		queryHeads = 2
		keyHeads   = 1
		pastTokens = 3
		newTokens  = 1
		sequences  = 2
	)
	builder := tensor.NewBuilder()
	query := builder.Input("query", dtype.F32, tensor.MustShape(width, queryHeads, newTokens, sequences))
	pastKey := builder.Input("past_key", dtype.F32, tensor.MustShape(width, keyHeads, pastTokens, sequences))
	newKey := builder.Input("new_key", dtype.F32, tensor.MustShape(width, keyHeads, newTokens, sequences))
	pastValue := builder.Input("past_value", dtype.F32, tensor.MustShape(width, keyHeads, pastTokens, sequences))
	newValue := builder.Input("new_value", dtype.F32, tensor.MustShape(width, keyHeads, newTokens, sequences))
	key := builder.Concat(pastKey, newKey, 2)
	value := builder.Concat(pastValue, newValue, 2)
	output := builder.AttentionWithOffset(query, key, value, 0.5, true, pastTokens)
	if err := builder.Err(); err != nil {
		t.Fatal(err)
	}
	feeds := map[*tensor.Tensor]reference.Value{
		query:     patternedValue(query.Shape, 1, 0.2, 0),
		pastKey:   patternedValue(pastKey.Shape, 2, 0.15, 0),
		newKey:    patternedValue(newKey.Shape, 3, 0.15, 0),
		pastValue: patternedValue(pastValue.Shape, 4, 0.2, 0),
		newValue:  patternedValue(newValue.Shape, 5, 0.2, 0),
	}
	outputs := []*tensor.Tensor{key, value, output}
	want, err := reference.Execute(outputs, feeds)
	if err != nil {
		t.Fatal(err)
	}
	cuda := newFixtureExecutor(t)
	got, err := cuda.Execute(context.Background(), outputs, feeds)
	if err != nil {
		t.Fatal(err)
	}
	compare(t, got[key].Data, want[key].Data, 0)
	compare(t, got[value].Data, want[value].Data, 0)
	compare(t, got[output].Data, want[output].Data, 3e-5)
}

func TestExecutorNonCausalAttentionMatchesReference(t *testing.T) {
	cudatest.Require(t)
	builder := tensor.NewBuilder()
	query := builder.Input("query", dtype.F32, tensor.MustShape(4, 2, 3))
	key := builder.Input("key", dtype.F32, tensor.MustShape(4, 1, 3))
	value := builder.Input("value", dtype.F32, tensor.MustShape(4, 1, 3))
	output := builder.Attention(query, key, value, 0.5, false)
	if err := builder.Err(); err != nil {
		t.Fatal(err)
	}
	feeds := map[*tensor.Tensor]reference.Value{
		query: patternedValue(query.Shape, 31, 0.4, 0),
		key:   patternedValue(key.Shape, 37, 0.3, 0),
		value: patternedValue(value.Shape, 41, 0.2, 0),
	}
	want, err := reference.Execute([]*tensor.Tensor{output}, feeds)
	if err != nil {
		t.Fatal(err)
	}
	cuda := newFixtureExecutor(t)
	got, err := cuda.Execute(context.Background(), []*tensor.Tensor{output}, feeds)
	if err != nil {
		t.Fatal(err)
	}
	compare(t, got[output].Data, want[output].Data, 3e-5)
}

func TestExecutorWindowAttentionBlockMaskMatchesReference(t *testing.T) {
	cudatest.Require(t)
	builder := tensor.NewBuilder()
	shape := tensor.MustShape(4, 2, 5)
	query := builder.Input("query", dtype.F32, shape)
	key := builder.Input("key", dtype.F32, tensor.MustShape(4, 1, 5))
	value := builder.Input("value", dtype.F32, tensor.MustShape(4, 1, 5))
	blocks := builder.Input("blocks", dtype.F32, tensor.MustShape(5))
	output := builder.AttentionWindowWithBlockMaskWithOffset(
		query, key, value, blocks, 0.5, 0, 3,
	)
	if err := builder.Err(); err != nil {
		t.Fatal(err)
	}
	feeds := map[*tensor.Tensor]reference.Value{
		query:  patternedValue(query.Shape, 131, 0.4, 0),
		key:    patternedValue(key.Shape, 137, 0.3, 0),
		value:  patternedValue(value.Shape, 139, 0.2, 0),
		blocks: {Shape: blocks.Shape, Data: []float32{-1, 0, 0, -1, -1}},
	}
	want, err := reference.Execute([]*tensor.Tensor{output}, feeds)
	if err != nil {
		t.Fatal(err)
	}
	cuda := newFixtureExecutor(t)
	got, err := cuda.Execute(context.Background(), []*tensor.Tensor{output}, feeds)
	if err != nil {
		t.Fatal(err)
	}
	compare(t, got[output].Data, want[output].Data, 3e-5)
}

func TestExecutorAttentionSinksMatchReference(t *testing.T) {
	cudatest.Require(t)
	builder := tensor.NewBuilder()
	query := builder.Input("query", dtype.F32, tensor.MustShape(4, 2, 2))
	key := builder.Input("key", dtype.F32, tensor.MustShape(4, 1, 3))
	value := builder.Input("value", dtype.F32, tensor.MustShape(4, 1, 3))
	sinks := builder.Input("sinks", dtype.F32, tensor.MustShape(2))
	output := builder.AttentionWindowWithSinksWithOffset(query, key, value, sinks, 0.5, true, 1, 2)
	if err := builder.Err(); err != nil {
		t.Fatal(err)
	}
	feeds := map[*tensor.Tensor]reference.Value{
		query: patternedValue(query.Shape, 53, 0.4, 0),
		key:   patternedValue(key.Shape, 59, 0.3, 0),
		value: patternedValue(value.Shape, 61, 0.2, 0),
		sinks: {Shape: sinks.Shape, Data: []float32{0.5, -0.75}},
	}
	want, err := reference.Execute([]*tensor.Tensor{output}, feeds)
	if err != nil {
		t.Fatal(err)
	}
	cuda := newFixtureExecutor(t)
	got, err := cuda.Execute(context.Background(), []*tensor.Tensor{output}, feeds)
	if err != nil {
		t.Fatal(err)
	}
	compare(t, got[output].Data, want[output].Data, 3e-5)
}

func TestExecutorMiMo2BlockMatchesReference(t *testing.T) {
	cudatest.Require(t)
	builder := tensor.NewBuilder()
	spec := model.Spec{CommonSpec: model.CommonSpec{Architecture: "mimo2", BlockCount: 2, EmbeddingLength: 8,
		FeedForwardLength: 12,

		RMSNormEpsilon: 1e-5}, AttentionSpec: model.AttentionSpec{HeadCount: 2, HeadCountKV: 1,
		LayerKVHeadCounts: []uint32{1, 1}, KeyLength: 4, ValueLength: 3,
		RopeDimensionCount: 4, RopeFrequencyBase: 10000, RopeFrequencySWA: 20000,
		SlidingWindow: 128, SlidingLayers: []bool{false, true},
		AttentionValueScale: 0.5}, MoESpec: model.MoESpec{ExpertCount: 4, ExpertUsedCount: 2,
		ExpertFeedForward: 6, ExpertWeightsScale: 1.25, ExpertWeightsNorm: true,
		ExpertGatingFunc: 2},
	}
	input := builder.Input("input", dtype.F32, tensor.MustShape(8, 2))
	weights := model.LayerGraphWeights{
		AttentionNorm:          builder.Input("attn_norm", dtype.F32, tensor.MustShape(8)),
		AttentionQKV:           builder.Input("qkv", dtype.F32, tensor.MustShape(8, 15)),
		AttentionOutput:        builder.Input("attn_out", dtype.F32, tensor.MustShape(6, 8)),
		AttentionSinks:         builder.Input("sinks", dtype.F32, tensor.MustShape(2)),
		FeedForwardNorm:        builder.Input("ffn_norm", dtype.F32, tensor.MustShape(8)),
		FeedForwardRouter:      builder.Input("router", dtype.F32, tensor.MustShape(8, 4)),
		FeedForwardGateExperts: builder.Input("gate_exps", dtype.F32, tensor.MustShape(8, 6, 4)),
		FeedForwardUpExperts:   builder.Input("up_exps", dtype.F32, tensor.MustShape(8, 6, 4)),
		FeedForwardDownExperts: builder.Input("down_exps", dtype.F32, tensor.MustShape(6, 8, 4)),
		FeedForwardExpertBias:  builder.Input("expert_bias", dtype.F32, tensor.MustShape(4)),
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
		weights.AttentionSinks:        {Shape: weights.AttentionSinks.Shape, Data: []float32{0.5, -0.75}},
		weights.FeedForwardNorm:       patternedValue(weights.FeedForwardNorm.Shape, 7, 0.03, 1),
		weights.FeedForwardExpertBias: {Shape: weights.FeedForwardExpertBias.Shape, Data: []float32{0.1, -0.2, 0.3, -0.1}},
	}
	for index, node := range []*tensor.Tensor{
		weights.AttentionQKV, weights.AttentionOutput, weights.FeedForwardRouter,
		weights.FeedForwardGateExperts, weights.FeedForwardUpExperts, weights.FeedForwardDownExperts,
	} {
		feeds[node] = patternedValue(node.Shape, index+11, 0.07, 0)
	}
	outputs := []*tensor.Tensor{result.Output, result.Key, result.Value}
	want, err := reference.Execute(outputs, feeds)
	if err != nil {
		t.Fatal(err)
	}
	cuda := newFixtureExecutor(t)
	got, err := cuda.Execute(context.Background(), outputs, feeds)
	if err != nil {
		t.Fatal(err)
	}
	compare(t, got[result.Output].Data, want[result.Output].Data, 8e-4)
	compare(t, got[result.Key].Data, want[result.Key].Data, 5e-5)
	compare(t, got[result.Value].Data, want[result.Value].Data, 5e-5)
}

func TestExecutorStep35BlockMatchesReference(t *testing.T) {
	cudatest.Require(t)
	builder := tensor.NewBuilder()
	spec := model.Spec{CommonSpec: model.CommonSpec{Architecture: "step35", BlockCount: 2, EmbeddingLength: 8,
		FeedForwardLength: 12,

		RMSNormEpsilon: 1e-5}, AttentionSpec: model.AttentionSpec{HeadCount: 2, HeadCountKV: 1, LayerHeadCounts: []uint32{2, 4},
		LayerKVHeadCounts: []uint32{1, 2}, KeyLength: 4, ValueLength: 4,
		RopeDimensionCount: 4, RopeFrequencyBase: 10000, RopeFrequencySWA: 20000,
		SlidingWindow: 128, SlidingLayers: []bool{false, true}}, MoESpec: model.MoESpec{ExpertCount: 4, ExpertUsedCount: 2,
		ExpertFeedForward: 6, SharedExpertFF: 8, ExpertWeightsScale: 1.25,
		ExpertWeightsNorm: true, ExpertGatingFunc: 2,

		LayerSwiGLUClamp: []float32{0.05, 0}, LayerSharedSwiGLUClamp: []float32{0.05, 0}},
	}
	input := builder.Input("input", dtype.F32, tensor.MustShape(8, 2))
	weights := model.LayerGraphWeights{
		AttentionNorm:          builder.Input("attn_norm", dtype.F32, tensor.MustShape(8)),
		AttentionQ:             builder.Input("q", dtype.F32, tensor.MustShape(8, 8)),
		AttentionK:             builder.Input("k", dtype.F32, tensor.MustShape(8, 4)),
		AttentionV:             builder.Input("v", dtype.F32, tensor.MustShape(8, 4)),
		AttentionOutput:        builder.Input("attn_out", dtype.F32, tensor.MustShape(8, 8)),
		AttentionQNorm:         builder.Input("q_norm", dtype.F32, tensor.MustShape(4)),
		AttentionKNorm:         builder.Input("k_norm", dtype.F32, tensor.MustShape(4)),
		AttentionOutputGate:    builder.Input("attn_gate", dtype.F32, tensor.MustShape(8, 2)),
		RopeFactors:            builder.Input("rope_factors", dtype.F32, tensor.MustShape(2)),
		FeedForwardNorm:        builder.Input("ffn_norm", dtype.F32, tensor.MustShape(8)),
		FeedForwardRouter:      builder.Input("router", dtype.F32, tensor.MustShape(8, 4)),
		FeedForwardGateExperts: builder.Input("gate_exps", dtype.F32, tensor.MustShape(8, 6, 4)),
		FeedForwardUpExperts:   builder.Input("up_exps", dtype.F32, tensor.MustShape(8, 6, 4)),
		FeedForwardDownExperts: builder.Input("down_exps", dtype.F32, tensor.MustShape(6, 8, 4)),
		FeedForwardExpertBias:  builder.Input("expert_bias", dtype.F32, tensor.MustShape(4)),
		FeedForwardSharedGate:  builder.Input("shared_gate", dtype.F32, tensor.MustShape(8, 8)),
		FeedForwardSharedUp:    builder.Input("shared_up", dtype.F32, tensor.MustShape(8, 8)),
		FeedForwardSharedDown:  builder.Input("shared_down", dtype.F32, tensor.MustShape(8, 8)),
	}
	result, err := model.BuildDenseBlockCachedForLayer(
		builder, input, spec, weights, []uint32{0, 1}, nil, nil, 0,
	)
	if err != nil {
		t.Fatal(err)
	}
	feeds := map[*tensor.Tensor]reference.Value{
		input:                         patternedValue(input.Shape, 3, 0.2, 0),
		weights.AttentionNorm:         patternedValue(weights.AttentionNorm.Shape, 5, 0.03, 1),
		weights.AttentionQNorm:        patternedValue(weights.AttentionQNorm.Shape, 7, 0.03, 1),
		weights.AttentionKNorm:        patternedValue(weights.AttentionKNorm.Shape, 11, 0.03, 1),
		weights.RopeFactors:           {Shape: weights.RopeFactors.Shape, Data: []float32{1.2, 1}},
		weights.FeedForwardNorm:       patternedValue(weights.FeedForwardNorm.Shape, 13, 0.03, 1),
		weights.FeedForwardExpertBias: {Shape: weights.FeedForwardExpertBias.Shape, Data: []float32{0.1, -0.2, 0.3, -0.1}},
	}
	for index, node := range []*tensor.Tensor{
		weights.AttentionQ, weights.AttentionK, weights.AttentionV, weights.AttentionOutput,
		weights.AttentionOutputGate, weights.FeedForwardRouter, weights.FeedForwardGateExperts,
		weights.FeedForwardUpExperts, weights.FeedForwardDownExperts,
		weights.FeedForwardSharedGate, weights.FeedForwardSharedUp, weights.FeedForwardSharedDown,
	} {
		feeds[node] = patternedValue(node.Shape, index+17, 0.07, 0)
	}
	outputs := []*tensor.Tensor{result.Output, result.Key, result.Value}
	want, err := reference.Execute(outputs, feeds)
	if err != nil {
		t.Fatal(err)
	}
	cuda := newFixtureExecutor(t)
	got, err := cuda.Execute(context.Background(), outputs, feeds)
	if err != nil {
		t.Fatal(err)
	}
	compare(t, got[result.Output].Data, want[result.Output].Data, 1e-3)
	compare(t, got[result.Key].Data, want[result.Key].Data, 5e-5)
	compare(t, got[result.Value].Data, want[result.Value].Data, 5e-5)
}

func TestExecutorEuroBERTBlockMatchesReference(t *testing.T) {
	cudatest.Require(t)
	builder := tensor.NewBuilder()
	spec := model.Spec{CommonSpec: model.CommonSpec{Architecture: "eurobert", EmbeddingLength: 8, FeedForwardLength: 16,

		RMSNormEpsilon: 1e-6}, AttentionSpec: model.AttentionSpec{HeadCount: 2, HeadCountKV: 1, KeyLength: 4, ValueLength: 4,
		RopeFrequencyBase: 10000, NonCausalAttention: true},
	}
	input := builder.Input("input", dtype.F32, tensor.MustShape(8, 3))
	weights := model.LayerGraphWeights{
		AttentionNorm:   builder.Input("attn_norm", dtype.F32, tensor.MustShape(8)),
		AttentionQKV:    builder.Input("qkv", dtype.F32, tensor.MustShape(8, 16)),
		AttentionOutput: builder.Input("attn_out", dtype.F32, tensor.MustShape(8, 8)),
		FeedForwardNorm: builder.Input("ffn_norm", dtype.F32, tensor.MustShape(8)),
		FeedForwardGate: builder.Input("ffn_gate", dtype.F32, tensor.MustShape(8, 16)),
		FeedForwardUp:   builder.Input("ffn_up", dtype.F32, tensor.MustShape(8, 16)),
		FeedForwardDown: builder.Input("ffn_down", dtype.F32, tensor.MustShape(16, 8)),
	}
	result, err := model.BuildDenseBlockCachedForLayer(
		builder, input, spec, weights, []uint32{0, 1, 2}, nil, nil, 0,
	)
	if err != nil {
		t.Fatal(err)
	}
	feeds := map[*tensor.Tensor]reference.Value{
		input:                   patternedValue(input.Shape, 3, 0.2, 0),
		weights.AttentionNorm:   patternedValue(weights.AttentionNorm.Shape, 5, 0.03, 1),
		weights.FeedForwardNorm: patternedValue(weights.FeedForwardNorm.Shape, 7, 0.03, 1),
	}
	for index, node := range []*tensor.Tensor{
		weights.AttentionQKV, weights.AttentionOutput, weights.FeedForwardGate,
		weights.FeedForwardUp, weights.FeedForwardDown,
	} {
		feeds[node] = patternedValue(node.Shape, index+11, 0.06, 0)
	}
	outputs := []*tensor.Tensor{result.Output, result.Key, result.Value}
	want, err := reference.Execute(outputs, feeds)
	if err != nil {
		t.Fatal(err)
	}
	cuda := newFixtureExecutor(t)
	got, err := cuda.Execute(context.Background(), outputs, feeds)
	if err != nil {
		t.Fatal(err)
	}
	compare(t, got[result.Output].Data, want[result.Output].Data, 5e-4)
	compare(t, got[result.Key].Data, want[result.Key].Data, 5e-5)
	compare(t, got[result.Value].Data, want[result.Value].Data, 5e-5)
}

func TestExecutorBERTBlockMatchesReference(t *testing.T) {
	cudatest.Require(t)
	builder := tensor.NewBuilder()
	spec := model.Spec{CommonSpec: model.CommonSpec{Architecture: "bert", EmbeddingLength: 8, FeedForwardLength: 16,

		LayerNormEpsilon: 1e-5}, AttentionSpec: model.AttentionSpec{HeadCount: 2, HeadCountKV: 2, KeyLength: 4, ValueLength: 4,
		NonCausalAttention: true, RopeDisabled: true},
	}
	input := builder.Input("input", dtype.F32, tensor.MustShape(8, 3))
	weights := model.LayerGraphWeights{
		AttentionQKV:            builder.Input("qkv", dtype.F32, tensor.MustShape(8, 24)),
		AttentionQKVBias:        builder.Input("qkv_bias", dtype.F32, tensor.MustShape(24)),
		AttentionOutput:         builder.Input("attn_out", dtype.F32, tensor.MustShape(8, 8)),
		AttentionOutputBias:     builder.Input("attn_out_bias", dtype.F32, tensor.MustShape(8)),
		AttentionPostNorm:       builder.Input("attn_post_norm", dtype.F32, tensor.MustShape(8)),
		AttentionPostNormBias:   builder.Input("attn_post_norm_bias", dtype.F32, tensor.MustShape(8)),
		FeedForwardUp:           builder.Input("ffn_up", dtype.F32, tensor.MustShape(8, 16)),
		FeedForwardUpBias:       builder.Input("ffn_up_bias", dtype.F32, tensor.MustShape(16)),
		FeedForwardDown:         builder.Input("ffn_down", dtype.F32, tensor.MustShape(16, 8)),
		FeedForwardDownBias:     builder.Input("ffn_down_bias", dtype.F32, tensor.MustShape(8)),
		FeedForwardPostNorm:     builder.Input("ffn_post_norm", dtype.F32, tensor.MustShape(8)),
		FeedForwardPostNormBias: builder.Input("ffn_post_norm_bias", dtype.F32, tensor.MustShape(8)),
	}
	result, err := model.BuildDenseBlockCachedForLayer(
		builder, input, spec, weights, []uint32{0, 1, 2}, nil, nil, 0,
	)
	if err != nil {
		t.Fatal(err)
	}
	feeds := map[*tensor.Tensor]reference.Value{input: patternedValue(input.Shape, 3, 0.2, 0)}
	for index, node := range []*tensor.Tensor{
		weights.AttentionQKV, weights.AttentionQKVBias, weights.AttentionOutput,
		weights.AttentionOutputBias, weights.AttentionPostNorm, weights.AttentionPostNormBias,
		weights.FeedForwardUp, weights.FeedForwardUpBias, weights.FeedForwardDown,
		weights.FeedForwardDownBias, weights.FeedForwardPostNorm, weights.FeedForwardPostNormBias,
	} {
		offset := float32(0)
		if node == weights.AttentionPostNorm || node == weights.FeedForwardPostNorm {
			offset = 1
		}
		feeds[node] = patternedValue(node.Shape, index+11, 0.05, offset)
	}
	outputs := []*tensor.Tensor{result.Output, result.Key, result.Value}
	want, err := reference.Execute(outputs, feeds)
	if err != nil {
		t.Fatal(err)
	}
	cuda := newFixtureExecutor(t)
	got, err := cuda.Execute(context.Background(), outputs, feeds)
	if err != nil {
		t.Fatal(err)
	}
	compare(t, got[result.Output].Data, want[result.Output].Data, 6e-4)
	compare(t, got[result.Key].Data, want[result.Key].Data, 5e-5)
	compare(t, got[result.Value].Data, want[result.Value].Data, 5e-5)
}

func TestExecutorNeoBERTBlockMatchesReference(t *testing.T) {
	cudatest.Require(t)
	builder := tensor.NewBuilder()
	spec := model.Spec{CommonSpec: model.CommonSpec{Architecture: "neo-bert", EmbeddingLength: 8, FeedForwardLength: 16,

		RMSNormEpsilon: 1e-6}, AttentionSpec: model.AttentionSpec{HeadCount: 2, HeadCountKV: 2, KeyLength: 4, ValueLength: 4,
		RopeDimensionCount: 4, RopeFrequencyBase: 10000,
		NonCausalAttention: true},
	}
	input := builder.Input("input", dtype.F32, tensor.MustShape(8, 3))
	weights := model.LayerGraphWeights{
		AttentionNorm:   builder.Input("attn_norm", dtype.F32, tensor.MustShape(8)),
		AttentionQKV:    builder.Input("qkv", dtype.F32, tensor.MustShape(8, 24)),
		AttentionOutput: builder.Input("attn_out", dtype.F32, tensor.MustShape(8, 8)),
		FeedForwardNorm: builder.Input("ffn_norm", dtype.F32, tensor.MustShape(8)),
		FeedForwardUp:   builder.Input("ffn_up", dtype.F32, tensor.MustShape(8, 32)),
		FeedForwardDown: builder.Input("ffn_down", dtype.F32, tensor.MustShape(16, 8)),
	}
	result, err := model.BuildDenseBlockCachedForLayer(
		builder, input, spec, weights, []uint32{0, 1, 2}, nil, nil, 0,
	)
	if err != nil {
		t.Fatal(err)
	}
	feeds := map[*tensor.Tensor]reference.Value{
		input:                   patternedValue(input.Shape, 3, 0.2, 0),
		weights.AttentionNorm:   patternedValue(weights.AttentionNorm.Shape, 5, 0.03, 1),
		weights.FeedForwardNorm: patternedValue(weights.FeedForwardNorm.Shape, 7, 0.03, 1),
	}
	for index, node := range []*tensor.Tensor{
		weights.AttentionQKV, weights.AttentionOutput,
		weights.FeedForwardUp, weights.FeedForwardDown,
	} {
		feeds[node] = patternedValue(node.Shape, index+11, 0.06, 0)
	}
	outputs := []*tensor.Tensor{result.Output, result.Key, result.Value}
	want, err := reference.Execute(outputs, feeds)
	if err != nil {
		t.Fatal(err)
	}
	cuda := newFixtureExecutor(t)
	got, err := cuda.Execute(context.Background(), outputs, feeds)
	if err != nil {
		t.Fatal(err)
	}
	compare(t, got[result.Output].Data, want[result.Output].Data, 6e-4)
	compare(t, got[result.Key].Data, want[result.Key].Data, 5e-5)
	compare(t, got[result.Value].Data, want[result.Value].Data, 5e-5)
}

func TestExecutorLlamaEmbedBlockMatchesReference(t *testing.T) {
	cudatest.Require(t)
	builder := tensor.NewBuilder()
	spec := model.Spec{CommonSpec: model.CommonSpec{Architecture: "llama-embed", EmbeddingLength: 8, FeedForwardLength: 12,

		RMSNormEpsilon: 1e-6}, AttentionSpec: model.AttentionSpec{HeadCount: 2, HeadCountKV: 1, KeyLength: 4, ValueLength: 4,
		RopeFrequencyBase: 10000, NonCausalAttention: true},
	}
	input := builder.Input("input", dtype.F32, tensor.MustShape(8, 3))
	weights := model.LayerGraphWeights{
		AttentionNorm:   builder.Input("attn_norm", dtype.F32, tensor.MustShape(8)),
		AttentionQ:      builder.Input("q", dtype.F32, tensor.MustShape(8, 8)),
		AttentionK:      builder.Input("k", dtype.F32, tensor.MustShape(8, 4)),
		AttentionV:      builder.Input("v", dtype.F32, tensor.MustShape(8, 4)),
		AttentionOutput: builder.Input("attn_out", dtype.F32, tensor.MustShape(8, 8)),
		FeedForwardNorm: builder.Input("ffn_norm", dtype.F32, tensor.MustShape(8)),
		FeedForwardGate: builder.Input("ffn_gate", dtype.F32, tensor.MustShape(8, 12)),
		FeedForwardUp:   builder.Input("ffn_up", dtype.F32, tensor.MustShape(8, 12)),
		FeedForwardDown: builder.Input("ffn_down", dtype.F32, tensor.MustShape(12, 8)),
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
		weights.FeedForwardGate, weights.FeedForwardUp, weights.FeedForwardDown,
	} {
		feeds[node] = patternedValue(node.Shape, index+11, 0.06, 0)
	}
	feeds[weights.AttentionNorm] = patternedValue(weights.AttentionNorm.Shape, 31, 0.03, 1)
	feeds[weights.FeedForwardNorm] = patternedValue(weights.FeedForwardNorm.Shape, 37, 0.03, 1)
	outputs := []*tensor.Tensor{result.Output, result.Key, result.Value}
	want, err := reference.Execute(outputs, feeds)
	if err != nil {
		t.Fatal(err)
	}
	cuda := newFixtureExecutor(t)
	got, err := cuda.Execute(context.Background(), outputs, feeds)
	if err != nil {
		t.Fatal(err)
	}
	compare(t, got[result.Output].Data, want[result.Output].Data, 7e-4)
	compare(t, got[result.Key].Data, want[result.Key].Data, 5e-5)
	compare(t, got[result.Value].Data, want[result.Value].Data, 5e-5)
}

func TestExecutorPanguEmbeddedBlockMatchesReference(t *testing.T) {
	cudatest.Require(t)
	builder := tensor.NewBuilder()
	spec := model.Spec{CommonSpec: model.CommonSpec{Architecture: "pangu-embedded", EmbeddingLength: 8, FeedForwardLength: 12,

		RMSNormEpsilon: 1e-6}, AttentionSpec: model.AttentionSpec{HeadCount: 2, HeadCountKV: 1, KeyLength: 4, ValueLength: 4,
		RopeDimensionCount: 4, RopeFrequencyBase: 10000, RopeAttentionFactor: 1.2},
	}
	input := builder.Input("input", dtype.F32, tensor.MustShape(8, 3))
	weights := model.LayerGraphWeights{
		AttentionNorm:       builder.Input("attn_norm", dtype.F32, tensor.MustShape(8)),
		AttentionQ:          builder.Input("q", dtype.F32, tensor.MustShape(8, 8)),
		AttentionK:          builder.Input("k", dtype.F32, tensor.MustShape(8, 4)),
		AttentionV:          builder.Input("v", dtype.F32, tensor.MustShape(8, 4)),
		AttentionOutput:     builder.Input("attn_out", dtype.F32, tensor.MustShape(8, 8)),
		AttentionOutputBias: builder.Input("attn_out_bias", dtype.F32, tensor.MustShape(8)),
		RopeFactors:         builder.Input("rope_factors", dtype.F32, tensor.MustShape(2)),
		FeedForwardNorm:     builder.Input("ffn_norm", dtype.F32, tensor.MustShape(8)),
		FeedForwardGate:     builder.Input("ffn_gate", dtype.F32, tensor.MustShape(8, 12)),
		FeedForwardUp:       builder.Input("ffn_up", dtype.F32, tensor.MustShape(8, 12)),
		FeedForwardDown:     builder.Input("ffn_down", dtype.F32, tensor.MustShape(12, 8)),
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
		weights.AttentionOutputBias, weights.FeedForwardGate, weights.FeedForwardUp,
		weights.FeedForwardDown,
	} {
		feeds[node] = patternedValue(node.Shape, index+11, 0.05, 0)
	}
	feeds[weights.AttentionNorm] = patternedValue(weights.AttentionNorm.Shape, 31, 0.03, 1)
	feeds[weights.FeedForwardNorm] = patternedValue(weights.FeedForwardNorm.Shape, 37, 0.03, 1)
	feeds[weights.RopeFactors] = patternedValue(weights.RopeFactors.Shape, 41, 0.04, 1)
	outputs := []*tensor.Tensor{result.Output, result.Key, result.Value}
	want, err := reference.Execute(outputs, feeds)
	if err != nil {
		t.Fatal(err)
	}
	cuda := newFixtureExecutor(t)
	got, err := cuda.Execute(context.Background(), outputs, feeds)
	if err != nil {
		t.Fatal(err)
	}
	compare(t, got[result.Output].Data, want[result.Output].Data, 8e-4)
	compare(t, got[result.Key].Data, want[result.Key].Data, 7e-5)
	compare(t, got[result.Value].Data, want[result.Value].Data, 5e-5)
}

func TestExecutorModernBERTBlockMatchesReference(t *testing.T) {
	cudatest.Require(t)
	builder := tensor.NewBuilder()
	spec := model.Spec{CommonSpec: model.CommonSpec{Architecture: "modern-bert", BlockCount: 3, EmbeddingLength: 8,
		FeedForwardLength: 16,

		LayerNormEpsilon: 1e-5,
		HiddenActivation: "silu"}, AttentionSpec: model.AttentionSpec{HeadCount: 2, HeadCountKV: 2,
		KeyLength: 4, ValueLength: 4, RopeDimensionCount: 4,
		RopeFrequencyBase: 10000, RopeFrequencySWA: 50000,
		SlidingWindow: 4, SlidingPattern: 3,
		NonCausalAttention: true},
	}
	input := builder.Input("input", dtype.F32, tensor.MustShape(8, 5))
	weights := model.LayerGraphWeights{
		AttentionNorm:   builder.Input("attn_norm", dtype.F32, tensor.MustShape(8)),
		AttentionQKV:    builder.Input("qkv", dtype.F32, tensor.MustShape(8, 24)),
		AttentionOutput: builder.Input("attn_out", dtype.F32, tensor.MustShape(8, 8)),
		FeedForwardNorm: builder.Input("ffn_norm", dtype.F32, tensor.MustShape(8)),
		FeedForwardUp:   builder.Input("ffn_up", dtype.F32, tensor.MustShape(8, 32)),
		FeedForwardDown: builder.Input("ffn_down", dtype.F32, tensor.MustShape(16, 8)),
	}
	result, err := model.BuildDenseBlockCachedForLayer(
		builder, input, spec, weights, []uint32{0, 1, 2, 3, 4}, nil, nil, 1,
	)
	if err != nil {
		t.Fatal(err)
	}
	feeds := map[*tensor.Tensor]reference.Value{input: patternedValue(input.Shape, 3, 0.2, 0)}
	for index, node := range []*tensor.Tensor{
		weights.AttentionQKV, weights.AttentionOutput, weights.FeedForwardUp, weights.FeedForwardDown,
	} {
		feeds[node] = patternedValue(node.Shape, index+11, 0.05, 0)
	}
	feeds[weights.AttentionNorm] = patternedValue(weights.AttentionNorm.Shape, 31, 0.03, 1)
	feeds[weights.FeedForwardNorm] = patternedValue(weights.FeedForwardNorm.Shape, 37, 0.03, 1)
	outputs := []*tensor.Tensor{result.Output, result.Key, result.Value}
	want, err := reference.Execute(outputs, feeds)
	if err != nil {
		t.Fatal(err)
	}
	cuda := newFixtureExecutor(t)
	got, err := cuda.Execute(context.Background(), outputs, feeds)
	if err != nil {
		t.Fatal(err)
	}
	compare(t, got[result.Output].Data, want[result.Output].Data, 9e-4)
	compare(t, got[result.Key].Data, want[result.Key].Data, 6e-5)
	compare(t, got[result.Value].Data, want[result.Value].Data, 5e-5)
}

func TestExecutorGemmaEmbeddingBlockMatchesReference(t *testing.T) {
	cudatest.Require(t)
	builder := tensor.NewBuilder()
	spec := model.Spec{CommonSpec: model.CommonSpec{Architecture: "gemma-embedding", BlockCount: 6, EmbeddingLength: 8,
		FeedForwardLength: 12,

		RMSNormEpsilon: 1e-6}, AttentionSpec: model.AttentionSpec{HeadCount: 2, HeadCountKV: 1,
		KeyLength: 4, ValueLength: 4, RopeDimensionCount: 4,
		RopeFrequencyBase: 10000, RopeFrequencySWA: 50000,
		SlidingWindow: 4, SlidingPattern: 6,
		NonCausalAttention: true},
	}
	input := builder.Input("input", dtype.F32, tensor.MustShape(8, 5))
	weights := model.LayerGraphWeights{
		AttentionNorm:       builder.Input("attn_norm", dtype.F32, tensor.MustShape(8)),
		AttentionQ:          builder.Input("q", dtype.F32, tensor.MustShape(8, 8)),
		AttentionK:          builder.Input("k", dtype.F32, tensor.MustShape(8, 4)),
		AttentionV:          builder.Input("v", dtype.F32, tensor.MustShape(8, 4)),
		AttentionQBias:      builder.Input("q_bias", dtype.F32, tensor.MustShape(8)),
		AttentionKBias:      builder.Input("k_bias", dtype.F32, tensor.MustShape(4)),
		AttentionVBias:      builder.Input("v_bias", dtype.F32, tensor.MustShape(4)),
		AttentionOutput:     builder.Input("attn_out", dtype.F32, tensor.MustShape(8, 8)),
		AttentionQNorm:      builder.Input("q_norm", dtype.F32, tensor.MustShape(4)),
		AttentionKNorm:      builder.Input("k_norm", dtype.F32, tensor.MustShape(4)),
		AttentionPostNorm:   builder.Input("attn_post_norm", dtype.F32, tensor.MustShape(8)),
		FeedForwardNorm:     builder.Input("ffn_norm", dtype.F32, tensor.MustShape(8)),
		FeedForwardGate:     builder.Input("ffn_gate", dtype.F32, tensor.MustShape(8, 12)),
		FeedForwardUp:       builder.Input("ffn_up", dtype.F32, tensor.MustShape(8, 12)),
		FeedForwardDown:     builder.Input("ffn_down", dtype.F32, tensor.MustShape(12, 8)),
		FeedForwardPostNorm: builder.Input("ffn_post_norm", dtype.F32, tensor.MustShape(8)),
	}
	result, err := model.BuildDenseBlockCachedForLayer(
		builder, input, spec, weights, []uint32{0, 1, 2, 3, 4}, nil, nil, 0,
	)
	if err != nil {
		t.Fatal(err)
	}
	feeds := map[*tensor.Tensor]reference.Value{input: patternedValue(input.Shape, 3, 0.2, 0)}
	for index, node := range []*tensor.Tensor{
		weights.AttentionQ, weights.AttentionK, weights.AttentionV,
		weights.AttentionQBias, weights.AttentionKBias, weights.AttentionVBias,
		weights.AttentionOutput, weights.FeedForwardGate, weights.FeedForwardUp,
		weights.FeedForwardDown,
	} {
		feeds[node] = patternedValue(node.Shape, index+11, 0.05, 0)
	}
	for index, node := range []*tensor.Tensor{
		weights.AttentionNorm, weights.AttentionQNorm, weights.AttentionKNorm,
		weights.AttentionPostNorm, weights.FeedForwardNorm, weights.FeedForwardPostNorm,
	} {
		feeds[node] = patternedValue(node.Shape, index+31, 0.03, 1)
	}
	outputs := []*tensor.Tensor{result.Output, result.Key, result.Value}
	want, err := reference.Execute(outputs, feeds)
	if err != nil {
		t.Fatal(err)
	}
	cuda := newFixtureExecutor(t)
	got, err := cuda.Execute(context.Background(), outputs, feeds)
	if err != nil {
		t.Fatal(err)
	}
	compare(t, got[result.Output].Data, want[result.Output].Data, 1e-3)
	compare(t, got[result.Key].Data, want[result.Key].Data, 8e-5)
	compare(t, got[result.Value].Data, want[result.Value].Data, 5e-5)
}

func TestExecutorTalkieBlockMatchesReference(t *testing.T) {
	cudatest.Require(t)
	builder := tensor.NewBuilder()
	spec := model.Spec{CommonSpec: model.CommonSpec{Architecture: "talkie", EmbeddingLength: 8, FeedForwardLength: 12,

		RMSNormEpsilon: 1e-6}, AttentionSpec: model.AttentionSpec{HeadCount: 2, HeadCountKV: 1, KeyLength: 4, ValueLength: 4,
		RopeDimensionCount: 4, RopeFrequencyBase: 10000},
	}
	input := builder.Input("input", dtype.F32, tensor.MustShape(8, 4))
	weights := model.LayerGraphWeights{
		AttentionQ:       builder.Input("q", dtype.F32, tensor.MustShape(8, 8)),
		AttentionK:       builder.Input("k", dtype.F32, tensor.MustShape(8, 4)),
		AttentionV:       builder.Input("v", dtype.F32, tensor.MustShape(8, 4)),
		AttentionQBias:   builder.Input("q_bias", dtype.F32, tensor.MustShape(8)),
		AttentionKBias:   builder.Input("k_bias", dtype.F32, tensor.MustShape(4)),
		AttentionVBias:   builder.Input("v_bias", dtype.F32, tensor.MustShape(4)),
		AttentionOutput:  builder.Input("attn_out", dtype.F32, tensor.MustShape(8, 8)),
		AttentionQNorm:   builder.Input("q_norm", dtype.F32, tensor.MustShape(1, 2)),
		FeedForwardGate:  builder.Input("ffn_gate", dtype.F32, tensor.MustShape(8, 12)),
		FeedForwardUp:    builder.Input("ffn_up", dtype.F32, tensor.MustShape(8, 12)),
		FeedForwardDown:  builder.Input("ffn_down", dtype.F32, tensor.MustShape(12, 8)),
		LayerOutputScale: builder.Input("layer_scale", dtype.F32, tensor.MustShape(1)),
		EmbeddingSkip:    builder.Input("embedding_skip", dtype.F32, tensor.MustShape(8, 4)),
	}
	result, err := model.BuildDenseBlockCachedForLayer(
		builder, input, spec, weights, []uint32{0, 1, 2, 3}, nil, nil, 0,
	)
	if err != nil {
		t.Fatal(err)
	}
	feeds := map[*tensor.Tensor]reference.Value{
		input:                  patternedValue(input.Shape, 3, 0.2, 0),
		weights.EmbeddingSkip:  patternedValue(weights.EmbeddingSkip.Shape, 5, 0.18, 0),
		weights.AttentionQNorm: patternedValue(weights.AttentionQNorm.Shape, 7, 0.03, 1),
		weights.LayerOutputScale: {
			Shape: weights.LayerOutputScale.Shape, Data: []float32{0.125},
		},
	}
	for index, node := range []*tensor.Tensor{
		weights.AttentionQ, weights.AttentionK, weights.AttentionV,
		weights.AttentionQBias, weights.AttentionKBias, weights.AttentionVBias,
		weights.AttentionOutput, weights.FeedForwardGate, weights.FeedForwardUp,
		weights.FeedForwardDown,
	} {
		feeds[node] = patternedValue(node.Shape, index+11, 0.05, 0)
	}
	outputs := []*tensor.Tensor{result.Output, result.Key, result.Value}
	want, err := reference.Execute(outputs, feeds)
	if err != nil {
		t.Fatal(err)
	}
	cuda := newFixtureExecutor(t)
	got, err := cuda.Execute(context.Background(), outputs, feeds)
	if err != nil {
		t.Fatal(err)
	}
	compare(t, got[result.Output].Data, want[result.Output].Data, 1e-3)
	compare(t, got[result.Key].Data, want[result.Key].Data, 8e-5)
	compare(t, got[result.Value].Data, want[result.Value].Data, 5e-5)
}

func TestExecutorNomicBERTBlockMatchesReference(t *testing.T) {
	cudatest.Require(t)
	builder := tensor.NewBuilder()
	spec := model.Spec{CommonSpec: model.CommonSpec{Architecture: "nomic-bert", EmbeddingLength: 8, FeedForwardLength: 16,

		LayerNormEpsilon: 1e-5}, AttentionSpec: model.AttentionSpec{HeadCount: 2, HeadCountKV: 2, KeyLength: 4, ValueLength: 4,
		RopeDimensionCount: 4, RopeFrequencyBase: 10000,
		NonCausalAttention: true},
	}
	input := builder.Input("input", dtype.F32, tensor.MustShape(8, 3))
	weights := model.LayerGraphWeights{
		AttentionQKV:            builder.Input("qkv", dtype.F32, tensor.MustShape(8, 24)),
		AttentionOutput:         builder.Input("attn_out", dtype.F32, tensor.MustShape(8, 8)),
		AttentionPostNorm:       builder.Input("attn_post_norm", dtype.F32, tensor.MustShape(8)),
		AttentionPostNormBias:   builder.Input("attn_post_norm_bias", dtype.F32, tensor.MustShape(8)),
		FeedForwardGate:         builder.Input("ffn_gate", dtype.F32, tensor.MustShape(8, 16)),
		FeedForwardUp:           builder.Input("ffn_up", dtype.F32, tensor.MustShape(8, 16)),
		FeedForwardDown:         builder.Input("ffn_down", dtype.F32, tensor.MustShape(16, 8)),
		FeedForwardPostNorm:     builder.Input("ffn_post_norm", dtype.F32, tensor.MustShape(8)),
		FeedForwardPostNormBias: builder.Input("ffn_post_norm_bias", dtype.F32, tensor.MustShape(8)),
	}
	result, err := model.BuildDenseBlockCachedForLayer(
		builder, input, spec, weights, []uint32{0, 1, 2}, nil, nil, 0,
	)
	if err != nil {
		t.Fatal(err)
	}
	feeds := map[*tensor.Tensor]reference.Value{input: patternedValue(input.Shape, 3, 0.2, 0)}
	for index, node := range []*tensor.Tensor{
		weights.AttentionQKV, weights.AttentionOutput, weights.AttentionPostNorm,
		weights.AttentionPostNormBias, weights.FeedForwardGate, weights.FeedForwardUp,
		weights.FeedForwardDown, weights.FeedForwardPostNorm, weights.FeedForwardPostNormBias,
	} {
		offset := float32(0)
		if node == weights.AttentionPostNorm || node == weights.FeedForwardPostNorm {
			offset = 1
		}
		feeds[node] = patternedValue(node.Shape, index+11, 0.05, offset)
	}
	outputs := []*tensor.Tensor{result.Output, result.Key, result.Value}
	want, err := reference.Execute(outputs, feeds)
	if err != nil {
		t.Fatal(err)
	}
	cuda := newFixtureExecutor(t)
	got, err := cuda.Execute(context.Background(), outputs, feeds)
	if err != nil {
		t.Fatal(err)
	}
	compare(t, got[result.Output].Data, want[result.Output].Data, 7e-4)
	compare(t, got[result.Key].Data, want[result.Key].Data, 5e-5)
	compare(t, got[result.Value].Data, want[result.Value].Data, 5e-5)
}

func TestExecutorJinaBERTV2BlockMatchesReference(t *testing.T) {
	cudatest.Require(t)
	builder := tensor.NewBuilder()
	spec := model.Spec{CommonSpec: model.CommonSpec{Architecture: "jina-bert-v2", EmbeddingLength: 8, FeedForwardLength: 16,

		LayerNormEpsilon: 1e-5}, AttentionSpec: model.AttentionSpec{HeadCount: 2, HeadCountKV: 2, KeyLength: 4, ValueLength: 4,
		NonCausalAttention: true, RopeDisabled: true, MaxALiBiBias: 8},
	}
	input := builder.Input("input", dtype.F32, tensor.MustShape(8, 3))
	weights := model.LayerGraphWeights{
		AttentionQKV:            builder.Input("qkv", dtype.F32, tensor.MustShape(8, 24)),
		AttentionQNorm:          builder.Input("q_norm", dtype.F32, tensor.MustShape(8)),
		AttentionQNormBias:      builder.Input("q_norm_bias", dtype.F32, tensor.MustShape(8)),
		AttentionKNorm:          builder.Input("k_norm", dtype.F32, tensor.MustShape(8)),
		AttentionKNormBias:      builder.Input("k_norm_bias", dtype.F32, tensor.MustShape(8)),
		AttentionOutput:         builder.Input("attn_out", dtype.F32, tensor.MustShape(8, 8)),
		AttentionPostNorm:       builder.Input("attn_post_norm", dtype.F32, tensor.MustShape(8)),
		AttentionPostNormBias:   builder.Input("attn_post_norm_bias", dtype.F32, tensor.MustShape(8)),
		AttentionNorm2:          builder.Input("attn_norm_2", dtype.F32, tensor.MustShape(8)),
		AttentionNorm2Bias:      builder.Input("attn_norm_2_bias", dtype.F32, tensor.MustShape(8)),
		FeedForwardUp:           builder.Input("ffn_up", dtype.F32, tensor.MustShape(8, 32)),
		FeedForwardDown:         builder.Input("ffn_down", dtype.F32, tensor.MustShape(16, 8)),
		FeedForwardPostNorm:     builder.Input("ffn_post_norm", dtype.F32, tensor.MustShape(8)),
		FeedForwardPostNormBias: builder.Input("ffn_post_norm_bias", dtype.F32, tensor.MustShape(8)),
	}
	result, err := model.BuildDenseBlockCachedForLayer(
		builder, input, spec, weights, []uint32{0, 1, 2}, nil, nil, 0,
	)
	if err != nil {
		t.Fatal(err)
	}
	feeds := map[*tensor.Tensor]reference.Value{input: patternedValue(input.Shape, 3, 0.2, 0)}
	for index, node := range []*tensor.Tensor{
		weights.AttentionQKV, weights.AttentionQNorm, weights.AttentionQNormBias,
		weights.AttentionKNorm, weights.AttentionKNormBias, weights.AttentionOutput,
		weights.AttentionPostNorm, weights.AttentionPostNormBias, weights.AttentionNorm2,
		weights.AttentionNorm2Bias, weights.FeedForwardUp, weights.FeedForwardDown,
		weights.FeedForwardPostNorm, weights.FeedForwardPostNormBias,
	} {
		offset := float32(0)
		if node == weights.AttentionQNorm || node == weights.AttentionKNorm ||
			node == weights.AttentionPostNorm || node == weights.AttentionNorm2 ||
			node == weights.FeedForwardPostNorm {
			offset = 1
		}
		feeds[node] = patternedValue(node.Shape, index+11, 0.04, offset)
	}
	outputs := []*tensor.Tensor{result.Output, result.Key, result.Value}
	want, err := reference.Execute(outputs, feeds)
	if err != nil {
		t.Fatal(err)
	}
	cuda := newFixtureExecutor(t)
	got, err := cuda.Execute(context.Background(), outputs, feeds)
	if err != nil {
		t.Fatal(err)
	}
	compare(t, got[result.Output].Data, want[result.Output].Data, 8e-4)
	compare(t, got[result.Key].Data, want[result.Key].Data, 8e-5)
	compare(t, got[result.Value].Data, want[result.Value].Data, 5e-5)
}

func TestExecutorJinaBERTV3BlockMatchesReference(t *testing.T) {
	cudatest.Require(t)
	builder := tensor.NewBuilder()
	spec := model.Spec{CommonSpec: model.CommonSpec{Architecture: "jina-bert-v3", EmbeddingLength: 8, FeedForwardLength: 16,

		LayerNormEpsilon: 1e-5}, AttentionSpec: model.AttentionSpec{HeadCount: 2, HeadCountKV: 2, KeyLength: 4, ValueLength: 4,
		RopeDimensionCount: 4, RopeFrequencyBase: 10000,
		NonCausalAttention: true},
	}
	input := builder.Input("input", dtype.F32, tensor.MustShape(8, 3))
	weights := model.LayerGraphWeights{
		AttentionQKV:            builder.Input("qkv", dtype.F32, tensor.MustShape(8, 24)),
		AttentionOutput:         builder.Input("attn_out", dtype.F32, tensor.MustShape(8, 8)),
		AttentionPostNorm:       builder.Input("attn_post_norm", dtype.F32, tensor.MustShape(8)),
		AttentionPostNormBias:   builder.Input("attn_post_norm_bias", dtype.F32, tensor.MustShape(8)),
		FeedForwardUp:           builder.Input("ffn_up", dtype.F32, tensor.MustShape(8, 16)),
		FeedForwardDown:         builder.Input("ffn_down", dtype.F32, tensor.MustShape(16, 8)),
		FeedForwardPostNorm:     builder.Input("ffn_post_norm", dtype.F32, tensor.MustShape(8)),
		FeedForwardPostNormBias: builder.Input("ffn_post_norm_bias", dtype.F32, tensor.MustShape(8)),
	}
	result, err := model.BuildDenseBlockCachedForLayer(
		builder, input, spec, weights, []uint32{0, 1, 2}, nil, nil, 0,
	)
	if err != nil {
		t.Fatal(err)
	}
	feeds := map[*tensor.Tensor]reference.Value{input: patternedValue(input.Shape, 3, 0.2, 0)}
	for index, node := range []*tensor.Tensor{
		weights.AttentionQKV, weights.AttentionOutput, weights.AttentionPostNorm,
		weights.AttentionPostNormBias, weights.FeedForwardUp, weights.FeedForwardDown,
		weights.FeedForwardPostNorm, weights.FeedForwardPostNormBias,
	} {
		offset := float32(0)
		if node == weights.AttentionPostNorm || node == weights.FeedForwardPostNorm {
			offset = 1
		}
		feeds[node] = patternedValue(node.Shape, index+11, 0.05, offset)
	}
	outputs := []*tensor.Tensor{result.Output, result.Key, result.Value}
	want, err := reference.Execute(outputs, feeds)
	if err != nil {
		t.Fatal(err)
	}
	cuda := newFixtureExecutor(t)
	got, err := cuda.Execute(context.Background(), outputs, feeds)
	if err != nil {
		t.Fatal(err)
	}
	compare(t, got[result.Output].Data, want[result.Output].Data, 7e-4)
	compare(t, got[result.Key].Data, want[result.Key].Data, 5e-5)
	compare(t, got[result.Value].Data, want[result.Value].Data, 5e-5)
}

func TestExecutorNomicBERTMoEBlockMatchesReference(t *testing.T) {
	cudatest.Require(t)
	builder := tensor.NewBuilder()
	spec := model.Spec{CommonSpec: model.CommonSpec{Architecture: "nomic-bert-moe", EmbeddingLength: 8, FeedForwardLength: 16,

		LayerNormEpsilon: 1e-5, BlockCount: 2}, AttentionSpec: model.AttentionSpec{HeadCount: 2, HeadCountKV: 2, KeyLength: 4, ValueLength: 4,

		RopeDimensionCount: 4, RopeFrequencyBase: 10000,
		NonCausalAttention: true}, MoESpec: model.MoESpec{ExpertCount: 4, ExpertUsedCount: 2, ExpertFeedForward: 16, ExpertWeightsScale: 1,
		MoELayerStep: 2},
	}
	input := builder.Input("input", dtype.F32, tensor.MustShape(8, 3))
	weights := model.LayerGraphWeights{
		AttentionQKV:            builder.Input("qkv", dtype.F32, tensor.MustShape(8, 24)),
		AttentionOutput:         builder.Input("attn_out", dtype.F32, tensor.MustShape(8, 8)),
		AttentionPostNorm:       builder.Input("attn_post_norm", dtype.F32, tensor.MustShape(8)),
		AttentionPostNormBias:   builder.Input("attn_post_norm_bias", dtype.F32, tensor.MustShape(8)),
		FeedForwardRouter:       builder.Input("router", dtype.F32, tensor.MustShape(8, 4)),
		FeedForwardUpExperts:    builder.Input("up_exps", dtype.F32, tensor.MustShape(8, 16, 4)),
		FeedForwardDownExperts:  builder.Input("down_exps", dtype.F32, tensor.MustShape(16, 8, 4)),
		FeedForwardPostNorm:     builder.Input("ffn_post_norm", dtype.F32, tensor.MustShape(8)),
		FeedForwardPostNormBias: builder.Input("ffn_post_norm_bias", dtype.F32, tensor.MustShape(8)),
	}
	result, err := model.BuildDenseBlockCachedForLayer(builder, input, spec, weights, []uint32{0, 1, 2}, nil, nil, 1)
	if err != nil {
		t.Fatal(err)
	}
	feeds := map[*tensor.Tensor]reference.Value{input: patternedValue(input.Shape, 3, 0.2, 0)}
	for index, node := range []*tensor.Tensor{
		weights.AttentionQKV, weights.AttentionOutput, weights.AttentionPostNorm,
		weights.AttentionPostNormBias, weights.FeedForwardRouter, weights.FeedForwardUpExperts,
		weights.FeedForwardDownExperts, weights.FeedForwardPostNorm, weights.FeedForwardPostNormBias,
	} {
		offset := float32(0)
		if node == weights.AttentionPostNorm || node == weights.FeedForwardPostNorm {
			offset = 1
		}
		feeds[node] = patternedValue(node.Shape, index+11, 0.04, offset)
	}
	outputs := []*tensor.Tensor{result.Output, result.Key, result.Value}
	want, err := reference.Execute(outputs, feeds)
	if err != nil {
		t.Fatal(err)
	}
	cuda := newFixtureExecutor(t)
	got, err := cuda.Execute(context.Background(), outputs, feeds)
	if err != nil {
		t.Fatal(err)
	}
	compare(t, got[result.Output].Data, want[result.Output].Data, 9e-4)
	compare(t, got[result.Key].Data, want[result.Key].Data, 5e-5)
	compare(t, got[result.Value].Data, want[result.Value].Data, 5e-5)
}
