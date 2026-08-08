//go:build windows

package executor

import (
	"context"

	"fmt"

	"overgo/internal/model"
	"overgo/internal/modeltest"

	"overgo/internal/tensor"

	"overgo/internal/tensor/dtype"

	"overgo/internal/tensor/reference"

	"testing"

	cudatest "overgo/internal/cuda/testutil"
)

func TestExecutorSSMScanMatchesReference(t *testing.T) {
	cudatest.Require(t)
	builder := tensor.NewBuilder()
	state := builder.Input("state", dtype.F32, tensor.MustShape(4, 1, 4, 2))
	x := builder.Input("x", dtype.F32, tensor.MustShape(1, 4, 3, 2))
	dt := builder.Input("dt", dtype.F32, tensor.MustShape(4, 3, 2))
	a := builder.Input("a", dtype.F32, tensor.MustShape(1, 4))
	beta := builder.Input("beta", dtype.F32, tensor.MustShape(4, 2, 3, 2))
	c := builder.Input("c", dtype.F32, tensor.MustShape(4, 2, 3, 2))
	output := builder.SSMScan(state, x, dt, a, beta, c)
	if err := builder.Err(); err != nil {
		t.Fatal(err)
	}
	feeds := map[*tensor.Tensor]reference.Value{
		state: patternedValue(state.Shape, 5, 0.03, -0.1),
		x:     patternedValue(x.Shape, 7, 0.07, -0.2),
		dt:    patternedValue(dt.Shape, 11, 0.04, -0.3),
		a:     patternedValue(a.Shape, 13, 0.02, -0.5),
		beta:  patternedValue(beta.Shape, 17, 0.05, -0.1),
		c:     patternedValue(c.Shape, 19, 0.06, 0.03),
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
	compare(t, got[output].Data, want[output].Data, 2e-5)
}

func TestExecutorMambaBlockMatchesReference(t *testing.T) {
	cudatest.Require(t)
	builder := tensor.NewBuilder()
	spec := model.Spec{CommonSpec: model.CommonSpec{Architecture: "mamba", EmbeddingLength: 4, RMSNormEpsilon: 1e-5}, RecurrentSpec: model.RecurrentSpec{SSMConvKernel: 3, SSMInnerSize: 8, SSMStateSize: 2, SSMTimeStepRank: 2,
		SSMDtBCNorm: true}}
	input := builder.Input("input", dtype.F32, tensor.MustShape(4, 3))
	weights := model.LayerGraphWeights{
		AttentionNorm:     builder.Input("norm", dtype.F32, tensor.MustShape(4)),
		SSMInput:          builder.Input("in", dtype.F32, tensor.MustShape(4, 16)),
		SSMConv1D:         builder.Input("conv", dtype.F32, tensor.MustShape(3, 8)),
		SSMConv1DBias:     builder.Input("conv_bias", dtype.F32, tensor.MustShape(8)),
		SSMX:              builder.Input("x", dtype.F32, tensor.MustShape(8, 6)),
		SSMTimeStepWeight: builder.Input("dt", dtype.F32, tensor.MustShape(2, 8)),
		SSMTimeStep:       builder.Input("dt_bias", dtype.F32, tensor.MustShape(8)),
		SSMA:              builder.Input("a", dtype.F32, tensor.MustShape(2, 8)),
		SSMD:              builder.Input("d", dtype.F32, tensor.MustShape(8)),
		SSMOutput:         builder.Input("out", dtype.F32, tensor.MustShape(8, 4)),
	}
	convState := builder.Input("conv_state", dtype.F32, tensor.MustShape(2, 8))
	ssmState := builder.Input("ssm_state", dtype.F32, tensor.MustShape(2, 8))
	plan := spec.PlanLayer(0, false)
	result, err := model.BuildArchitectureBlockCached(model.BlockDispatchOptions{
		Spec: spec, Weights: weights, Plan: &plan,
		Context: model.CachedBlockContext{
			Builder: builder, Input: input, PastKey: convState, PastValue: ssmState,
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	feeds := map[*tensor.Tensor]reference.Value{
		input:                     patternedValue(input.Shape, 3, 0.08, -0.2),
		weights.AttentionNorm:     patternedValue(weights.AttentionNorm.Shape, 5, 0.03, 0.9),
		weights.SSMInput:          patternedValue(weights.SSMInput.Shape, 7, 0.04, -0.1),
		weights.SSMConv1D:         patternedValue(weights.SSMConv1D.Shape, 11, 0.03, -0.05),
		weights.SSMConv1DBias:     patternedValue(weights.SSMConv1DBias.Shape, 13, 0.02, -0.1),
		weights.SSMX:              patternedValue(weights.SSMX.Shape, 17, 0.03, -0.1),
		weights.SSMTimeStepWeight: patternedValue(weights.SSMTimeStepWeight.Shape, 19, 0.02, -0.05),
		weights.SSMTimeStep:       patternedValue(weights.SSMTimeStep.Shape, 23, 0.02, -0.2),
		weights.SSMA:              patternedValue(weights.SSMA.Shape, 29, 0.01, -0.5),
		weights.SSMD:              patternedValue(weights.SSMD.Shape, 31, 0.02, 0.1),
		weights.SSMOutput:         patternedValue(weights.SSMOutput.Shape, 37, 0.03, -0.1),
		convState:                 patternedValue(convState.Shape, 41, 0.02, -0.05),
		ssmState:                  patternedValue(ssmState.Shape, 43, 0.02, -0.03),
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

func TestExecutorMamba2BlockMatchesReference(t *testing.T) {
	cudatest.Require(t)
	builder := tensor.NewBuilder()
	spec := model.Spec{CommonSpec: model.CommonSpec{Architecture: "mamba2", EmbeddingLength: 4, RMSNormEpsilon: 1e-5}, RecurrentSpec: model.RecurrentSpec{SSMConvKernel: 3, SSMInnerSize: 8, SSMStateSize: 2, SSMTimeStepRank: 4,
		SSMGroupCount: 2}}
	input := builder.Input("input", dtype.F32, tensor.MustShape(4, 3))
	weights := model.LayerGraphWeights{
		AttentionNorm: builder.Input("norm", dtype.F32, tensor.MustShape(4)),
		SSMInput:      builder.Input("in", dtype.F32, tensor.MustShape(4, 28)),
		SSMConv1D:     builder.Input("conv", dtype.F32, tensor.MustShape(3, 16)),
		SSMConv1DBias: builder.Input("conv_bias", dtype.F32, tensor.MustShape(16)),
		SSMTimeStep:   builder.Input("dt_bias", dtype.F32, tensor.MustShape(4)),
		SSMA:          builder.Input("a", dtype.F32, tensor.MustShape(1, 4)),
		SSMD:          builder.Input("d", dtype.F32, tensor.MustShape(1, 4)),
		SSMNorm:       builder.Input("ssm_norm", dtype.F32, tensor.MustShape(4, 2)),
		SSMOutput:     builder.Input("out", dtype.F32, tensor.MustShape(8, 4)),
	}
	convState := builder.Input("conv_state", dtype.F32, tensor.MustShape(2, 16))
	ssmState := builder.Input("ssm_state", dtype.F32, tensor.MustShape(2, 8))
	plan := spec.PlanLayer(0, false)
	result, err := model.BuildArchitectureBlockCached(model.BlockDispatchOptions{
		Spec: spec, Weights: weights, Plan: &plan,
		Context: model.CachedBlockContext{
			Builder: builder, Input: input, PastKey: convState, PastValue: ssmState,
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	feeds := map[*tensor.Tensor]reference.Value{
		input:                 patternedValue(input.Shape, 3, 0.08, -0.2),
		weights.AttentionNorm: patternedValue(weights.AttentionNorm.Shape, 5, 0.03, 0.9),
		weights.SSMInput:      patternedValue(weights.SSMInput.Shape, 7, 0.04, -0.1),
		weights.SSMConv1D:     patternedValue(weights.SSMConv1D.Shape, 11, 0.03, -0.05),
		weights.SSMConv1DBias: patternedValue(weights.SSMConv1DBias.Shape, 13, 0.02, -0.1),
		weights.SSMTimeStep:   patternedValue(weights.SSMTimeStep.Shape, 17, 0.02, -0.2),
		weights.SSMA:          patternedValue(weights.SSMA.Shape, 19, 0.01, -0.5),
		weights.SSMD:          patternedValue(weights.SSMD.Shape, 23, 0.02, 0.1),
		weights.SSMNorm:       patternedValue(weights.SSMNorm.Shape, 29, 0.03, 0.9),
		weights.SSMOutput:     patternedValue(weights.SSMOutput.Shape, 31, 0.03, -0.1),
		convState:             patternedValue(convState.Shape, 37, 0.02, -0.05),
		ssmState:              patternedValue(ssmState.Shape, 41, 0.02, -0.03),
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

func TestExecutorFalconH1BlockMatchesReference(t *testing.T) {
	cudatest.Require(t)
	fixturePositions := []uint32{0, 1, 2}
	builder := tensor.NewBuilder()
	spec := model.Spec{CommonSpec: model.CommonSpec{Architecture: "falcon-h1", EmbeddingLength: 4, FeedForwardLength: 6,

		RMSNormEpsilon: 1e-5}, AttentionSpec: model.AttentionSpec{HeadCount: 2, HeadCountKV: 1, KeyLength: 2, ValueLength: 2, RopeDimensionCount: 2,
		RopeFrequencyBase: 10000}, RecurrentSpec: model.RecurrentSpec{SSMConvKernel: 3,
		SSMInnerSize: 8, SSMStateSize: 2, SSMTimeStepRank: 4, SSMGroupCount: 2}}
	input := builder.Input("input", dtype.F32, tensor.MustShape(4, 3))
	weights := model.LayerGraphWeights{
		AttentionNorm:       builder.Input("norm", dtype.F32, tensor.MustShape(4)),
		AttentionQ:          builder.Input("q", dtype.F32, tensor.MustShape(4, 4)),
		AttentionK:          builder.Input("k", dtype.F32, tensor.MustShape(4, 2)),
		AttentionV:          builder.Input("v", dtype.F32, tensor.MustShape(4, 2)),
		AttentionOutput:     builder.Input("attn_out", dtype.F32, tensor.MustShape(4, 4)),
		AttentionOutputBias: builder.Input("attn_out_bias", dtype.F32, tensor.MustShape(4)),
		SSMInput:            builder.Input("ssm_in", dtype.F32, tensor.MustShape(4, 28)),
		SSMConv1D:           builder.Input("conv", dtype.F32, tensor.MustShape(3, 16)),
		SSMConv1DBias:       builder.Input("conv_bias", dtype.F32, tensor.MustShape(16)),
		SSMTimeStep:         builder.Input("dt", dtype.F32, tensor.MustShape(4)),
		SSMA:                builder.Input("a", dtype.F32, tensor.MustShape(1, 4)),
		SSMD:                builder.Input("d", dtype.F32, tensor.MustShape(1, 4)),
		SSMNorm:             builder.Input("ssm_norm", dtype.F32, tensor.MustShape(4, 2)),
		SSMOutput:           builder.Input("ssm_out", dtype.F32, tensor.MustShape(8, 4)),
		FeedForwardNorm:     builder.Input("ffn_norm", dtype.F32, tensor.MustShape(4)),
		FeedForwardGate:     builder.Input("gate", dtype.F32, tensor.MustShape(4, 6)),
		FeedForwardUp:       builder.Input("up", dtype.F32, tensor.MustShape(4, 6)),
		FeedForwardDown:     builder.Input("down", dtype.F32, tensor.MustShape(6, 4)),
		FeedForwardGateBias: builder.Input("gate_bias", dtype.F32, tensor.MustShape(6)),
		FeedForwardUpBias:   builder.Input("up_bias", dtype.F32, tensor.MustShape(6)),
		FeedForwardDownBias: builder.Input("down_bias", dtype.F32, tensor.MustShape(4)),
	}
	convState := builder.Input("conv_state", dtype.F32, tensor.MustShape(2, 16))
	ssmState := builder.Input("ssm_state", dtype.F32, tensor.MustShape(2, 8))
	plan := spec.PlanLayer(0, false)
	result, err := model.BuildArchitectureBlockCached(model.BlockDispatchOptions{
		Spec: spec, Weights: weights, Plan: &plan,
		Context: model.CachedBlockContext{
			Builder: builder, Input: input, Positions: fixturePositions,
			PastStates: model.CacheStates[*tensor.Tensor]{
				model.CacheStateConvolution: {Mode: model.CacheStateFixed, Value: convState},
				model.CacheStateSSM:         {Mode: model.CacheStateFixed, Value: ssmState},
			},
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	feeds := make(map[*tensor.Tensor]reference.Value)
	for index, node := range builder.Nodes() {
		if node.Op != tensor.OpInput {
			continue
		}
		offset := float32(-0.1)
		if node == weights.AttentionNorm || node == weights.SSMNorm || node == weights.FeedForwardNorm {
			offset = 0.9
		}
		feeds[node] = patternedValue(node.Shape, index+3, 0.025, offset)
	}
	outputs := []*tensor.Tensor{
		result.Output, result.Key, result.Value,
		result.States[model.CacheStateConvolution].Value,
		result.States[model.CacheStateSSM].Value,
	}
	want, err := reference.Execute(outputs, feeds)
	if err != nil {
		t.Fatal(err)
	}
	cuda := newFixtureExecutor(t)
	got, err := cuda.Execute(context.Background(), outputs, feeds)
	if err != nil {
		t.Fatal(err)
	}
	for _, output := range outputs {
		compare(t, got[output].Data, want[output].Data, 1e-3)
	}
}

func TestExecutorGraniteHybridRecurrentBlockMatchesReference(t *testing.T) {
	cudatest.Require(t)
	builder := tensor.NewBuilder()
	spec := model.Spec{CommonSpec: model.CommonSpec{Architecture: "granitehybrid", EmbeddingLength: 4, FeedForwardLength: 6,
		RMSNormEpsilon: 1e-5, ResidualScale: 0.5}, RecurrentSpec: model.RecurrentSpec{SSMConvKernel: 3, SSMInnerSize: 8,
		SSMStateSize: 2, SSMTimeStepRank: 4, SSMGroupCount: 2}}
	input := builder.Input("input", dtype.F32, tensor.MustShape(4, 3))
	weights := model.LayerGraphWeights{
		AttentionNorm:       builder.Input("norm", dtype.F32, tensor.MustShape(4)),
		SSMInput:            builder.Input("in", dtype.F32, tensor.MustShape(4, 28)),
		SSMConv1D:           builder.Input("conv", dtype.F32, tensor.MustShape(3, 16)),
		SSMTimeStep:         builder.Input("dt_bias", dtype.F32, tensor.MustShape(4)),
		SSMA:                builder.Input("a", dtype.F32, tensor.MustShape(1, 4)),
		SSMD:                builder.Input("d", dtype.F32, tensor.MustShape(1, 4)),
		SSMNorm:             builder.Input("ssm_norm", dtype.F32, tensor.MustShape(4, 2)),
		SSMOutput:           builder.Input("out", dtype.F32, tensor.MustShape(8, 4)),
		FeedForwardNorm:     builder.Input("ffn_norm", dtype.F32, tensor.MustShape(4)),
		FeedForwardGate:     builder.Input("ffn_gate", dtype.F32, tensor.MustShape(4, 6)),
		FeedForwardUp:       builder.Input("ffn_up", dtype.F32, tensor.MustShape(4, 6)),
		FeedForwardDown:     builder.Input("ffn_down", dtype.F32, tensor.MustShape(6, 4)),
		FeedForwardGateBias: builder.Input("ffn_gate_bias", dtype.F32, tensor.MustShape(6)),
		FeedForwardUpBias:   builder.Input("ffn_up_bias", dtype.F32, tensor.MustShape(6)),
		FeedForwardDownBias: builder.Input("ffn_down_bias", dtype.F32, tensor.MustShape(4)),
	}
	convState := builder.Input("conv_state", dtype.F32, tensor.MustShape(2, 16))
	ssmState := builder.Input("ssm_state", dtype.F32, tensor.MustShape(2, 8))
	plan := spec.PlanLayer(0, true)
	result, err := model.BuildArchitectureBlockCached(model.BlockDispatchOptions{
		Spec: spec, Weights: weights, Plan: &plan,
		Context: model.CachedBlockContext{
			Builder: builder, Input: input, PastKey: convState, PastValue: ssmState,
			Recurrent: true,
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	feeds := map[*tensor.Tensor]reference.Value{}
	for index, node := range []*tensor.Tensor{
		input, weights.AttentionNorm, weights.SSMInput, weights.SSMConv1D, weights.SSMTimeStep,
		weights.SSMA, weights.SSMD, weights.SSMNorm, weights.SSMOutput, weights.FeedForwardNorm,
		weights.FeedForwardGate, weights.FeedForwardUp, weights.FeedForwardDown,
		weights.FeedForwardGateBias, weights.FeedForwardUpBias, weights.FeedForwardDownBias,
		convState, ssmState,
	} {
		offset := float32(-0.15)
		if node == weights.AttentionNorm || node == weights.SSMNorm || node == weights.FeedForwardNorm {
			offset = 0.9
		}
		feeds[node] = patternedValue(node.Shape, index+3, 0.025, offset)
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

func TestExecutorPLaMo2RecurrentBlockMatchesReference(t *testing.T) {
	cudatest.Require(t)
	builder := tensor.NewBuilder()
	spec := model.Spec{CommonSpec: model.CommonSpec{Architecture: "plamo2", EmbeddingLength: 4, FeedForwardLength: 6,
		RMSNormEpsilon: 1e-5}, RecurrentSpec: model.RecurrentSpec{SSMConvKernel: 3, SSMInnerSize: 8, SSMStateSize: 2,
		SSMTimeStepRank: 4}}
	input := builder.Input("input", dtype.F32, tensor.MustShape(4, 3))
	weights := model.LayerGraphWeights{
		AttentionNorm:       builder.Input("norm", dtype.F32, tensor.MustShape(4)),
		AttentionPostNorm:   builder.Input("post_norm", dtype.F32, tensor.MustShape(4)),
		SSMInput:            builder.Input("in", dtype.F32, tensor.MustShape(4, 16)),
		SSMConv1D:           builder.Input("conv", dtype.F32, tensor.MustShape(3, 8)),
		SSMX:                builder.Input("x", dtype.F32, tensor.MustShape(8, 68)),
		SSMTimeStepWeight:   builder.Input("dt", dtype.F32, tensor.MustShape(64, 4)),
		SSMTimeStep:         builder.Input("dt_bias", dtype.F32, tensor.MustShape(4)),
		SSMTimeStepNorm:     builder.Input("dt_norm", dtype.F32, tensor.MustShape(64)),
		SSMA:                builder.Input("a", dtype.F32, tensor.MustShape(4)),
		SSMD:                builder.Input("d", dtype.F32, tensor.MustShape(4)),
		SSMBNorm:            builder.Input("b_norm", dtype.F32, tensor.MustShape(2)),
		SSMCNorm:            builder.Input("c_norm", dtype.F32, tensor.MustShape(2)),
		SSMOutput:           builder.Input("out", dtype.F32, tensor.MustShape(8, 4)),
		FeedForwardNorm:     builder.Input("ffn_norm", dtype.F32, tensor.MustShape(4)),
		FeedForwardUp:       builder.Input("ffn_up", dtype.F32, tensor.MustShape(4, 12)),
		FeedForwardDown:     builder.Input("ffn_down", dtype.F32, tensor.MustShape(6, 4)),
		FeedForwardPostNorm: builder.Input("ffn_post", dtype.F32, tensor.MustShape(4)),
	}
	convState := builder.Input("conv_state", dtype.F32, tensor.MustShape(2, 8))
	ssmState := builder.Input("ssm_state", dtype.F32, tensor.MustShape(2, 8))
	plan := spec.PlanLayer(0, true)
	result, err := model.BuildArchitectureBlockCached(model.BlockDispatchOptions{
		Spec: spec, Weights: weights, Plan: &plan,
		Context: model.CachedBlockContext{
			Builder: builder, Input: input, PastKey: convState, PastValue: ssmState,
			Recurrent: true,
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	feeds := map[*tensor.Tensor]reference.Value{}
	for index, node := range []*tensor.Tensor{
		input, weights.AttentionNorm, weights.AttentionPostNorm, weights.SSMInput, weights.SSMConv1D,
		weights.SSMX, weights.SSMTimeStepWeight, weights.SSMTimeStep, weights.SSMTimeStepNorm,
		weights.SSMA, weights.SSMD, weights.SSMBNorm, weights.SSMCNorm, weights.SSMOutput,
		weights.FeedForwardNorm, weights.FeedForwardUp, weights.FeedForwardDown,
		weights.FeedForwardPostNorm, convState, ssmState,
	} {
		offset := float32(-0.15)
		if node == weights.AttentionNorm || node == weights.AttentionPostNorm ||
			node == weights.SSMTimeStepNorm || node == weights.SSMBNorm || node == weights.SSMCNorm ||
			node == weights.FeedForwardNorm || node == weights.FeedForwardPostNorm {
			offset = 0.9
		}
		feeds[node] = patternedValue(node.Shape, index+3, 0.02, offset)
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

func TestExecutorJambaRecurrentMoEBlockMatchesReference(t *testing.T) {
	cudatest.Require(t)
	builder := tensor.NewBuilder()
	spec := model.Spec{CommonSpec: model.CommonSpec{Architecture: "jamba", EmbeddingLength: 4, FeedForwardLength: 6,
		RMSNormEpsilon: 1e-5}, MoESpec: model.MoESpec{ExpertCount: 4, ExpertUsedCount: 2,
		ExpertFeedForward: 6, ExpertWeightsScale: 1}, RecurrentSpec: model.RecurrentSpec{SSMConvKernel: 3, SSMInnerSize: 8, SSMStateSize: 2,
		SSMTimeStepRank: 2, SSMGroupCount: 1},
	}
	input := builder.Input("input", dtype.F32, tensor.MustShape(4, 3))
	weights := model.LayerGraphWeights{
		AttentionNorm:          builder.Input("norm", dtype.F32, tensor.MustShape(4)),
		SSMInput:               builder.Input("in", dtype.F32, tensor.MustShape(4, 16)),
		SSMConv1D:              builder.Input("conv", dtype.F32, tensor.MustShape(3, 8)),
		SSMConv1DBias:          builder.Input("conv_bias", dtype.F32, tensor.MustShape(8)),
		SSMX:                   builder.Input("x", dtype.F32, tensor.MustShape(8, 6)),
		SSMTimeStepNorm:        builder.Input("dt_norm", dtype.F32, tensor.MustShape(2)),
		SSMTimeStepWeight:      builder.Input("dt", dtype.F32, tensor.MustShape(2, 8)),
		SSMTimeStep:            builder.Input("dt_bias", dtype.F32, tensor.MustShape(8)),
		SSMBNorm:               builder.Input("b_norm", dtype.F32, tensor.MustShape(2)),
		SSMCNorm:               builder.Input("c_norm", dtype.F32, tensor.MustShape(2)),
		SSMA:                   builder.Input("a", dtype.F32, tensor.MustShape(2, 8)),
		SSMD:                   builder.Input("d", dtype.F32, tensor.MustShape(8)),
		SSMOutput:              builder.Input("out", dtype.F32, tensor.MustShape(8, 4)),
		FeedForwardNorm:        builder.Input("ffn_norm", dtype.F32, tensor.MustShape(4)),
		FeedForwardRouter:      builder.Input("router", dtype.F32, tensor.MustShape(4, 4)),
		FeedForwardGateExperts: builder.Input("gate_exps", dtype.F32, tensor.MustShape(4, 6, 4)),
		FeedForwardUpExperts:   builder.Input("up_exps", dtype.F32, tensor.MustShape(4, 6, 4)),
		FeedForwardDownExperts: builder.Input("down_exps", dtype.F32, tensor.MustShape(6, 4, 4)),
	}
	convState := builder.Input("conv_state", dtype.F32, tensor.MustShape(2, 8))
	ssmState := builder.Input("ssm_state", dtype.F32, tensor.MustShape(2, 8))
	plan := spec.PlanLayer(0, true)
	result, err := model.BuildArchitectureBlockCached(model.BlockDispatchOptions{
		Spec: spec, Weights: weights, Plan: &plan,
		Context: model.CachedBlockContext{
			Builder: builder, Input: input, PastKey: convState, PastValue: ssmState,
			Recurrent: true,
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	feeds := map[*tensor.Tensor]reference.Value{}
	for index, node := range []*tensor.Tensor{
		input, weights.AttentionNorm, weights.SSMInput, weights.SSMConv1D, weights.SSMConv1DBias,
		weights.SSMX, weights.SSMTimeStepNorm, weights.SSMTimeStepWeight, weights.SSMTimeStep,
		weights.SSMBNorm, weights.SSMCNorm, weights.SSMA, weights.SSMD, weights.SSMOutput,
		weights.FeedForwardNorm, weights.FeedForwardRouter, weights.FeedForwardGateExperts,
		weights.FeedForwardUpExperts, weights.FeedForwardDownExperts, convState, ssmState,
	} {
		offset := float32(-0.15)
		if node == weights.AttentionNorm || node == weights.SSMTimeStepNorm || node == weights.SSMBNorm ||
			node == weights.SSMCNorm || node == weights.FeedForwardNorm {
			offset = 0.9
		}
		feeds[node] = patternedValue(node.Shape, index+3, 0.025, offset)
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

func TestExecutorGatedDeltaNetMatchesReference(t *testing.T) {
	cudatest.Require(t)
	for _, repeatInterleave := range []bool{false, true} {
		for _, gateWidth := range []uint64{1, 4} {
			t.Run(fmt.Sprintf("gate_width_%d/repeat_%t", gateWidth, repeatInterleave), func(t *testing.T) {
				builder := tensor.NewBuilder()
				q := builder.Input("q", dtype.F32, tensor.MustShape(4, 1, 3, 2))
				k := builder.Input("k", dtype.F32, tensor.MustShape(4, 1, 3, 2))
				v := builder.Input("v", dtype.F32, tensor.MustShape(4, 2, 3, 2))
				gate := builder.Input("gate", dtype.F32, tensor.MustShape(gateWidth, 2, 3, 2))
				beta := builder.Input("beta", dtype.F32, tensor.MustShape(1, 2, 3, 2))
				state := builder.Input("state", dtype.F32, tensor.MustShape(4, 4, 2, 2))
				output := builder.GatedDeltaNet(q, k, v, gate, beta, state)
				if repeatInterleave {
					output = builder.GatedDeltaNetRepeatInterleave(q, k, v, gate, beta, state)
				}
				if err := builder.Err(); err != nil {
					t.Fatal(err)
				}
				feeds := map[*tensor.Tensor]reference.Value{
					q:     patternedValue(q.Shape, 7, 0.08, -0.15),
					k:     patternedValue(k.Shape, 11, 0.06, -0.1),
					v:     patternedValue(v.Shape, 13, 0.09, 0.03),
					gate:  patternedValue(gate.Shape, 5, 0.02, -0.08),
					beta:  patternedValue(beta.Shape, 3, 0.03, 0.4),
					state: patternedValue(state.Shape, 17, 0.04, -0.07),
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
			})
		}
	}
}

func TestExecutorQwen35LayoutOperationsMatchReference(t *testing.T) {
	cudatest.Require(t)
	builder := tensor.NewBuilder()
	input := builder.Input("input", dtype.F32, tensor.MustShape(16, 5))
	query := builder.GroupSlice(input, 0, 2, 4, 4)
	gate := builder.GroupSlice(input, 2, 2, 4, 4)
	transposed := builder.Transpose2D(input)
	state := builder.Input("state", dtype.F32, tensor.MustShape(3, 16))
	convInput := builder.Concat(state, transposed, 0)
	slice := builder.FlatSlice(convInput, 7, 35)
	if err := builder.Err(); err != nil {
		t.Fatal(err)
	}
	feeds := map[*tensor.Tensor]reference.Value{
		input: patternedValue(input.Shape, 11, 0.13, -0.2),
		state: patternedValue(state.Shape, 7, 0.09, 0.05),
	}
	outputs := []*tensor.Tensor{query, gate, transposed, convInput, slice}
	want, err := reference.Execute(outputs, feeds)
	if err != nil {
		t.Fatal(err)
	}
	cuda := newFixtureExecutor(t)
	got, err := cuda.Execute(context.Background(), outputs, feeds)
	if err != nil {
		t.Fatal(err)
	}
	for _, output := range outputs {
		compare(t, got[output].Data, want[output].Data, 0)
	}
}

func TestExecutorQwen35BlocksMatchReference(t *testing.T) {
	cudatest.Require(t)
	for _, architecture := range []string{"qwen35", "qwen35moe", "qwen3next", "qwen3next-legacy"} {
		for _, recurrent := range []bool{false, true} {
			if architecture == "qwen3next-legacy" && !recurrent {
				continue
			}
			name := architecture + "/attention"
			if recurrent {
				name = architecture + "/recurrent"
			}
			t.Run(name, func(t *testing.T) {
				builder := tensor.NewBuilder()
				spec := modeltest.Qwen35().Spec
				if architecture == "qwen35moe" || architecture == "qwen3next" || architecture == "qwen3next-legacy" {
					spec.Architecture = architecture
					if architecture == "qwen3next-legacy" {
						spec.Architecture = "qwen3next"
					}
					spec.ExpertCount = 4
					spec.ExpertUsedCount = 2
					spec.ExpertFeedForward = 6
					spec.SharedExpertFF = 10
					spec.ExpertWeightsScale = 1.25
				}
				input := builder.Input("input", dtype.F32, tensor.MustShape(8, 2))
				weights, feeds := qwen35ExecutorWeights(builder, spec, recurrent)
				if architecture == "qwen3next-legacy" {
					legacyQKVZ := builder.Input("ssm_in", dtype.F32, tensor.MustShape(8, 12))
					feeds[legacyQKVZ] = patternedValue(legacyQKVZ.Shape, 61, 0.02, -0.03)
					weights.AttentionQKV = legacyQKVZ
					weights.AttentionGate = nil
				}
				feeds[input] = patternedValue(input.Shape, 17, 0.04, -0.08)
				var convState, ssmState *tensor.Tensor
				if recurrent {
					convState = builder.Input("conv_state", dtype.F32, tensor.MustShape(2, 8))
					ssmState = builder.Input("ssm_state", dtype.F32, tensor.MustShape(2, 2, 2, 1))
					feeds[convState] = patternedValue(convState.Shape, 7, 0.03, -0.02)
					feeds[ssmState] = patternedValue(ssmState.Shape, 11, 0.02, 0.01)
				}
				plan := spec.PlanLayer(0, recurrent)
				var result model.DenseBlockResult
				var err error
				if !recurrent && architecture != "qwen3next" {
					positions := [4][]uint32{{10, 11}, {20, 21}, {30, 31}, {40, 41}}
					result, err = model.BuildArchitectureBlockCached(model.BlockDispatchOptions{
						Spec: spec, Weights: weights, Plan: &plan,
						Context: model.CachedBlockContext{
							Builder: builder, Input: input, Positions: positions[0], MultiPositions: &positions,
							Sequences: 1, CacheWrite: tensor.CacheWriteConcat,
						},
					})
				} else {
					result, err = model.BuildArchitectureBlockCached(model.BlockDispatchOptions{
						Spec: spec, Weights: weights, Plan: &plan,
						Context: model.CachedBlockContext{
							Builder: builder, Input: input, Positions: []uint32{0, 1},
							PastKey: convState, PastValue: ssmState, Recurrent: recurrent,
							Sequences: 1, CacheWrite: tensor.CacheWriteConcat,
						},
					})
				}
				if err != nil {
					t.Fatal(err)
				}
				outputs := []*tensor.Tensor{result.Output}
				outputs = append(outputs, result.Key, result.Value)
				want, err := reference.Execute(outputs, feeds)
				if err != nil {
					t.Fatal(err)
				}
				cuda := newFixtureExecutor(t)
				got, err := cuda.Execute(context.Background(), outputs, feeds)
				if err != nil {
					t.Fatal(err)
				}
				for _, output := range outputs {
					compare(t, got[output].Data, want[output].Data, 4e-5)
				}
			})
		}
	}
}

func TestExecutorQwen35MTPMatchesReference(t *testing.T) {
	cudatest.Require(t)
	builder := tensor.NewBuilder()
	spec := modeltest.Qwen35().Spec
	spec.Architecture = "qwen35moe"
	spec.NextNPredictLayers = 1
	dense := spec
	dense.Architecture = "qwen35"
	token := builder.Input("mtp_token", dtype.F32, tensor.MustShape(8, 1))
	hidden := builder.Input("mtp_hidden", dtype.F32, tensor.MustShape(8, 1))
	embeddingNorm := builder.Input("mtp_enorm", dtype.F32, tensor.MustShape(8))
	hiddenNorm := builder.Input("mtp_hnorm", dtype.F32, tensor.MustShape(8))
	projection := builder.Input("mtp_eh", dtype.F32, tensor.MustShape(16, 8))
	weights, feeds := qwen35ExecutorWeights(builder, dense, false)
	for node, value := range map[*tensor.Tensor]reference.Value{
		token:         patternedValue(token.Shape, 7, 0.03, -0.04),
		hidden:        patternedValue(hidden.Shape, 11, 0.04, 0.02),
		embeddingNorm: patternedValue(embeddingNorm.Shape, 5, 0.02, 0.8),
		hiddenNorm:    patternedValue(hiddenNorm.Shape, 3, 0.02, 0.9),
		projection:    patternedValue(projection.Shape, 17, 0.015, -0.05),
	} {
		feeds[node] = value
	}
	current, err := model.BuildQwen35MTPInput(
		builder, token, hidden, embeddingNorm, hiddenNorm, projection, spec,
	)
	if err != nil {
		t.Fatal(err)
	}
	block, err := model.BuildQwen35MTPBlockCached(
		builder, current, spec, weights, []uint32{19}, nil, nil, dense.PlanLayer(0, false),
	)
	if err != nil {
		t.Fatal(err)
	}
	outputNorm := builder.Input("mtp_output_norm", dtype.F32, tensor.MustShape(8))
	head := builder.Input("mtp_head", dtype.F32, tensor.MustShape(8, 17))
	feeds[outputNorm] = patternedValue(outputNorm.Shape, 5, 0.02, 0.85)
	feeds[head] = patternedValue(head.Shape, 23, 0.01, -0.03)
	logits, nextHidden, err := model.BuildQwen35MTPOutputs(
		builder, block.Output, outputNorm, head, spec,
	)
	if err != nil {
		t.Fatal(err)
	}
	outputs := []*tensor.Tensor{logits, nextHidden, block.Key, block.Value}
	want, err := reference.Execute(outputs, feeds)
	if err != nil {
		t.Fatal(err)
	}
	cuda := newFixtureExecutor(t)
	got, err := cuda.Execute(context.Background(), outputs, feeds)
	if err != nil {
		t.Fatal(err)
	}
	for _, output := range outputs {
		compare(t, got[output].Data, want[output].Data, 4e-5)
	}
}

func TestExecutorKimiLinearKDABlockMatchesReference(t *testing.T) {
	cudatest.Require(t)
	builder := tensor.NewBuilder()
	spec := model.Spec{CommonSpec: model.CommonSpec{Architecture: "kimi-linear", EmbeddingLength: 8, FeedForwardLength: 12,

		RMSNormEpsilon: 1e-6}, AttentionSpec: model.AttentionSpec{HeadCount: 2, HeadCountKV: 1, KeyLength: 4, ValueLength: 2,
		RopeDisabled: true, RopeDimensionCount: 2,
		KVLoRARank: 3}, MoESpec: model.MoESpec{LeadingDenseBlocks: 1}, RecurrentSpec: model.RecurrentSpec{KDAHeadDim: 2, SSMConvKernel: 3, SSMInnerSize: 4},
	}
	input := builder.Input("input", dtype.F32, tensor.MustShape(8, 2))
	weights := model.LayerGraphWeights{
		AttentionNorm:   builder.Input("attn_norm", dtype.F32, tensor.MustShape(8)),
		AttentionQ:      builder.Input("q", dtype.F32, tensor.MustShape(8, 4)),
		AttentionK:      builder.Input("k", dtype.F32, tensor.MustShape(8, 4)),
		AttentionV:      builder.Input("v", dtype.F32, tensor.MustShape(8, 4)),
		AttentionOutput: builder.Input("o", dtype.F32, tensor.MustShape(4, 8)),
		SSMQueryConv:    builder.Input("cq", dtype.F32, tensor.MustShape(3, 1, 4, 1)),
		SSMKeyConv:      builder.Input("ck", dtype.F32, tensor.MustShape(3, 1, 4, 1)),
		SSMValueConv:    builder.Input("cv", dtype.F32, tensor.MustShape(3, 1, 4, 1)),
		SSMForgetA:      builder.Input("fa", dtype.F32, tensor.MustShape(8, 2)),
		SSMForgetB:      builder.Input("fb", dtype.F32, tensor.MustShape(2, 4)),
		SSMBeta:         builder.Input("beta", dtype.F32, tensor.MustShape(8, 2)),
		SSMA:            builder.Input("a", dtype.F32, tensor.MustShape(1, 2, 1, 1)),
		SSMTimeStep:     builder.Input("dt", dtype.F32, tensor.MustShape(4)),
		SSMOutputGateA:  builder.Input("ga", dtype.F32, tensor.MustShape(8, 2)),
		SSMOutputGateB:  builder.Input("gb", dtype.F32, tensor.MustShape(2, 4)),
		SSMNorm:         builder.Input("ssm_norm", dtype.F32, tensor.MustShape(2)),
		FeedForwardNorm: builder.Input("ffn_norm", dtype.F32, tensor.MustShape(8)),
		FeedForwardGate: builder.Input("ffn_gate", dtype.F32, tensor.MustShape(8, 12)),
		FeedForwardUp:   builder.Input("ffn_up", dtype.F32, tensor.MustShape(8, 12)),
		FeedForwardDown: builder.Input("ffn_down", dtype.F32, tensor.MustShape(12, 8)),
	}
	conv := builder.Input("conv", dtype.F32, tensor.MustShape(2, 12))
	state := builder.Input("state", dtype.F32, tensor.MustShape(2, 2, 2, 1))
	result, err := buildFixtureCachedBlock(
		builder, input, spec, weights, []uint32{0, 1}, conv, state, 0, true,
	)
	if err != nil {
		t.Fatal(err)
	}
	feeds := map[*tensor.Tensor]reference.Value{
		input:                   patternedValue(input.Shape, 3, 0.08, -0.03),
		weights.AttentionNorm:   patternedValue(weights.AttentionNorm.Shape, 5, 0.02, 1),
		weights.SSMA:            patternedValue(weights.SSMA.Shape, 7, 0.01, -0.1),
		weights.SSMTimeStep:     patternedValue(weights.SSMTimeStep.Shape, 9, 0.02, 0.1),
		weights.SSMNorm:         patternedValue(weights.SSMNorm.Shape, 11, 0.02, 1),
		weights.FeedForwardNorm: patternedValue(weights.FeedForwardNorm.Shape, 13, 0.02, 1),
		conv:                    patternedValue(conv.Shape, 15, 0.02, -0.01),
		state:                   patternedValue(state.Shape, 17, 0.01, 0),
	}
	for index, node := range []*tensor.Tensor{
		weights.AttentionQ, weights.AttentionK, weights.AttentionV, weights.AttentionOutput,
		weights.SSMQueryConv, weights.SSMKeyConv, weights.SSMValueConv,
		weights.SSMForgetA, weights.SSMForgetB, weights.SSMBeta,
		weights.SSMOutputGateA, weights.SSMOutputGateB,
		weights.FeedForwardGate, weights.FeedForwardUp, weights.FeedForwardDown,
	} {
		feeds[node] = patternedValue(node.Shape, index+19, 0.025, 0)
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

func TestExecutorRWKV6Qwen2BlockMatchesReference(t *testing.T) {
	cudatest.Require(t)
	builder := tensor.NewBuilder()
	spec := model.Spec{CommonSpec: model.CommonSpec{Architecture: "rwkv6qwen2", EmbeddingLength: 8, FeedForwardLength: 12,

		RMSNormEpsilon: 1e-6}, AttentionSpec: model.AttentionSpec{HeadCount: 2, HeadCountKV: 1}, RecurrentSpec: model.RecurrentSpec{WKVHeadSize: 4, TimeMixExtraDim: 3,
		TimeDecayExtraDim: 2},
	}
	input := builder.Input("input", dtype.F32, tensor.MustShape(8, 2))
	weights := model.LayerGraphWeights{
		AttentionNorm:     builder.Input("attn_norm", dtype.F32, tensor.MustShape(8)),
		TimeMixW1:         builder.Input("mix_w1", dtype.F32, tensor.MustShape(8, 15)),
		TimeMixW2:         builder.Input("mix_w2", dtype.F32, tensor.MustShape(3, 8, 5)),
		TimeMixLerpX:      builder.Input("lerp_x", dtype.F32, tensor.MustShape(8, 1, 1)),
		TimeMixLerpFused:  builder.Input("lerp", dtype.F32, tensor.MustShape(8, 1, 1, 5)),
		TimeMixDecay:      builder.Input("decay", dtype.F32, tensor.MustShape(8)),
		TimeMixDecayW1:    builder.Input("decay_w1", dtype.F32, tensor.MustShape(8, 2)),
		TimeMixDecayW2:    builder.Input("decay_w2", dtype.F32, tensor.MustShape(2, 8)),
		TimeMixKey:        builder.Input("key", dtype.F32, tensor.MustShape(8, 4)),
		TimeMixValue:      builder.Input("value", dtype.F32, tensor.MustShape(8, 4)),
		TimeMixReceptance: builder.Input("receptance", dtype.F32, tensor.MustShape(8, 8)),
		TimeMixGate:       builder.Input("gate", dtype.F32, tensor.MustShape(8, 8)),
		TimeMixOutput:     builder.Input("output", dtype.F32, tensor.MustShape(8, 8)),
		FeedForwardNorm:   builder.Input("ffn_norm", dtype.F32, tensor.MustShape(8)),
		FeedForwardGate:   builder.Input("ffn_gate", dtype.F32, tensor.MustShape(8, 12)),
		FeedForwardUp:     builder.Input("ffn_up", dtype.F32, tensor.MustShape(8, 12)),
		FeedForwardDown:   builder.Input("ffn_down", dtype.F32, tensor.MustShape(12, 8)),
	}
	shift := builder.Input("shift", dtype.F32, tensor.MustShape(8))
	state := builder.Input("state", dtype.F32, tensor.MustShape(4, 4, 2, 1))
	result, err := buildFixtureCachedBlock(builder, input, spec, weights, nil, shift, state, 0, false)
	if err != nil {
		t.Fatal(err)
	}
	feeds := map[*tensor.Tensor]reference.Value{
		input:                    patternedValue(input.Shape, 3, 0.05, -0.02),
		weights.AttentionNorm:    patternedValue(weights.AttentionNorm.Shape, 5, 0.02, 1),
		weights.TimeMixLerpX:     patternedValue(weights.TimeMixLerpX.Shape, 7, 0.01, 0.2),
		weights.TimeMixLerpFused: patternedValue(weights.TimeMixLerpFused.Shape, 9, 0.01, 0.1),
		weights.TimeMixDecay:     patternedValue(weights.TimeMixDecay.Shape, 11, 0.02, -1),
		weights.FeedForwardNorm:  patternedValue(weights.FeedForwardNorm.Shape, 13, 0.02, 1),
		shift:                    patternedValue(shift.Shape, 15, 0.02, 0),
		state:                    patternedValue(state.Shape, 17, 0.01, 0),
	}
	for index, node := range []*tensor.Tensor{
		weights.TimeMixW1, weights.TimeMixW2, weights.TimeMixDecayW1, weights.TimeMixDecayW2,
		weights.TimeMixKey, weights.TimeMixValue, weights.TimeMixReceptance,
		weights.TimeMixGate, weights.TimeMixOutput,
		weights.FeedForwardGate, weights.FeedForwardUp, weights.FeedForwardDown,
	} {
		feeds[node] = patternedValue(node.Shape, index+19, 0.02, 0)
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

func TestExecutorRWKV6BlockMatchesReference(t *testing.T) {
	cudatest.Require(t)
	builder := tensor.NewBuilder()
	spec := model.Spec{CommonSpec: model.CommonSpec{Architecture: "rwkv6", EmbeddingLength: 8, FeedForwardLength: 12,

		LayerNormEpsilon: 1e-5}, AttentionSpec: model.AttentionSpec{HeadCount: 2, HeadCountKV: 2}, RecurrentSpec: model.RecurrentSpec{WKVHeadSize: 4, TimeMixExtraDim: 3,
		TimeDecayExtraDim: 2},
	}
	input := builder.Input("input", dtype.F32, tensor.MustShape(8, 2))
	weights := model.LayerGraphWeights{
		AttentionNorm:        builder.Input("attn_norm", dtype.F32, tensor.MustShape(8)),
		AttentionNormBias:    builder.Input("attn_norm_bias", dtype.F32, tensor.MustShape(8)),
		AttentionNorm2:       builder.Input("attn_norm_2", dtype.F32, tensor.MustShape(8)),
		AttentionNorm2Bias:   builder.Input("attn_norm_2_bias", dtype.F32, tensor.MustShape(8)),
		TimeMixW1:            builder.Input("mix_w1", dtype.F32, tensor.MustShape(8, 15)),
		TimeMixW2:            builder.Input("mix_w2", dtype.F32, tensor.MustShape(3, 8, 5)),
		TimeMixLerpX:         builder.Input("lerp_x", dtype.F32, tensor.MustShape(8, 1, 1)),
		TimeMixLerpFused:     builder.Input("lerp", dtype.F32, tensor.MustShape(8, 1, 1, 5)),
		TimeMixFirst:         builder.Input("first", dtype.F32, tensor.MustShape(4, 2)),
		TimeMixDecay:         builder.Input("decay", dtype.F32, tensor.MustShape(8)),
		TimeMixDecayW1:       builder.Input("decay_w1", dtype.F32, tensor.MustShape(8, 2)),
		TimeMixDecayW2:       builder.Input("decay_w2", dtype.F32, tensor.MustShape(2, 8)),
		TimeMixKey:           builder.Input("key", dtype.F32, tensor.MustShape(8, 8)),
		TimeMixValue:         builder.Input("value", dtype.F32, tensor.MustShape(8, 8)),
		TimeMixReceptance:    builder.Input("receptance", dtype.F32, tensor.MustShape(8, 8)),
		TimeMixGate:          builder.Input("gate", dtype.F32, tensor.MustShape(8, 8)),
		TimeMixLN:            builder.Input("mix_ln", dtype.F32, tensor.MustShape(8)),
		TimeMixLNBias:        builder.Input("mix_ln_bias", dtype.F32, tensor.MustShape(8)),
		TimeMixOutput:        builder.Input("output", dtype.F32, tensor.MustShape(8, 8)),
		ChannelMixLerpK:      builder.Input("channel_lerp_k", dtype.F32, tensor.MustShape(8, 1, 1)),
		ChannelMixLerpR:      builder.Input("channel_lerp_r", dtype.F32, tensor.MustShape(8, 1, 1)),
		ChannelMixKey:        builder.Input("channel_key", dtype.F32, tensor.MustShape(8, 12)),
		ChannelMixValue:      builder.Input("channel_value", dtype.F32, tensor.MustShape(12, 8)),
		ChannelMixReceptance: builder.Input("channel_receptance", dtype.F32, tensor.MustShape(8, 8)),
	}
	shift := builder.Input("shift", dtype.F32, tensor.MustShape(8, 2))
	state := builder.Input("state", dtype.F32, tensor.MustShape(4, 4, 2, 1))
	result, err := buildFixtureCachedBlock(builder, input, spec, weights, nil, shift, state, 0, false)
	if err != nil {
		t.Fatal(err)
	}
	feeds := map[*tensor.Tensor]reference.Value{
		input:                      patternedValue(input.Shape, 3, 0.05, -0.02),
		weights.AttentionNorm:      patternedValue(weights.AttentionNorm.Shape, 5, 0.02, 1),
		weights.AttentionNormBias:  patternedValue(weights.AttentionNormBias.Shape, 7, 0.01, 0),
		weights.AttentionNorm2:     patternedValue(weights.AttentionNorm2.Shape, 9, 0.02, 1),
		weights.AttentionNorm2Bias: patternedValue(weights.AttentionNorm2Bias.Shape, 11, 0.01, 0),
		weights.TimeMixLerpX:       patternedValue(weights.TimeMixLerpX.Shape, 13, 0.01, 0.2),
		weights.TimeMixLerpFused:   patternedValue(weights.TimeMixLerpFused.Shape, 15, 0.01, 0.1),
		weights.TimeMixFirst:       patternedValue(weights.TimeMixFirst.Shape, 17, 0.01, 0.2),
		weights.TimeMixDecay:       patternedValue(weights.TimeMixDecay.Shape, 19, 0.02, -1),
		weights.TimeMixLN:          patternedValue(weights.TimeMixLN.Shape, 21, 0.02, 1),
		weights.TimeMixLNBias:      patternedValue(weights.TimeMixLNBias.Shape, 23, 0.01, 0),
		weights.ChannelMixLerpK:    patternedValue(weights.ChannelMixLerpK.Shape, 25, 0.01, 0.2),
		weights.ChannelMixLerpR:    patternedValue(weights.ChannelMixLerpR.Shape, 27, 0.01, 0.2),
		shift:                      patternedValue(shift.Shape, 29, 0.02, 0),
		state:                      patternedValue(state.Shape, 31, 0.01, 0),
	}
	for index, node := range []*tensor.Tensor{
		weights.TimeMixW1, weights.TimeMixW2, weights.TimeMixDecayW1, weights.TimeMixDecayW2,
		weights.TimeMixKey, weights.TimeMixValue, weights.TimeMixReceptance, weights.TimeMixGate,
		weights.TimeMixOutput, weights.ChannelMixKey, weights.ChannelMixValue, weights.ChannelMixReceptance,
	} {
		feeds[node] = patternedValue(node.Shape, index+33, 0.02, 0)
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

func TestExecutorRWKV7OpsMatchReference(t *testing.T) {
	cudatest.Require(t)
	builder := tensor.NewBuilder()
	shape := tensor.MustShape(4, 2, 3, 1)
	receptance := builder.Input("receptance", dtype.F32, shape)
	decay := builder.Input("decay", dtype.F32, shape)
	key := builder.Input("key", dtype.F32, shape)
	value := builder.Input("value", dtype.F32, shape)
	a := builder.Input("a", dtype.F32, shape)
	bVector := builder.Input("b", dtype.F32, shape)
	state := builder.Input("state", dtype.F32, tensor.MustShape(4, 4, 2, 1))
	packed := builder.RWKV7(receptance, decay, key, value, a, bVector, state)
	reduced := builder.SumRows(builder.Multiply(key, receptance))
	feeds := map[*tensor.Tensor]reference.Value{
		receptance: patternedValue(shape, 3, 0.02, 0.1),
		decay:      patternedValue(shape, 5, 0.01, 0.8),
		key:        patternedValue(shape, 7, 0.02, -0.1),
		value:      patternedValue(shape, 9, 0.02, 0.05),
		a:          patternedValue(shape, 11, 0.01, 0.02),
		bVector:    patternedValue(shape, 13, 0.01, -0.03),
		state:      patternedValue(state.Shape, 15, 0.01, 0.04),
	}
	outputs := []*tensor.Tensor{packed, reduced}
	want, err := reference.Execute(outputs, feeds)
	if err != nil {
		t.Fatal(err)
	}
	cuda := newFixtureExecutor(t)
	got, err := cuda.Execute(context.Background(), outputs, feeds)
	if err != nil {
		t.Fatal(err)
	}
	compare(t, got[packed].Data, want[packed].Data, 1e-5)
	compare(t, got[reduced].Data, want[reduced].Data, 1e-6)
}

func TestExecutorKimiLinearMLABlockMatchesReference(t *testing.T) {
	cudatest.Require(t)
	builder := tensor.NewBuilder()
	spec := model.Spec{CommonSpec: model.CommonSpec{Architecture: "kimi-linear", EmbeddingLength: 8, FeedForwardLength: 12,

		RMSNormEpsilon: 1e-6}, AttentionSpec: model.AttentionSpec{HeadCount: 2, HeadCountKV: 1, KeyLength: 4, ValueLength: 2,
		RopeDisabled: true, RopeDimensionCount: 2,
		KVLoRARank: 3}, MoESpec: model.MoESpec{LeadingDenseBlocks: 1, ExpertCount: 4, ExpertUsedCount: 2,
		ExpertFeedForward: 6, SharedExpertFF: 6, ExpertGatingFunc: 2,
		ExpertWeightsScale: 1.25, ExpertWeightsNorm: true},
	}
	input := builder.Input("input", dtype.F32, tensor.MustShape(8, 2))
	weights := model.LayerGraphWeights{
		AttentionNorm:          builder.Input("attn_norm", dtype.F32, tensor.MustShape(8)),
		AttentionQ:             builder.Input("q", dtype.F32, tensor.MustShape(8, 8)),
		AttentionKVAMQA:        builder.Input("kva", dtype.F32, tensor.MustShape(8, 5)),
		AttentionKVANorm:       builder.Input("kva_norm", dtype.F32, tensor.MustShape(3)),
		AttentionKB:            builder.Input("kb", dtype.F32, tensor.MustShape(2, 3, 2)),
		AttentionVB:            builder.Input("vb", dtype.F32, tensor.MustShape(3, 2, 2)),
		AttentionOutput:        builder.Input("o", dtype.F32, tensor.MustShape(4, 8)),
		FeedForwardNorm:        builder.Input("ffn_norm", dtype.F32, tensor.MustShape(8)),
		FeedForwardRouter:      builder.Input("router", dtype.F32, tensor.MustShape(8, 4)),
		FeedForwardGateExperts: builder.Input("eg", dtype.F32, tensor.MustShape(8, 6, 4)),
		FeedForwardUpExperts:   builder.Input("eu", dtype.F32, tensor.MustShape(8, 6, 4)),
		FeedForwardDownExperts: builder.Input("ed", dtype.F32, tensor.MustShape(6, 8, 4)),
		FeedForwardExpertBias:  builder.Input("eb", dtype.F32, tensor.MustShape(4)),
		FeedForwardSharedGate:  builder.Input("sg", dtype.F32, tensor.MustShape(8, 6)),
		FeedForwardSharedUp:    builder.Input("su", dtype.F32, tensor.MustShape(8, 6)),
		FeedForwardSharedDown:  builder.Input("sd", dtype.F32, tensor.MustShape(6, 8)),
	}
	result, err := buildFixtureCachedBlock(
		builder, input, spec, weights, []uint32{0, 1}, nil, nil, 1, false,
	)
	if err != nil {
		t.Fatal(err)
	}
	feeds := map[*tensor.Tensor]reference.Value{
		input:                         patternedValue(input.Shape, 3, 0.08, -0.03),
		weights.AttentionNorm:         patternedValue(weights.AttentionNorm.Shape, 5, 0.02, 1),
		weights.AttentionKVANorm:      patternedValue(weights.AttentionKVANorm.Shape, 7, 0.02, 1),
		weights.FeedForwardNorm:       patternedValue(weights.FeedForwardNorm.Shape, 9, 0.02, 1),
		weights.FeedForwardExpertBias: {Shape: weights.FeedForwardExpertBias.Shape, Data: []float32{0.1, -0.2, 0.3, -0.1}},
	}
	for index, node := range []*tensor.Tensor{
		weights.AttentionQ, weights.AttentionKVAMQA, weights.AttentionKB, weights.AttentionVB,
		weights.AttentionOutput, weights.FeedForwardRouter, weights.FeedForwardGateExperts,
		weights.FeedForwardUpExperts, weights.FeedForwardDownExperts,
		weights.FeedForwardSharedGate, weights.FeedForwardSharedUp, weights.FeedForwardSharedDown,
	} {
		feeds[node] = patternedValue(node.Shape, index+11, 0.025, 0)
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

func TestExecutorRoPEMultiMatchesReference(t *testing.T) {
	cudatest.Require(t)
	builder := tensor.NewBuilder()
	input := builder.Input("input", dtype.F32, tensor.MustShape(12, 3, 4, 2))
	positions := [4][]uint32{
		{0, 1, 2, 3},
		{3, 5, 7, 9},
		{2, 4, 6, 8},
		{11, 13, 17, 19},
	}
	output := builder.RoPEMultiScaled(
		input,
		positions,
		[4]int32{2, 2, 2, 2},
		8,
		1_000_000,
		0.25,
	)
	if err := builder.Err(); err != nil {
		t.Fatal(err)
	}
	feeds := map[*tensor.Tensor]reference.Value{
		input: patternedValue(input.Shape, 17, 0.08, -0.2),
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
	compare(t, got[output].Data, want[output].Data, 2e-6)
}

func qwen35ExecutorWeights(
	builder *tensor.Builder,
	spec model.Spec,
	recurrent bool,
) (model.LayerGraphWeights, map[*tensor.Tensor]reference.Value) {
	feeds := make(map[*tensor.Tensor]reference.Value)
	seed := 1
	input := func(name string, shape tensor.Shape, scale, bias float32) *tensor.Tensor {
		node := builder.Input(name, dtype.F32, shape)
		feeds[node] = patternedValue(shape, seed, scale, bias)
		seed += 2
		return node
	}
	embedding := uint64(spec.EmbeddingLength)
	result := model.LayerGraphWeights{
		AttentionNorm: input("attn_norm", tensor.MustShape(embedding), 0.03, 0.9),
		FeedForwardNorm: input(
			"post_attention_norm",
			tensor.MustShape(embedding),
			0.03,
			0.9,
		),
	}
	if spec.Architecture == "qwen35moe" || spec.Architecture == "qwen3next" {
		expertWidth := uint64(spec.ExpertFeedForward)
		experts := uint64(spec.ExpertCount)
		sharedWidth := uint64(spec.SharedExpertFF)
		result.FeedForwardRouter = input(
			"ffn_router", tensor.MustShape(embedding, experts), 0.03, -0.02,
		)
		result.FeedForwardGateUpExperts = input(
			"ffn_gate_up_exps",
			tensor.MustShape(embedding, 2*expertWidth, experts),
			0.02,
			-0.03,
		)
		result.FeedForwardDownExperts = input(
			"ffn_down_exps",
			tensor.MustShape(expertWidth, embedding, experts),
			0.02,
			0.01,
		)
		result.FeedForwardSharedRouter = input(
			"ffn_shared_router", tensor.MustShape(embedding), 0.03, -0.01,
		)
		result.FeedForwardSharedGate = input(
			"ffn_shared_gate", tensor.MustShape(embedding, sharedWidth), 0.02, -0.02,
		)
		result.FeedForwardSharedUp = input(
			"ffn_shared_up", tensor.MustShape(embedding, sharedWidth), 0.02, 0.01,
		)
		result.FeedForwardSharedDown = input(
			"ffn_shared_down", tensor.MustShape(sharedWidth, embedding), 0.02, -0.01,
		)
	} else {
		feedForward := uint64(spec.FeedForwardLength)
		result.FeedForwardGate = input(
			"ffn_gate", tensor.MustShape(embedding, feedForward), 0.025, -0.04,
		)
		result.FeedForwardUp = input(
			"ffn_up", tensor.MustShape(embedding, feedForward), 0.02, 0.03,
		)
		result.FeedForwardDown = input(
			"ffn_down", tensor.MustShape(feedForward, embedding), 0.02, -0.01,
		)
	}
	if !recurrent {
		headWidth := uint64(spec.KeyLength)
		result.AttentionQ = input(
			"attn_q",
			tensor.MustShape(embedding, 2*uint64(spec.HeadCount)*headWidth),
			0.02,
			-0.03,
		)
		result.AttentionK = input(
			"attn_k",
			tensor.MustShape(embedding, uint64(spec.HeadCountKV)*headWidth),
			0.02,
			0.01,
		)
		result.AttentionV = input(
			"attn_v",
			tensor.MustShape(embedding, uint64(spec.HeadCountKV)*uint64(spec.ValueLength)),
			0.025,
			-0.02,
		)
		result.AttentionOutput = input(
			"attn_output",
			tensor.MustShape(embedding, embedding),
			0.02,
			0.01,
		)
		result.AttentionQNorm = input("attn_q_norm", tensor.MustShape(headWidth), 0.02, 0.95)
		result.AttentionKNorm = input("attn_k_norm", tensor.MustShape(headWidth), 0.02, 0.95)
		return result, feeds
	}
	stateWidth := uint64(spec.SSMStateSize)
	keyDimension := stateWidth * uint64(spec.SSMGroupCount)
	valueDimension := uint64(spec.SSMInnerSize)
	channels := 2*keyDimension + valueDimension
	valueHeads := uint64(spec.SSMTimeStepRank)
	result.AttentionQKV = input(
		"attn_qkv",
		tensor.MustShape(embedding, channels),
		0.025,
		-0.03,
	)
	result.AttentionGate = input(
		"attn_gate",
		tensor.MustShape(embedding, valueDimension),
		0.02,
		0.01,
	)
	result.SSMConv1D = input(
		"ssm_conv1d",
		tensor.MustShape(uint64(spec.SSMConvKernel), channels),
		0.03,
		-0.02,
	)
	result.SSMTimeStep = input("ssm_dt", tensor.MustShape(valueHeads), 0.02, -0.1)
	result.SSMA = input("ssm_a", tensor.MustShape(valueHeads), 0.01, -0.5)
	result.SSMBeta = input(
		"ssm_beta",
		tensor.MustShape(embedding, valueHeads),
		0.025,
		0.02,
	)
	result.SSMAlpha = input(
		"ssm_alpha",
		tensor.MustShape(embedding, valueHeads),
		0.02,
		-0.01,
	)
	if spec.Architecture == "qwen3next" {
		result.SSMBeta = nil
		result.SSMAlpha = nil
		result.SSMBetaAlpha = input(
			"ssm_ba", tensor.MustShape(embedding, 2*valueHeads), 0.02, -0.01,
		)
	}
	result.SSMNorm = input("ssm_norm", tensor.MustShape(stateWidth), 0.02, 0.95)
	result.SSMOutput = input(
		"ssm_out",
		tensor.MustShape(valueDimension, embedding),
		0.025,
		-0.01,
	)
	return result, feeds
}

func TestExecutorEmbeddingBroadcastSwiGLUAndRoPEMatchesReference(t *testing.T) {
	cudatest.Require(t)
	builder := tensor.NewBuilder()
	table := builder.Input("table", dtype.F32, tensor.MustShape(4, 3))
	weight := builder.Input("weight", dtype.F32, tensor.MustShape(4))
	up := builder.Input("up", dtype.F32, tensor.MustShape(4, 2))
	embedding := builder.GetRows(table, []uint32{2, 0})
	swiglu := builder.SwiGLU(builder.WeightedRMSNorm(embedding, weight, 1e-5), up)
	ropeInput := builder.Input("rope", dtype.F32, tensor.MustShape(4, 1, 2))
	ropeFactors := builder.Input("rope_factors", dtype.F32, tensor.MustShape(2))
	rope := builder.RoPENeoX(ropeInput, []uint32{0, 17}, 4, 1_000_000)
	normalRope := builder.RoPENormal(ropeInput, []uint32{0, 17}, 4, 1_000_000)
	factoredRope := builder.RoPENormalScaledWithFactors(
		ropeInput,
		[]uint32{0, 17},
		4,
		1_000_000,
		0.25,
		ropeFactors,
	)
	factoredYaRNRope := builder.RoPENormalYaRNWithFactors(
		ropeInput, []uint32{0, 17}, 4, 8, 10_000, 0.25, 1, 0.9, 16, 2,
		ropeFactors,
	)
	yarnRope := builder.RoPENeoXYaRN(
		ropeInput, []uint32{0, 17}, 4, 8, 10_000, 0.25, 1, 1, 32, 1,
	)
	if err := builder.Err(); err != nil {
		t.Fatal(err)
	}
	tableValue, _ := reference.NewValue(table.Shape, []float32{
		1, 2, 3, 4,
		5, 6, 7, 8,
		9, 10, 11, 12,
	})
	weightValue, _ := reference.NewValue(weight.Shape, []float32{1, 2, 3, 4})
	upValue, _ := reference.NewValue(up.Shape, []float32{1, 2, 3, 4, 4, 3, 2, 1})
	ropeValue, _ := reference.NewValue(ropeInput.Shape, []float32{
		1, 2, 3, 4,
		-1, -2, -3, -4,
	})
	ropeFactorValue, _ := reference.NewValue(ropeFactors.Shape, []float32{1, 8})
	feeds := map[*tensor.Tensor]reference.Value{
		table:       tableValue,
		weight:      weightValue,
		up:          upValue,
		ropeInput:   ropeValue,
		ropeFactors: ropeFactorValue,
	}
	outputs := []*tensor.Tensor{embedding, swiglu, rope, normalRope, factoredRope, factoredYaRNRope, yarnRope}
	want, err := reference.Execute(outputs, feeds)
	if err != nil {
		t.Fatal(err)
	}
	cuda := newFixtureExecutor(t)
	got, err := cuda.Execute(context.Background(), outputs, feeds)
	if err != nil {
		t.Fatal(err)
	}
	compare(t, got[embedding].Data, want[embedding].Data, 0)
	compare(t, got[swiglu].Data, want[swiglu].Data, 3e-5)
	compare(t, got[rope].Data, want[rope].Data, 3e-5)
	compare(t, got[normalRope].Data, want[normalRope].Data, 3e-5)
	compare(t, got[factoredRope].Data, want[factoredRope].Data, 3e-5)
	compare(t, got[factoredYaRNRope].Data, want[factoredYaRNRope].Data, 3e-5)
	compare(t, got[yarnRope].Data, want[yarnRope].Data, 3e-5)
}

func TestExecutorDenseQwen3BlockMatchesReference(t *testing.T) {
	cudatest.Require(t)
	builder := tensor.NewBuilder()
	spec := model.Spec{CommonSpec: model.CommonSpec{Architecture: "qwen3",
		EmbeddingLength:   8,
		FeedForwardLength: 12,

		RMSNormEpsilon: 1e-6}, AttentionSpec: model.AttentionSpec{HeadCount: 2,
		HeadCountKV:       1,
		KeyLength:         4,
		ValueLength:       4,
		RopeFrequencyBase: 1_000_000},
	}
	input := builder.Input("input", dtype.F32, tensor.MustShape(8, 3))
	weights := model.LayerGraphWeights{
		AttentionNorm:   builder.Input("attn_norm", dtype.F32, tensor.MustShape(8)),
		AttentionQ:      builder.Input("attn_q", dtype.F32, tensor.MustShape(8, 8)),
		AttentionK:      builder.Input("attn_k", dtype.F32, tensor.MustShape(8, 4)),
		AttentionV:      builder.Input("attn_v", dtype.F32, tensor.MustShape(8, 4)),
		AttentionOutput: builder.Input("attn_output", dtype.F32, tensor.MustShape(8, 8)),
		AttentionQNorm:  builder.Input("attn_q_norm", dtype.F32, tensor.MustShape(4)),
		AttentionKNorm:  builder.Input("attn_k_norm", dtype.F32, tensor.MustShape(4)),
		AttentionQBias:  builder.Input("attn_q_bias", dtype.F32, tensor.MustShape(8)),
		AttentionKBias:  builder.Input("attn_k_bias", dtype.F32, tensor.MustShape(4)),
		AttentionVBias:  builder.Input("attn_v_bias", dtype.F32, tensor.MustShape(4)),
		AttentionOutputBias: builder.Input(
			"attn_output_bias",
			dtype.F32,
			tensor.MustShape(8),
		),
		FeedForwardNorm: builder.Input("ffn_norm", dtype.F32, tensor.MustShape(8)),
		FeedForwardGate: builder.Input("ffn_gate", dtype.F32, tensor.MustShape(8, 12)),
		FeedForwardUp:   builder.Input("ffn_up", dtype.F32, tensor.MustShape(8, 12)),
		FeedForwardDown: builder.Input("ffn_down", dtype.F32, tensor.MustShape(12, 8)),
		FeedForwardGateBias: builder.Input(
			"ffn_gate_bias",
			dtype.F32,
			tensor.MustShape(12),
		),
		FeedForwardUpBias: builder.Input(
			"ffn_up_bias",
			dtype.F32,
			tensor.MustShape(12),
		),
		FeedForwardDownBias: builder.Input(
			"ffn_down_bias",
			dtype.F32,
			tensor.MustShape(8),
		),
	}
	output, err := buildFixtureDenseBlock(builder, input, spec, weights, []uint32{0, 1, 2})
	if err != nil {
		t.Fatal(err)
	}
	feeds := make(map[*tensor.Tensor]reference.Value)
	feeds[input] = patternedValue(input.Shape, 1, 0.25, 0)
	for index, node := range []*tensor.Tensor{
		weights.AttentionQ,
		weights.AttentionK,
		weights.AttentionV,
		weights.AttentionOutput,
		weights.FeedForwardGate,
		weights.FeedForwardUp,
		weights.FeedForwardDown,
	} {
		feeds[node] = patternedValue(node.Shape, index+2, 0.08, 0)
	}
	for index, node := range []*tensor.Tensor{
		weights.AttentionNorm,
		weights.AttentionQNorm,
		weights.AttentionKNorm,
		weights.FeedForwardNorm,
	} {
		feeds[node] = patternedValue(node.Shape, index+13, 0.03, 1)
	}
	for index, node := range []*tensor.Tensor{
		weights.AttentionQBias,
		weights.AttentionKBias,
		weights.AttentionVBias,
		weights.AttentionOutputBias,
		weights.FeedForwardGateBias,
		weights.FeedForwardUpBias,
		weights.FeedForwardDownBias,
	} {
		feeds[node] = patternedValue(node.Shape, index+20, 0.02, -0.01)
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
	compare(t, got[output].Data, want[output].Data, 3e-4)
}

func TestExecutorDenseOLMo2BlockMatchesReference(t *testing.T) {
	cudatest.Require(t)
	builder := tensor.NewBuilder()
	spec := model.Spec{CommonSpec: model.CommonSpec{Architecture: "olmo2",
		BlockCount:        4,
		EmbeddingLength:   8,
		FeedForwardLength: 12,

		RMSNormEpsilon: 1e-6}, AttentionSpec: model.AttentionSpec{HeadCount: 2,
		HeadCountKV:       1,
		KeyLength:         4,
		ValueLength:       4,
		RopeFrequencyBase: 10000,
		RopeFrequencySWA:  10000,

		SlidingWindow:  2,
		SlidingPattern: 4},
	}
	input := builder.Input("input", dtype.F32, tensor.MustShape(8, 2))
	weights := model.LayerGraphWeights{
		AttentionQ:        builder.Input("attn_q", dtype.F32, tensor.MustShape(8, 8)),
		AttentionK:        builder.Input("attn_k", dtype.F32, tensor.MustShape(8, 4)),
		AttentionV:        builder.Input("attn_v", dtype.F32, tensor.MustShape(8, 4)),
		AttentionOutput:   builder.Input("attn_output", dtype.F32, tensor.MustShape(8, 8)),
		AttentionQNorm:    builder.Input("attn_q_norm", dtype.F32, tensor.MustShape(8)),
		AttentionKNorm:    builder.Input("attn_k_norm", dtype.F32, tensor.MustShape(4)),
		AttentionPostNorm: builder.Input("post_attention_norm", dtype.F32, tensor.MustShape(8)),
		FeedForwardGate:   builder.Input("ffn_gate", dtype.F32, tensor.MustShape(8, 12)),
		FeedForwardUp:     builder.Input("ffn_up", dtype.F32, tensor.MustShape(8, 12)),
		FeedForwardDown:   builder.Input("ffn_down", dtype.F32, tensor.MustShape(12, 8)),
		FeedForwardPostNorm: builder.Input(
			"post_ffw_norm",
			dtype.F32,
			tensor.MustShape(8),
		),
	}
	result, err := buildFixtureDenseBlockCachedForLayer(
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
		weights.AttentionQNorm,
		weights.AttentionKNorm,
		weights.AttentionPostNorm,
		weights.FeedForwardPostNorm,
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

func TestExecutorDenseOLMoClampedQKVMatchesReference(t *testing.T) {
	cudatest.Require(t)
	builder := tensor.NewBuilder()
	spec := model.Spec{CommonSpec: model.CommonSpec{Architecture: "olmo", EmbeddingLength: 8, FeedForwardLength: 12,

		LayerNormEpsilon: 1e-5}, AttentionSpec: model.AttentionSpec{HeadCount: 2, HeadCountKV: 1, KeyLength: 4, ValueLength: 4,
		RopeFrequencyBase: 10000, AttentionClamp: 0.25},
	}
	input := builder.Input("input", dtype.F32, tensor.MustShape(8, 2))
	weights := model.LayerGraphWeights{
		AttentionQ:      builder.Input("attn_q", dtype.F32, tensor.MustShape(8, 8)),
		AttentionK:      builder.Input("attn_k", dtype.F32, tensor.MustShape(8, 4)),
		AttentionV:      builder.Input("attn_v", dtype.F32, tensor.MustShape(8, 4)),
		AttentionOutput: builder.Input("attn_output", dtype.F32, tensor.MustShape(8, 8)),
		FeedForwardGate: builder.Input("ffn_gate", dtype.F32, tensor.MustShape(8, 12)),
		FeedForwardUp:   builder.Input("ffn_up", dtype.F32, tensor.MustShape(8, 12)),
		FeedForwardDown: builder.Input("ffn_down", dtype.F32, tensor.MustShape(12, 8)),
	}
	result, err := buildFixtureDenseBlockCachedForLayer(
		builder, input, spec, weights, []uint32{0, 1}, nil, nil, 0,
	)
	if err != nil {
		t.Fatal(err)
	}
	feeds := map[*tensor.Tensor]reference.Value{input: patternedValue(input.Shape, 1, 0.2, 0)}
	for index, node := range []*tensor.Tensor{
		weights.AttentionQ, weights.AttentionK, weights.AttentionV, weights.AttentionOutput,
		weights.FeedForwardGate, weights.FeedForwardUp, weights.FeedForwardDown,
	} {
		feeds[node] = patternedValue(node.Shape, index+2, 0.08, 0)
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
	compare(t, got[result.Output].Data, want[result.Output].Data, 4e-4)
	compare(t, got[result.Key].Data, want[result.Key].Data, 5e-5)
	compare(t, got[result.Value].Data, want[result.Value].Data, 5e-5)
}
