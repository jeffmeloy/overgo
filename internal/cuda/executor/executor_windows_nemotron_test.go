//go:build windows

package executor

import (
	"testing"

	"overgo/internal/model"
	"overgo/internal/modeltest"
	"overgo/internal/tensor"
	"overgo/internal/tensor/dtype"
	"overgo/internal/tensor/reference"
)

const nemotronFixtureTokens = 2

func TestExecutorNemotronHRecurrentBlockMatchesReference(t *testing.T) {
	const (
		fixtureSeedBase = 3
		valueScale      = 0.04
	)
	builder := tensor.NewBuilder()
	spec := modeltest.NemotronHRecurrent().Spec
	positions := fixturePositions(nemotronFixtureTokens)
	embedding := uint64(spec.EmbeddingLength)
	inner := uint64(spec.SSMInnerSize)
	state := uint64(spec.SSMStateSize)
	groups := uint64(spec.SSMGroupCount)
	stepHeads := uint64(spec.SSMTimeStepRank)
	convChannels := inner + 2*groups*state
	inputProjection := 2*inner + 2*groups*state + stepHeads
	input := builder.Input("input", dtype.F32, tensor.MustShape(embedding, nemotronFixtureTokens))
	convState := builder.Input("conv_state", dtype.F32, tensor.MustShape(
		uint64(spec.SSMConvKernel-1), convChannels,
	))
	ssmState := builder.Input("ssm_state", dtype.F32, tensor.MustShape(state, inner))
	weights := model.LayerGraphWeights{
		AttentionNorm: builder.Input("norm", dtype.F32, tensor.MustShape(embedding)),
		SSMInput:      builder.Input("ssm_in", dtype.F32, tensor.MustShape(embedding, inputProjection)),
		SSMConv1D:     builder.Input("conv", dtype.F32, tensor.MustShape(uint64(spec.SSMConvKernel), convChannels)),
		SSMConv1DBias: builder.Input("conv_bias", dtype.F32, tensor.MustShape(convChannels)),
		SSMTimeStep:   builder.Input("dt_bias", dtype.F32, tensor.MustShape(stepHeads)),
		SSMA:          builder.Input("a", dtype.F32, tensor.MustShape(1, stepHeads)),
		SSMD:          builder.Input("d", dtype.F32, tensor.MustShape(1, stepHeads)),
		SSMNorm:       builder.Input("ssm_norm", dtype.F32, tensor.MustShape(stepHeads, state)),
		SSMOutput:     builder.Input("ssm_out", dtype.F32, tensor.MustShape(inner, embedding)),
	}
	plan := spec.PlanLayer(0, true)
	result, err := buildCompiledLayer(model.BlockDispatchOptions{
		Spec: spec, Weights: weights, Plan: &plan,
		Context: model.CachedBlockContext{
			Builder: builder, Input: input, Positions: positions,
			PastKey: convState, PastValue: ssmState, Recurrent: true,
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	feeds := make(map[*tensor.Tensor]reference.Value)
	for index, node := range []*tensor.Tensor{
		input, convState, ssmState, weights.AttentionNorm, weights.SSMInput, weights.SSMConv1D,
		weights.SSMConv1DBias, weights.SSMTimeStep, weights.SSMA, weights.SSMD, weights.SSMNorm,
		weights.SSMOutput,
	} {
		offset := float32(0)
		if node == weights.AttentionNorm || node == weights.SSMNorm {
			offset = 1
		}
		feeds[node] = patternedValue(node.Shape, fixtureSeedBase+index, valueScale, offset)
	}
	checkCUDAGraph(t, feeds,
		graphOutputCheck{output: result.Output, tolerance: accuracyRecurrent},
		graphOutputCheck{output: result.Key, tolerance: accuracyState},
		graphOutputCheck{output: result.Value, tolerance: accuracyRecurrent},
	)
}

func TestExecutorNemotronHMoEBlockMatchesReference(t *testing.T) {
	const (
		inputSeed      = 3
		normSeed       = 5
		weightSeedBase = 11
		inputScale     = 0.15
		normScale      = 0.03
		weightScale    = 0.06
	)
	builder := tensor.NewBuilder()
	spec := modeltest.NemotronHMoE().Spec
	positions := fixturePositions(nemotronFixtureTokens)
	shapes := spec.TensorShapes(0)
	latent := uint64(spec.MoELatentSize)
	shared := uint64(spec.SharedExpertFF)
	input := builder.Input("input", dtype.F32, tensor.MustShape(shapes.Embedding, nemotronFixtureTokens))
	weights := model.LayerGraphWeights{
		AttentionNorm: builder.Input("norm", dtype.F32, tensor.MustShape(shapes.Embedding)),
		FeedForwardRouter: builder.Input("router", dtype.F32, tensor.MustShape(
			shapes.Embedding, shapes.Experts,
		)),
		FeedForwardExpertBias: builder.Input("bias", dtype.F32, tensor.MustShape(shapes.Experts)),
		FeedForwardLatentDown: builder.Input("latent_down", dtype.F32, tensor.MustShape(
			shapes.Embedding, latent,
		)),
		FeedForwardLatentUp: builder.Input("latent_up", dtype.F32, tensor.MustShape(
			latent, shapes.Embedding,
		)),
		FeedForwardUpExperts: builder.Input("up_exps", dtype.F32, tensor.MustShape(
			latent, shapes.ExpertWidth, shapes.Experts,
		)),
		FeedForwardDownExperts: builder.Input("down_exps", dtype.F32, tensor.MustShape(
			shapes.ExpertWidth, latent, shapes.Experts,
		)),
		FeedForwardSharedUp: builder.Input("shared_up", dtype.F32, tensor.MustShape(
			shapes.Embedding, shared,
		)),
		FeedForwardSharedDown: builder.Input("shared_down", dtype.F32, tensor.MustShape(
			shared, shapes.Embedding,
		)),
	}
	plan := spec.PlanLayer(0, false)
	result, err := buildCompiledLayer(model.BlockDispatchOptions{
		Spec: spec, Weights: weights, Plan: &plan,
		Context: model.CachedBlockContext{Builder: builder, Input: input, Positions: positions},
	})
	if err != nil {
		t.Fatal(err)
	}
	feeds := map[*tensor.Tensor]reference.Value{
		input: patternedValue(input.Shape, inputSeed, inputScale, 0),
		weights.AttentionNorm: patternedValue(
			weights.AttentionNorm.Shape, normSeed, normScale, 1,
		),
		weights.FeedForwardExpertBias: {
			Shape: weights.FeedForwardExpertBias.Shape,
			Data:  []float32{0.1, -0.2, 0.3, -0.1},
		},
	}
	for index, node := range []*tensor.Tensor{
		weights.FeedForwardRouter, weights.FeedForwardLatentDown, weights.FeedForwardLatentUp,
		weights.FeedForwardUpExperts, weights.FeedForwardDownExperts,
		weights.FeedForwardSharedUp, weights.FeedForwardSharedDown,
	} {
		feeds[node] = patternedValue(node.Shape, weightSeedBase+index, weightScale, 0)
	}
	checkCUDAGraph(t, feeds,
		graphOutputCheck{output: result.Output, tolerance: accuracyRecurrent},
		graphOutputCheck{output: result.Key, tolerance: accuracyState},
		graphOutputCheck{output: result.Value, tolerance: accuracyState},
	)
}

func TestExecutorNemotronHAttentionBlockMatchesReference(t *testing.T) {
	const (
		inputSeed      = 3
		normSeed       = 5
		weightSeedBase = 11
		inputScale     = 0.15
		normScale      = 0.03
		weightScale    = 0.06
	)
	builder := tensor.NewBuilder()
	spec := modeltest.NemotronHAttention().Spec
	positions := fixturePositions(nemotronFixtureTokens)
	shapes := spec.TensorShapes(0)
	input := builder.Input("input", dtype.F32, tensor.MustShape(shapes.Embedding, nemotronFixtureTokens))
	weights := model.LayerGraphWeights{
		AttentionNorm: builder.Input("norm", dtype.F32, tensor.MustShape(shapes.Embedding)),
		AttentionQ: builder.Input("q", dtype.F32, tensor.MustShape(
			shapes.Embedding, shapes.QueryProjectionWidth(),
		)),
		AttentionK: builder.Input("k", dtype.F32, tensor.MustShape(
			shapes.Embedding, shapes.KeyProjectionWidth(),
		)),
		AttentionV: builder.Input("v", dtype.F32, tensor.MustShape(
			shapes.Embedding, shapes.ValueProjectionWidth(),
		)),
		AttentionOutput: builder.Input("output", dtype.F32, tensor.MustShape(
			shapes.AttentionOutputWidth(), shapes.Embedding,
		)),
		AttentionOutputBias: builder.Input("output_bias", dtype.F32, tensor.MustShape(shapes.Embedding)),
	}
	plan := spec.PlanLayer(0, false)
	result, err := buildCompiledLayer(model.BlockDispatchOptions{
		Spec: spec, Weights: weights, Plan: &plan,
		Context: model.CachedBlockContext{Builder: builder, Input: input, Positions: positions},
	})
	if err != nil {
		t.Fatal(err)
	}
	feeds := map[*tensor.Tensor]reference.Value{
		input: patternedValue(input.Shape, inputSeed, inputScale, 0),
		weights.AttentionNorm: patternedValue(
			weights.AttentionNorm.Shape, normSeed, normScale, 1,
		),
	}
	for index, node := range []*tensor.Tensor{
		weights.AttentionQ, weights.AttentionK, weights.AttentionV,
		weights.AttentionOutput, weights.AttentionOutputBias,
	} {
		feeds[node] = patternedValue(node.Shape, weightSeedBase+index, weightScale, 0)
	}
	checkCUDAGraph(t, feeds,
		graphOutputCheck{output: result.Output, tolerance: accuracyAttention},
		graphOutputCheck{output: result.Key, tolerance: accuracyState},
		graphOutputCheck{output: result.Value, tolerance: accuracyState},
	)
}
