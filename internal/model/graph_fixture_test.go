package model

import (
	"testing"

	"overgo/internal/tensor"
)

func fixtureModelPlan(t *testing.T, spec Spec, weights Weights) ModelPlan {
	t.Helper()
	plan, err := CompileModelPlan(spec, weights)
	if err != nil {
		t.Fatal(err)
	}
	return plan
}

func fixtureLayerPlan(t *testing.T, spec Spec, weights Weights, layer int) LayerPlan {
	t.Helper()
	plan, err := fixtureModelPlan(t, spec, weights).Layer(layer)
	if err != nil {
		t.Fatal(err)
	}
	return plan
}

func fixtureLayerProgram(t *testing.T, spec Spec, weights Weights, layer int) CompiledLayerProgram {
	t.Helper()
	program, err := fixtureModelPlan(t, spec, weights).LayerProgram(layer)
	if err != nil {
		t.Fatal(err)
	}
	return program
}

func fixtureDraftProgram(
	t *testing.T,
	spec Spec,
	weights Weights,
	head uint32,
) CompiledLayerProgram {
	t.Helper()
	draft, err := fixtureModelPlan(t, spec, weights).DraftProgram(head)
	if err != nil {
		t.Fatal(err)
	}
	return draft
}

type qwen35BlockOptions struct {
	Builder        *tensor.Builder
	Input          *tensor.Tensor
	Spec           Spec
	Weights        LayerGraphWeights
	Positions      []uint32
	MultiPositions *[4][]uint32
	Sequences      uint64
	Recurrent      bool
	PastKey        *tensor.Tensor
	PastValue      *tensor.Tensor
	ConvState      *tensor.Tensor
	SSMState       *tensor.Tensor
	CacheWrite     tensor.CacheWriteMode
}

type qwen35BlockFixtureResult struct {
	Output, Key, Value, ConvState, SSMState *tensor.Tensor
	Recurrent                               bool
}

func buildQwen35BlockWithOptions(options qwen35BlockOptions) (qwen35BlockFixtureResult, error) {
	plan := options.Spec.PlanLayer(0, options.Recurrent)
	pastKey, pastValue := options.PastKey, options.PastValue
	if options.Recurrent {
		pastKey, pastValue = options.ConvState, options.SSMState
	}
	result, err := executeCompiledLayer(BlockDispatchOptions{
		Spec: options.Spec, Weights: options.Weights, Plan: &plan,
		Context: CachedBlockContext{
			Builder: options.Builder, Input: options.Input, Positions: options.Positions,
			MultiPositions: options.MultiPositions, PastKey: pastKey, PastValue: pastValue,
			Recurrent: options.Recurrent, CacheWrite: options.CacheWrite,
			Sequences: options.Sequences,
		},
	})
	fixture := qwen35BlockFixtureResult{Output: result.Output, Key: result.Key, Value: result.Value}
	if options.Recurrent {
		fixture.Key, fixture.Value = nil, nil
		fixture.ConvState, fixture.SSMState, fixture.Recurrent = result.Key, result.Value, true
	}
	return fixture, err
}

func buildFixtureLayerWithPlan(
	builder *tensor.Builder,
	input *tensor.Tensor,
	spec Spec,
	weights LayerGraphWeights,
	positions []uint32,
	pastKey, pastValue *tensor.Tensor,
	plan LayerPlan,
) (DenseBlockResult, error) {
	return buildFixtureLayerWithAuxiliary(
		builder, input, spec, weights, positions, pastKey, pastValue, nil, nil, plan,
	)
}

func buildFixtureLayerWithAuxiliary(
	builder *tensor.Builder,
	input *tensor.Tensor,
	spec Spec,
	weights LayerGraphWeights,
	positions []uint32,
	pastKey, pastValue, pastIndexerKey, perLayerInput *tensor.Tensor,
	plan LayerPlan,
) (DenseBlockResult, error) {
	states := CacheStates[*tensor.Tensor](nil)
	if pastIndexerKey != nil {
		states = CacheStates[*tensor.Tensor]{
			CacheStateIndexerKey: {Mode: CacheStateToken, Value: pastIndexerKey},
		}
	}
	return executeCompiledLayer(BlockDispatchOptions{
		Context: CachedBlockContext{
			Builder: builder, Input: input, Positions: positions,
			PastKey: pastKey, PastValue: pastValue, PastStates: states,
			PerLayerInput: perLayerInput, Layer: plan.Layer, Recurrent: plan.Recurrent,
		},
		Spec: spec, Weights: weights, Plan: &plan,
	})
}

func buildFixtureDenseBlockCachedForLayer(
	builder *tensor.Builder,
	input *tensor.Tensor,
	spec Spec,
	weights LayerGraphWeights,
	positions []uint32,
	pastKey, pastValue *tensor.Tensor,
	layer uint32,
) (DenseBlockResult, error) {
	return buildFixtureLayerWithPlan(
		builder, input, spec, weights, positions, pastKey, pastValue,
		spec.PlanLayer(layer, false),
	)
}

func buildFixtureDenseBlockCached(
	builder *tensor.Builder,
	input *tensor.Tensor,
	spec Spec,
	weights LayerGraphWeights,
	positions []uint32,
	pastKey, pastValue *tensor.Tensor,
) (DenseBlockResult, error) {
	return buildFixtureDenseBlockCachedForLayer(
		builder, input, spec, weights, positions, pastKey, pastValue, 0,
	)
}

func buildFixtureDenseBlock(
	builder *tensor.Builder,
	input *tensor.Tensor,
	spec Spec,
	weights LayerGraphWeights,
	positions []uint32,
) (*tensor.Tensor, error) {
	result, err := buildFixtureDenseBlockCached(
		builder, input, spec, weights, positions, nil, nil,
	)
	return result.Output, err
}

func buildFixtureDenseBlockCachedWithMultiPositions(
	builder *tensor.Builder,
	input *tensor.Tensor,
	spec Spec,
	weights LayerGraphWeights,
	positions [4][]uint32,
	pastKey, pastValue *tensor.Tensor,
	layer uint32,
) (DenseBlockResult, error) {
	plan := spec.PlanLayer(layer, false)
	return executeCompiledLayer(BlockDispatchOptions{
		Context: CachedBlockContext{
			Builder: builder, Input: input, MultiPositions: &positions,
			PastKey: pastKey, PastValue: pastValue, Layer: layer,
		},
		Spec: spec, Weights: weights, Plan: &plan,
	})
}
