package model

import "overgo/internal/tensor"

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
	return BuildArchitectureBlockCached(BlockDispatchOptions{
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
	return BuildArchitectureBlockCached(BlockDispatchOptions{
		Context: CachedBlockContext{
			Builder: builder, Input: input, MultiPositions: &positions,
			PastKey: pastKey, PastValue: pastValue, Layer: layer,
		},
		Spec: spec, Weights: weights, Plan: &plan,
	})
}
