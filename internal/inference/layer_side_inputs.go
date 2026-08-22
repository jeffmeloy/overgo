package inference

import (
	"fmt"
	"math"

	"overgo/internal/checked"
	"overgo/internal/model"
	"overgo/internal/tensor"
	"overgo/internal/tensor/dtype"
	"overgo/internal/tensor/reference"
)

type layerSideInputs struct {
	embeddingSkip  *tensor.Tensor
	perLayerInput  *tensor.Tensor
	attentionBlock *tensor.Tensor
}

type boundLayerSideInputs struct {
	currentPositions *tensor.Tensor
}

func bindLayerSideInputs(
	builder *tensor.Builder,
	spec model.Spec,
	positions []uint32,
	plan model.LayerPlan,
	hostFeeds map[*tensor.Tensor]reference.Value,
	weights *model.LayerGraphWeights,
	inputs layerSideInputs,
) (boundLayerSideInputs, error) {
	if builder == nil || weights == nil {
		return boundLayerSideInputs{}, fmt.Errorf("%s layer side inputs are invalid", spec.Architecture)
	}
	if plan.EmbeddingSkip {
		weights.EmbeddingSkip = inputs.embeddingSkip
	}
	if plan.PerLayerInput {
		weights.PerLayerInput = inputs.perLayerInput
	}
	if plan.AttentionBlocks != model.AttentionBlocksNone {
		weights.AttentionBlockIDs = inputs.attentionBlock
	}
	if err := bindAttentionTemperature(builder, spec, positions, plan, hostFeeds, weights); err != nil {
		return boundLayerSideInputs{}, err
	}
	var result boundLayerSideInputs
	if plan.Cache == model.CacheCompressedAttention {
		data := make([]float32, len(positions))
		for index, position := range positions {
			if !checked.ExactFloat32Uint(position) {
				return boundLayerSideInputs{}, fmt.Errorf(
					"%s position %d exceeds exact F32 cache representation",
					spec.Architecture,
					position,
				)
			}
			data[index] = float32(position)
		}
		shape := tensor.MustShape(tensor.SingletonExtent, tensor.SingletonExtent, uint64(len(positions)))
		result.currentPositions = builder.Input(
			fmt.Sprintf("blk.%d.positions.current", plan.Layer), dtype.F32, shape,
		)
		hostFeeds[result.currentPositions] = reference.Value{Shape: shape, Data: data}
	}
	return result, builder.Err()
}

func bindAttentionTemperature(
	builder *tensor.Builder,
	spec model.Spec,
	positions []uint32,
	plan model.LayerPlan,
	hostFeeds map[*tensor.Tensor]reference.Value,
	weights *model.LayerGraphWeights,
) error {
	if plan.Temperature == model.AttentionTemperatureNone ||
		plan.Temperature == model.AttentionTemperatureConfigured && !checked.Nonzero(spec.AttentionTempScale) {
		return nil
	}
	if !checked.Nonzero(spec.AttentionTempFloor) {
		return fmt.Errorf("%s attention temperature input is invalid", spec.Architecture)
	}
	shape := tensor.MustShape(tensor.SingletonExtent, tensor.SingletonExtent, uint64(len(positions)))
	data := make([]float32, len(positions))
	for index, position := range positions {
		step := math.Floor((float64(position) + float64(spec.AttentionTempOffset)) / float64(spec.AttentionTempFloor))
		identity := float64(tensor.SingletonExtent)
		data[index] = float32(math.Log(step+identity))*spec.AttentionTempScale + float32(identity)
	}
	input := builder.Input("attention_temperature", dtype.F32, shape)
	hostFeeds[input] = reference.Value{Shape: shape, Data: data}
	weights.AttentionTemperatureScale = input
	return builder.Err()
}
