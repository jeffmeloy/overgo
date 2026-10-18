package inference

import (
	"errors"
	"fmt"
	"math"

	"overgo/internal/model"
	"overgo/internal/tensor"
	"overgo/internal/tensor/dtype"
	"overgo/internal/tensor/reference"
)

type layerSideInputs struct {
	multiPositions *MultiAxisPositions
	embeddingSkip  *tensor.Tensor
	perLayerInput  *tensor.Tensor
	attentionBlock *tensor.Tensor
}

type boundLayerSideInputs struct {
	currentPositions *tensor.Tensor
}

const maxExactFloat32Position = uint32(1 << 24)

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
	if plan.Cache == model.CacheCompressedAttention || plan.AttentionGraph.BlockSize != 0 {
		axes := tensor.SingletonExtent
		if plan.AttentionGraph.BlockSize != 0 {
			axes = tensor.MaxDimensions
		}
		data := make([]float32, len(positions)*axes)
		if err := writeCurrentPositions(data, axes, func(index, axis int) (uint32, error) {
			if plan.AttentionGraph.BlockSize == 0 || inputs.multiPositions == nil {
				return positions[index], nil
			}
			if len(inputs.multiPositions[axis]) != len(positions) {
				return 0, errors.New("cached multi-axis coordinate count differs from tokens")
			}
			return inputs.multiPositions[axis][index], nil
		}); err != nil {
			return boundLayerSideInputs{}, fmt.Errorf("%s %w", spec.Architecture, err)
		}
		shape := tensor.MustShape(uint64(axes), tensor.SingletonExtent, uint64(len(positions)))
		result.currentPositions = builder.Input(
			fmt.Sprintf("blk.%d.positions.current", plan.Layer), dtype.F32, shape,
		)
		hostFeeds[result.currentPositions] = reference.Value{Shape: shape, Data: data}
	}
	return result, builder.Err()
}

// writeCurrentPositions lays a step's coordinates out as a layer's current-position input holds them.
//   - data = tokens x axes, token-major
//   - coordinate > maxExactFloat32Position => refused, because the F32 cache would round it
func writeCurrentPositions(data []float32, axes int, coordinate func(index, axis int) (uint32, error)) error {
	for index := range len(data) / axes {
		for axis := range axes {
			value, err := coordinate(index, axis)
			if err != nil {
				return err
			}
			if value > maxExactFloat32Position {
				return fmt.Errorf("position %d exceeds exact F32 cache representation", value)
			}
			data[index*axes+axis] = float32(value)
		}
	}
	return nil
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
		plan.Temperature == model.AttentionTemperatureConfigured && spec.AttentionTempScale == 0 {
		return nil
	}
	if spec.AttentionTempFloor == 0 {
		return fmt.Errorf("%s attention temperature input is invalid", spec.Architecture)
	}
	shape := tensor.MustShape(1, 1, uint64(len(positions)))
	data := make([]float32, len(positions))
	for index, position := range positions {
		step := math.Floor((float64(position) + float64(spec.AttentionTempOffset)) / float64(spec.AttentionTempFloor))
		data[index] = float32(math.Log(step+1))*spec.AttentionTempScale + 1
	}
	input := builder.Input("attention_temperature", dtype.F32, shape)
	hostFeeds[input] = reference.Value{Shape: shape, Data: data}
	weights.AttentionTemperatureScale = input
	return builder.Err()
}
