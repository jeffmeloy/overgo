package diffusionimage

import (
	"context"
	"errors"

	"overgo/internal/cuda/executor"
	"overgo/internal/graphruntime"
	"overgo/internal/tensor"
	"overgo/internal/tensor/reference"
)

type imageGeometry struct {
	channels, height, width int
}

func executeImageGraph(
	ctx context.Context,
	device *executor.Executor,
	inputNode, outputNode *tensor.Tensor,
	static map[*tensor.Tensor]reference.Value,
	input []float32,
	inputGeometry, outputGeometry imageGeometry,
) ([]float32, error) {
	want := inputGeometry.channels * inputGeometry.height * inputGeometry.width
	if inputNode == nil || outputNode == nil || len(input) != want {
		return nil, errors.New("diffusionimage: graph image input mismatch")
	}
	graphInput := nchwToGraphImage(input, inputGeometry.channels, inputGeometry.height, inputGeometry.width)
	value, err := reference.NewValue(inputNode.Shape, graphInput)
	if err != nil {
		return nil, err
	}
	feeds := graphruntime.NewFeeds()
	feeds.AddHost(static)
	feeds.Host[inputNode] = value
	result, err := feeds.Execute(ctx, []*tensor.Tensor{outputNode}, device)
	if err != nil {
		return nil, err
	}
	return graphImageToNCHW(
		result[outputNode].Data,
		outputGeometry.channels,
		outputGeometry.height,
		outputGeometry.width,
	), nil
}
