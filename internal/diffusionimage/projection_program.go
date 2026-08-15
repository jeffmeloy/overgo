package diffusionimage

import (
	"context"
	"errors"
	"fmt"

	"overgo/internal/cuda/executor"
	"overgo/internal/tensor"
	"overgo/internal/tensor/dtype"
	"overgo/internal/tensor/reference"
)

type projectionProgram struct {
	input, output  *tensor.Tensor
	static         map[*tensor.Tensor]reference.Value
	inputGeometry  imageGeometry
	outputGeometry imageGeometry
}

type projectionGraph struct {
	builder *tensor.Builder
	binding convBinding
	input   *tensor.Tensor
	static  map[*tensor.Tensor]reference.Value
	bindErr error
}

func newProjectionGraph(binding convBinding, geometry imageGeometry) *projectionGraph {
	builder := tensor.NewBuilder()
	return &projectionGraph{
		builder: builder, binding: binding, static: make(map[*tensor.Tensor]reference.Value, 6),
		input: builder.Input(binding.name+".input", dtype.F32, tensor.MustShape(
			uint64(geometry.channels), uint64(geometry.width), uint64(geometry.height),
		)),
	}
}

func (graph *projectionGraph) bind(name string, data []float32, dimensions ...uint64) *tensor.Tensor {
	node := graph.builder.Input(graph.binding.name+name, dtype.F32, tensor.MustShape(dimensions...))
	value, err := reference.NewValue(node.Shape, data)
	if err != nil {
		graph.bindErr = errors.Join(graph.bindErr, err)
	} else {
		graph.static[node] = value
	}
	return node
}

func (graph *projectionGraph) finish(output *tensor.Tensor, inputGeometry, outputGeometry imageGeometry) (projectionProgram, error) {
	if graph.bindErr != nil {
		return projectionProgram{}, fmt.Errorf("diffusionimage: bind projection: %w", graph.bindErr)
	}
	if err := graph.builder.Err(); err != nil {
		return projectionProgram{}, fmt.Errorf("diffusionimage: compile projection: %w", err)
	}
	return projectionProgram{
		input: graph.input, output: output, static: graph.static,
		inputGeometry: inputGeometry, outputGeometry: outputGeometry,
	}, nil
}

func compileProjectionProgram(
	binding convBinding,
	inputGeometry imageGeometry,
	outputChannels, kernel, stride int,
	downsample bool,
) (projectionProgram, error) {
	if inputGeometry.channels <= 0 || inputGeometry.height <= 0 || inputGeometry.width <= 0 ||
		outputChannels <= 0 || kernel <= 0 || stride <= 0 {
		return projectionProgram{}, errors.New("diffusionimage: invalid projection program")
	}
	in := inputGeometry
	graph := newProjectionGraph(binding, in)
	weight := graph.bind(".weight", binding.weight, uint64(kernel), uint64(kernel), uint64(in.channels), uint64(outputChannels))
	bias := graph.bind(".bias", binding.bias, uint64(outputChannels))
	output := graph.builder.Conv2D(graph.input, weight, bias, uint32(stride), uint32(stride), 0, 0, 0, 0, false)
	outHeight, outWidth := (in.height-kernel)/stride+1, (in.width-kernel)/stride+1
	if downsample {
		poolWeight := make([]float32, 4*outputChannels)
		for index := range poolWeight {
			poolWeight[index] = 0.25
		}
		pool := graph.bind(".average_pool", poolWeight, 2, 2, 1, uint64(outputChannels))
		output = graph.builder.Conv2D(output, pool, nil, 2, 2, 0, 0, 0, 0, true)
		outHeight, outWidth = outHeight/2, outWidth/2
	}
	return graph.finish(output, in, imageGeometry{channels: outputChannels, height: outHeight, width: outWidth})
}

func compileDecoderTransitionProgram(binding convBinding, inputGeometry imageGeometry, outputChannels int) (projectionProgram, error) {
	if inputGeometry.channels <= 0 || inputGeometry.height <= 0 || inputGeometry.width <= 0 || outputChannels <= 0 {
		return projectionProgram{}, errors.New("diffusionimage: invalid decoder transition program")
	}
	graph := newProjectionGraph(binding, inputGeometry)
	weight := graph.bind(".weight", binding.weight, 1, 1, uint64(inputGeometry.channels), uint64(outputChannels))
	bias := graph.bind(".bias", binding.bias, uint64(outputChannels))
	projected := graph.builder.Conv2D(graph.input, weight, bias, 1, 1, 0, 0, 0, 0, false)
	expanded := graph.builder.Concat(projected, projected, 0)
	expanded = graph.builder.Concat(expanded, expanded, 0)
	output := graph.builder.PixelShuffle2D(expanded, 2)
	return graph.finish(output, inputGeometry, imageGeometry{
		channels: outputChannels, height: 2 * inputGeometry.height, width: 2 * inputGeometry.width,
	})
}

func compileFinalProjectionProgram(binding convBinding, inputGeometry imageGeometry, outputChannels, patch int) (projectionProgram, error) {
	if inputGeometry.channels <= 0 || inputGeometry.height <= 0 || inputGeometry.width <= 0 || outputChannels <= 0 || patch <= 0 {
		return projectionProgram{}, errors.New("diffusionimage: invalid final projection program")
	}
	graph := newProjectionGraph(binding, inputGeometry)
	expandedChannels := outputChannels * patch * patch
	weight := make([]float32, inputGeometry.channels*expandedChannels)
	for y := range patch {
		for x := range patch {
			for outputChannel := range outputChannels {
				row := (y*patch+x)*outputChannels + outputChannel
				for inputChannel := range inputGeometry.channels {
					source := ((inputChannel*outputChannels+outputChannel)*patch+y)*patch + x
					weight[row*inputGeometry.channels+inputChannel] = binding.weight[source]
				}
			}
		}
	}
	flat := graph.builder.Reshape(graph.input, uint64(inputGeometry.channels), uint64(inputGeometry.height*inputGeometry.width))
	projected := graph.builder.MulMat(
		graph.bind(".pixel_projection", weight, uint64(inputGeometry.channels), uint64(expandedChannels)), flat,
	)
	projected = graph.builder.Reshape(projected, uint64(expandedChannels), uint64(inputGeometry.width), uint64(inputGeometry.height))
	output := graph.builder.PixelShuffle2D(projected, uint32(patch))
	output = graph.builder.Add(output, graph.bind(".bias", binding.bias, uint64(outputChannels)))
	return graph.finish(output, inputGeometry, imageGeometry{
		channels: outputChannels, height: patch * inputGeometry.height, width: patch * inputGeometry.width,
	})
}

func (program projectionProgram) execute(ctx context.Context, device *executor.Executor, input []float32) ([]float32, error) {
	return executeImageGraph(
		ctx, device, program.input, program.output, program.static, input,
		program.inputGeometry, program.outputGeometry,
	)
}
