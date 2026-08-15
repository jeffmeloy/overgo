package diffusionimage

import (
	"context"
	"errors"

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

func finishProjectionProgram(
	graph *modelGraph,
	input, output *tensor.Tensor,
	inputGeometry, outputGeometry imageGeometry,
) (projectionProgram, error) {
	if err := graph.err("projection"); err != nil {
		return projectionProgram{}, err
	}
	return projectionProgram{
		input: input, output: output, static: graph.static,
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
	graph := newModelGraph()
	input := graph.builder.Input(binding.name+".input", dtype.F32, tensor.MustShape(
		uint64(inputGeometry.channels), uint64(inputGeometry.width), uint64(inputGeometry.height),
	))
	outHeight := (inputGeometry.height-kernel)/stride + 1
	outWidth := (inputGeometry.width-kernel)/stride + 1
	var output *tensor.Tensor
	if downsample {
		output = graph.downsample(input, binding, inputGeometry.channels, outputChannels)
		outHeight, outWidth = inputGeometry.height/2, inputGeometry.width/2
	} else {
		output = graph.projection(input, binding, inputGeometry.channels, outputChannels, kernel, stride)
	}
	return finishProjectionProgram(graph, input, output, inputGeometry, imageGeometry{
		channels: outputChannels, height: outHeight, width: outWidth,
	})
}

func compileDecoderTransitionProgram(binding convBinding, inputGeometry imageGeometry, outputChannels int) (projectionProgram, error) {
	if inputGeometry.channels <= 0 || inputGeometry.height <= 0 || inputGeometry.width <= 0 || outputChannels <= 0 {
		return projectionProgram{}, errors.New("diffusionimage: invalid decoder transition program")
	}
	graph := newModelGraph()
	input := graph.builder.Input(binding.name+".input", dtype.F32, tensor.MustShape(
		uint64(inputGeometry.channels), uint64(inputGeometry.width), uint64(inputGeometry.height),
	))
	output := graph.upsample(input, binding, inputGeometry.channels, outputChannels)
	return finishProjectionProgram(graph, input, output, inputGeometry, imageGeometry{
		channels: outputChannels, height: 2 * inputGeometry.height, width: 2 * inputGeometry.width,
	})
}

func compileFinalProjectionProgram(binding convBinding, inputGeometry imageGeometry, outputChannels, patch int) (projectionProgram, error) {
	if inputGeometry.channels <= 0 || inputGeometry.height <= 0 || inputGeometry.width <= 0 || outputChannels <= 0 || patch <= 0 {
		return projectionProgram{}, errors.New("diffusionimage: invalid final projection program")
	}
	graph := newModelGraph()
	input := graph.builder.Input(binding.name+".input", dtype.F32, tensor.MustShape(
		uint64(inputGeometry.channels), uint64(inputGeometry.width), uint64(inputGeometry.height),
	))
	output := graph.finalProjection(input, binding, inputGeometry, outputChannels, patch)
	return finishProjectionProgram(graph, input, output, inputGeometry, imageGeometry{
		channels: outputChannels, height: patch * inputGeometry.height, width: patch * inputGeometry.width,
	})
}

func (program projectionProgram) execute(ctx context.Context, device *executor.Executor, input []float32) ([]float32, error) {
	return executeImageGraph(
		ctx, device, program.input, program.output, program.static, input,
		program.inputGeometry, program.outputGeometry,
	)
}
