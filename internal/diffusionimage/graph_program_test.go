package diffusionimage

import (
	"context"
	"errors"

	"overgo/internal/cuda/executor"
	"overgo/internal/graphruntime"
	"overgo/internal/tensor"
	"overgo/internal/tensor/dtype"
	"overgo/internal/tensor/reference"
)

type graphProgram struct {
	input, output        *tensor.Tensor
	static               map[*tensor.Tensor]reference.Value
	inputImage, outImage *imageGeometry
}

type projectionProgram = graphProgram

func finishTestProgram(graph *modelGraph, input, output *tensor.Tensor, scope string) (graphProgram, error) {
	if err := graph.err(scope); err != nil {
		return graphProgram{}, err
	}
	static := make(map[*tensor.Tensor]reference.Value, len(graph.static))
	for _, binding := range graph.static {
		static[binding.Node] = binding.Value
	}
	return graphProgram{input: input, output: output, static: static}, nil
}

func compileResBlockProgram(block *resBlock, config Config, channels, height, width int) (graphProgram, error) {
	if block == nil || channels <= 0 || height <= 0 || width <= 0 || channels%config.Groups != 0 {
		return graphProgram{}, errors.New("diffusionimage: invalid residual block program")
	}
	graph := newModelGraph()
	input := graph.builder.Input(block.name+".input", dtype.F32, tensor.MustShape(uint64(channels), uint64(width), uint64(height)))
	program, err := finishTestProgram(graph, input, graph.resBlock(block, config, input, channels, height, width), "residual block")
	geometry := imageGeometry{channels: channels, height: height, width: width}
	program.inputImage, program.outImage = &geometry, &geometry
	return program, err
}

func compileTransformerBlockProgram(block *attnBlock, config Config, channels, tokens int) (graphProgram, error) {
	if block == nil || channels <= 0 || tokens <= 0 || config.Heads <= 0 || channels%config.Heads != 0 ||
		config.QRank <= 0 || config.KVRank <= 0 || config.RopeTheta <= 0 || config.NormEps <= 0 {
		return graphProgram{}, errors.New("diffusionimage: invalid transformer block program")
	}
	graph := newModelGraph()
	input := graph.builder.Input(block.name+".input", dtype.F32, tensor.MustShape(uint64(channels), uint64(tokens)))
	return finishTestProgram(graph, input, graph.transformerTokens(block, config, input, channels, tokens), "transformer block")
}

func compileProjectionProgram(
	binding convBinding,
	inputGeometry imageGeometry,
	outputChannels, kernel, stride int,
	downsample bool,
) (graphProgram, error) {
	if inputGeometry.channels <= 0 || inputGeometry.height <= 0 || inputGeometry.width <= 0 ||
		outputChannels <= 0 || kernel <= 0 || stride <= 0 {
		return graphProgram{}, errors.New("diffusionimage: invalid projection program")
	}
	graph := newModelGraph()
	input := graph.builder.Input(binding.name+".input", dtype.F32, tensor.MustShape(
		uint64(inputGeometry.channels), uint64(inputGeometry.width), uint64(inputGeometry.height),
	))
	outHeight, outWidth := (inputGeometry.height-kernel)/stride+1, (inputGeometry.width-kernel)/stride+1
	output := graph.projection(input, binding, inputGeometry.channels, outputChannels, kernel, stride)
	if downsample {
		output = graph.downsample(input, binding, inputGeometry.channels, outputChannels)
		outHeight, outWidth = inputGeometry.height/2, inputGeometry.width/2
	}
	program, err := finishTestProgram(graph, input, output, "projection")
	outputGeometry := imageGeometry{channels: outputChannels, height: outHeight, width: outWidth}
	program.inputImage, program.outImage = &inputGeometry, &outputGeometry
	return program, err
}

func compileDecoderTransitionProgram(binding convBinding, inputGeometry imageGeometry, outputChannels int) (graphProgram, error) {
	if inputGeometry.channels <= 0 || inputGeometry.height <= 0 || inputGeometry.width <= 0 || outputChannels <= 0 {
		return graphProgram{}, errors.New("diffusionimage: invalid decoder transition program")
	}
	graph := newModelGraph()
	input := graph.builder.Input(binding.name+".input", dtype.F32, tensor.MustShape(
		uint64(inputGeometry.channels), uint64(inputGeometry.width), uint64(inputGeometry.height),
	))
	program, err := finishTestProgram(
		graph, input, graph.upsample(input, binding, inputGeometry.channels, outputChannels), "decoder transition",
	)
	outputGeometry := imageGeometry{
		channels: outputChannels, height: 2 * inputGeometry.height, width: 2 * inputGeometry.width,
	}
	program.inputImage, program.outImage = &inputGeometry, &outputGeometry
	return program, err
}

func compileFinalProjectionProgram(binding convBinding, inputGeometry imageGeometry, outputChannels, patch int) (graphProgram, error) {
	if inputGeometry.channels <= 0 || inputGeometry.height <= 0 || inputGeometry.width <= 0 || outputChannels <= 0 || patch <= 0 {
		return graphProgram{}, errors.New("diffusionimage: invalid final projection program")
	}
	graph := newModelGraph()
	input := graph.builder.Input(binding.name+".input", dtype.F32, tensor.MustShape(
		uint64(inputGeometry.channels), uint64(inputGeometry.width), uint64(inputGeometry.height),
	))
	program, err := finishTestProgram(
		graph, input, graph.finalProjection(input, binding, inputGeometry, outputChannels, patch), "final projection",
	)
	outputGeometry := imageGeometry{
		channels: outputChannels, height: patch * inputGeometry.height, width: patch * inputGeometry.width,
	}
	program.inputImage, program.outImage = &inputGeometry, &outputGeometry
	return program, err
}

func (program graphProgram) execute(ctx context.Context, device *executor.Executor, input []float32) ([]float32, error) {
	if program.input == nil || program.output == nil {
		return nil, errors.New("diffusionimage: incomplete graph program")
	}
	graphInput := input
	if program.inputImage != nil {
		geometry := *program.inputImage
		if len(input) != geometry.channels*geometry.height*geometry.width {
			return nil, errors.New("diffusionimage: graph image input mismatch")
		}
		graphInput = nchwToGraphImage(input, geometry.channels, geometry.height, geometry.width)
	}
	value, err := reference.NewValue(program.input.Shape, graphInput)
	if err != nil {
		return nil, err
	}
	feeds := graphruntime.NewFeeds()
	feeds.AddHost(program.static)
	feeds.Host[program.input] = value
	result, err := feeds.Execute(ctx, []*tensor.Tensor{program.output}, device)
	if err != nil {
		return nil, err
	}
	output := result[program.output].Data
	if program.outImage != nil {
		geometry := *program.outImage
		output = graphImageToNCHW(output, geometry.channels, geometry.height, geometry.width)
	}
	return output, nil
}
