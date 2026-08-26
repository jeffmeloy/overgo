package diffusionimage

import (
	"context"
	"errors"

	"overgo/internal/cuda/executor"
	"overgo/internal/tensor"
	"overgo/internal/tensor/dtype"
	"overgo/internal/tensor/reference"
)

type resBlockProgram struct {
	input, output *tensor.Tensor
	static        map[*tensor.Tensor]reference.Value
	channels      int
	height, width int
}

func compileResBlockProgram(block *resBlock, config Config, channels, height, width int) (resBlockProgram, error) {
	if block == nil || channels <= 0 || height <= 0 || width <= 0 || channels%config.Groups != 0 {
		return resBlockProgram{}, errors.New("diffusionimage: invalid residual block program")
	}
	graph := newModelGraph()
	input := graph.builder.Input(block.name+".input", dtype.F32, tensor.MustShape(uint64(channels), uint64(width), uint64(height)))
	output := graph.resBlock(block, config, input, channels, height, width)
	if err := graph.err("residual block"); err != nil {
		return resBlockProgram{}, err
	}
	return resBlockProgram{
		input: input, output: output, static: graph.hostValues(),
		channels: channels, height: height, width: width,
	}, nil
}

func (program resBlockProgram) execute(ctx context.Context, device *executor.Executor, input []float32) ([]float32, error) {
	geometry := imageGeometry{channels: program.channels, height: program.height, width: program.width}
	return executeImageGraph(ctx, device, program.input, program.output, program.static, input, geometry, geometry)
}

func nchwToGraphImage(input []float32, channels, height, width int) []float32 {
	plane := height * width
	output := make([]float32, len(input))
	for position := range plane {
		for channel := range channels {
			output[position*channels+channel] = input[channel*plane+position]
		}
	}
	return output
}

func graphImageToNCHW(input []float32, channels, height, width int) []float32 {
	plane := height * width
	output := make([]float32, len(input))
	for position := range plane {
		for channel := range channels {
			output[channel*plane+position] = input[position*channels+channel]
		}
	}
	return output
}
