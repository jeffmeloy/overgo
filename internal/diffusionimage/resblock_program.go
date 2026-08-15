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
	builder := tensor.NewBuilder()
	static := make(map[*tensor.Tensor]reference.Value, 8)
	var bindErr error
	bind := func(name string, data []float32, dimensions ...uint64) *tensor.Tensor {
		node := builder.Input(block.name+name, dtype.F32, tensor.MustShape(dimensions...))
		value, err := reference.NewValue(node.Shape, data)
		if err != nil {
			bindErr = errors.Join(bindErr, err)
		} else {
			static[node] = value
		}
		return node
	}
	c, h, w := uint64(channels), uint64(height), uint64(width)
	input := builder.Input(block.name+".input", dtype.F32, tensor.MustShape(c, w, h))
	norm := func(value *tensor.Tensor, weight, bias []float32, suffix string) *tensor.Tensor {
		flat := builder.Reshape(value, c, h*w)
		weightNode := bind(suffix+".weight", weight, c)
		biasNode := bind(suffix+".bias", bias, c)
		return builder.Reshape(
			builder.GroupNorm(flat, weightNode, biasNode, uint32(config.Groups), float32(config.NormEps)),
			c, w, h,
		)
	}
	conv := func(value *tensor.Tensor, weight, bias []float32, suffix string) *tensor.Tensor {
		weightNode := bind(suffix+".weight", weight, 3, 3, c, c)
		biasNode := bind(suffix+".bias", bias, c)
		return builder.Conv2D(value, weightNode, biasNode, 1, 1, 1, 1, 1, 1, false)
	}
	hidden := conv(builder.SiLU(norm(input, block.norm1Weight, block.norm1Bias, ".norm1")), block.conv1Weight, block.conv1Bias, ".conv1")
	branch := conv(builder.SiLU(norm(hidden, block.norm2Weight, block.norm2Bias, ".norm2")), block.conv2Weight, block.conv2Bias, ".conv2")
	output := builder.Add(input, builder.Scale(branch, block.residualScale))
	if bindErr != nil {
		return resBlockProgram{}, fmt.Errorf("diffusionimage: bind residual block: %w", bindErr)
	}
	if err := builder.Err(); err != nil {
		return resBlockProgram{}, fmt.Errorf("diffusionimage: compile residual block: %w", err)
	}
	return resBlockProgram{
		input: input, output: output, static: static,
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
