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
	builder := tensor.NewBuilder()
	static := make(map[*tensor.Tensor]reference.Value, 4)
	var bindErr error
	bind := func(name string, data []float32, dimensions ...uint64) *tensor.Tensor {
		node := builder.Input(binding.name+name, dtype.F32, tensor.MustShape(dimensions...))
		value, err := reference.NewValue(node.Shape, data)
		if err != nil {
			bindErr = errors.Join(bindErr, err)
		} else {
			static[node] = value
		}
		return node
	}
	in := inputGeometry
	input := builder.Input(binding.name+".input", dtype.F32, tensor.MustShape(uint64(in.channels), uint64(in.width), uint64(in.height)))
	weight := bind(".weight", binding.weight, uint64(kernel), uint64(kernel), uint64(in.channels), uint64(outputChannels))
	bias := bind(".bias", binding.bias, uint64(outputChannels))
	output := builder.Conv2D(input, weight, bias, uint32(stride), uint32(stride), 0, 0, 0, 0, false)
	outHeight, outWidth := (in.height-kernel)/stride+1, (in.width-kernel)/stride+1
	if downsample {
		poolWeight := make([]float32, 4*outputChannels)
		for index := range poolWeight {
			poolWeight[index] = 0.25
		}
		pool := bind(".average_pool", poolWeight, 2, 2, 1, uint64(outputChannels))
		output = builder.Conv2D(output, pool, nil, 2, 2, 0, 0, 0, 0, true)
		outHeight, outWidth = outHeight/2, outWidth/2
	}
	if bindErr != nil {
		return projectionProgram{}, fmt.Errorf("diffusionimage: bind projection: %w", bindErr)
	}
	if err := builder.Err(); err != nil {
		return projectionProgram{}, fmt.Errorf("diffusionimage: compile projection: %w", err)
	}
	return projectionProgram{
		input: input, output: output, static: static,
		inputGeometry:  in,
		outputGeometry: imageGeometry{channels: outputChannels, height: outHeight, width: outWidth},
	}, nil
}

func (program projectionProgram) execute(ctx context.Context, device *executor.Executor, input []float32) ([]float32, error) {
	return executeImageGraph(
		ctx, device, program.input, program.output, program.static, input,
		program.inputGeometry, program.outputGeometry,
	)
}
