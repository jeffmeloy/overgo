//go:build windows

package diffusionimage

import (
	"context"
	"errors"

	"overgo/internal/cuda/device"
	"overgo/internal/cuda/driver"
	"overgo/internal/devicemath"
)

type finalProjectionVJP struct {
	program                       *devicemath.ResidentOpsSession
	input                         driver.DevicePtr
	weight                        driver.DevicePtr
	gradient                      driver.DevicePtr
	inputChannels, outputChannels int
	height, width, patch          int
}

func compileFinalProjectionVJP(
	ctx context.Context,
	worker *device.Worker,
	checkpointWeight []float32,
	inputChannels, outputChannels, height, width, patch int,
) (*finalProjectionVJP, error) {
	if worker == nil || inputChannels <= 0 || outputChannels <= 0 || height <= 0 || width <= 0 || patch <= 0 {
		return nil, errors.New("diffusionimage: invalid final projection resident VJP")
	}
	resident, err := devicemath.NewResidentOpsSession(worker)
	if err != nil {
		return nil, err
	}
	program := &finalProjectionVJP{
		program:       resident,
		inputChannels: inputChannels, outputChannels: outputChannels,
		height: height, width: width, patch: patch,
	}
	program.input, err = resident.Allocate(ctx, uint64(inputChannels*height*width*4))
	if err == nil {
		program.gradient, err = resident.Allocate(ctx, uint64(outputChannels*height*width*patch*patch*4))
	}
	if err == nil {
		weight := finalProjectionLinearWeight(checkpointWeight, inputChannels, outputChannels, patch)
		program.weight, err = resident.Upload(ctx, driver.Bytes(weight))
	}
	if err != nil {
		return nil, errors.Join(err, program.Close())
	}
	return program, nil
}

func (program *finalProjectionVJP) Execute(ctx context.Context, input, outputGradient []float32) ([]float32, []float32, error) {
	if program == nil || program.program == nil ||
		len(input) != program.inputChannels*program.height*program.width ||
		len(outputGradient) != program.outputChannels*program.height*program.width*program.patch*program.patch {
		return nil, nil, errors.New("diffusionimage: final projection VJP input mismatch")
	}
	graphInput := nchwToGraphImage(input, program.inputChannels, program.height, program.width)
	linearGradient := finalProjectionOutputGradient(outputGradient, program.outputChannels, program.height, program.width, program.patch)
	if err := program.program.Do(ctx, func(state *device.State) error {
		return errors.Join(
			state.Driver.MemcpyHtoD(program.input, driver.Bytes(graphInput)),
			state.Driver.MemcpyHtoD(program.gradient, driver.Bytes(linearGradient)),
		)
	}); err != nil {
		return nil, nil, err
	}
	expanded := program.outputChannels * program.patch * program.patch
	inputGradient := make([]float32, len(graphInput))
	weightGradient := make([]float32, program.inputChannels*expanded)
	err := program.program.Run(func(ops *devicemath.ResidentOps) error {
		dX, allocErr := ops.AllocF32(len(inputGradient))
		if allocErr != nil {
			return allocErr
		}
		dWeight, allocErr := ops.AllocF32(len(weightGradient))
		if allocErr != nil {
			return allocErr
		}
		if err := ops.LinearBackwardT(
			program.input, program.weight, program.gradient, dX, dWeight,
			program.height*program.width, program.inputChannels, expanded,
		); err != nil {
			return err
		}
		return errors.Join(ops.DownloadF32(dX, inputGradient), ops.DownloadF32(dWeight, weightGradient))
	})
	if err != nil {
		return nil, nil, err
	}
	return graphImageToNCHW(inputGradient, program.inputChannels, program.height, program.width),
		finalProjectionCheckpointGradient(weightGradient, program.inputChannels, program.outputChannels, program.patch), nil
}

func (program *finalProjectionVJP) Close() error {
	if program == nil {
		return nil
	}
	err := program.program.Close()
	program.program = nil
	return err
}

func finalProjectionOutputGradient(output []float32, channels, height, width, patch int) []float32 {
	outputWidth := width * patch
	graphOutput := nchwToGraphImage(output, channels, height*patch, outputWidth)
	expanded := channels * patch * patch
	gradient := make([]float32, height*width*expanded)
	for y := range height {
		for x := range width {
			row := (y*width + x) * expanded
			for subY := range patch {
				for subX := range patch {
					subpixel := subY*patch + subX
					source := ((y*patch+subY)*outputWidth + x*patch + subX) * channels
					copy(gradient[row+subpixel*channels:row+(subpixel+1)*channels], graphOutput[source:source+channels])
				}
			}
		}
	}
	return gradient
}
