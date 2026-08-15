//go:build windows

package diffusionimage

import (
	"context"
	"errors"

	"overgo/internal/cuda/device"
	"overgo/internal/cuda/driver"
	"overgo/internal/devicemath"
)

type resBlockVJPOffsets struct {
	input, norm1, activated1, hidden1 int
	norm2, activated2, incoming       int
	dBranch, dActivated2, dNorm2      int
	dHidden1, dActivated1, dNorm1     int
	dNormalizedInput, dInput          int
	dConv1Weight, dConv1Bias          int
	dConv2Weight, dConv2Bias          int
	dNorm1Weight, dNorm1Bias          int
	dNorm2Weight, dNorm2Bias          int
	total                             int
}

type resBlockVJP struct {
	worker      *device.Worker
	allocations device.AllocationSet
	session     *devicemath.ResidentOpsSession
	workspace   driver.DevicePtr
	norm1Weight driver.DevicePtr
	norm2Weight driver.DevicePtr
	conv1Weight driver.DevicePtr
	conv2Weight driver.DevicePtr
	offsets     resBlockVJPOffsets
	name        string
	channels    int
	height      int
	width       int
	groups      int
	epsilon     float64
	scale       float32
}

func compileResBlockVJP(ctx context.Context, worker *device.Worker, block *resBlock, config Config, channels, height, width int) (*resBlockVJP, error) {
	if worker == nil || block == nil || channels <= 0 || height <= 0 || width <= 0 || config.Groups <= 0 || channels%config.Groups != 0 {
		return nil, errors.New("diffusionimage: invalid residual block resident VJP")
	}
	elements := channels * height * width
	kernelElements := channels * channels * conv3x3Kernel * conv3x3Kernel
	if len(block.norm1Weight) != channels || len(block.norm2Weight) != channels ||
		len(block.conv1Weight) != kernelElements || len(block.conv2Weight) != kernelElements {
		return nil, errors.New("diffusionimage: residual block VJP weight mismatch")
	}
	offsets := compileResBlockVJPOffsets(elements, kernelElements, channels)
	program := &resBlockVJP{
		worker: worker, allocations: device.NewAllocationSet(worker), offsets: offsets,
		name: block.name, channels: channels, height: height, width: width,
		groups: config.Groups, epsilon: config.NormEps, scale: block.residualScale,
	}
	var err error
	program.workspace, err = program.allocations.Allocate(ctx, uint64(offsets.total*4))
	if err == nil {
		static := make([]float32, 0, 2*channels+2*kernelElements)
		static = append(static, block.norm1Weight...)
		static = append(static, block.norm2Weight...)
		static = append(static, block.conv1Weight...)
		static = append(static, block.conv2Weight...)
		var base driver.DevicePtr
		base, err = program.allocations.Upload(ctx, driver.Bytes(static))
		if err == nil {
			program.norm1Weight = base
			program.norm2Weight = devicemath.ResidentPtr(base, channels)
			program.conv1Weight = devicemath.ResidentPtr(base, 2*channels)
			program.conv2Weight = devicemath.ResidentPtr(base, 2*channels+kernelElements)
		}
	}
	if err == nil {
		program.session, err = devicemath.NewResidentOpsSession(worker)
	}
	if err != nil {
		return nil, errors.Join(err, program.Close(context.Background()))
	}
	return program, nil
}

func compileResBlockVJPOffsets(elements, kernelElements, channels int) resBlockVJPOffsets {
	var offsets resBlockVJPOffsets
	next := func(count int) int {
		current := offsets.total
		offsets.total += count
		return current
	}
	offsets.input = next(elements)
	offsets.norm1 = next(elements)
	offsets.activated1 = next(elements)
	offsets.hidden1 = next(elements)
	offsets.norm2 = next(elements)
	offsets.activated2 = next(elements)
	offsets.incoming = next(elements)
	offsets.dBranch = next(elements)
	offsets.dActivated2 = next(elements)
	offsets.dNorm2 = next(elements)
	offsets.dHidden1 = next(elements)
	offsets.dActivated1 = next(elements)
	offsets.dNorm1 = next(elements)
	offsets.dNormalizedInput = next(elements)
	offsets.dInput = next(elements)
	offsets.dConv1Weight = next(kernelElements)
	offsets.dConv1Bias = next(channels)
	offsets.dConv2Weight = next(kernelElements)
	offsets.dConv2Bias = next(channels)
	offsets.dNorm1Weight = next(channels)
	offsets.dNorm1Bias = next(channels)
	offsets.dNorm2Weight = next(channels)
	offsets.dNorm2Bias = next(channels)
	return offsets
}

func (program *resBlockVJP) ptr(offset int) driver.DevicePtr {
	return devicemath.ResidentPtr(program.workspace, offset)
}

func (program *resBlockVJP) Execute(ctx context.Context, trace resBlockTrace, incoming []float32) ([]float32, Grads, error) {
	if program == nil || program.session == nil {
		return nil, nil, errors.New("diffusionimage: residual block VJP unavailable")
	}
	elements := program.channels * program.height * program.width
	if len(incoming) != elements ||
		len(trace.input) != elements || len(trace.norm1) != elements || len(trace.activated1) != elements ||
		len(trace.hidden1) != elements || len(trace.norm2) != elements || len(trace.activated2) != elements || len(trace.branch) != elements {
		return nil, nil, errors.New("diffusionimage: residual block VJP input mismatch")
	}
	input := make([]float32, 7*elements)
	for index, values := range [][]float32{trace.input, trace.norm1, trace.activated1, trace.hidden1, trace.norm2, trace.activated2, incoming} {
		copy(input[index*elements:(index+1)*elements], nchwToGraphImage(values, program.channels, program.height, program.width))
	}
	if err := program.worker.Do(ctx, func(state *device.State) error {
		return state.Driver.MemcpyHtoD(program.workspace, driver.Bytes(input))
	}); err != nil {
		return nil, nil, err
	}
	o := program.offsets
	kernelElements := program.channels * program.channels * conv3x3Kernel * conv3x3Kernel
	result := make([]float32, o.total-o.dBranch)
	err := program.session.Run(func(ops *devicemath.ResidentOps) error {
		if err := ops.Scale(program.ptr(o.incoming), program.ptr(o.dBranch), program.scale, elements); err != nil {
			return err
		}
		if err := ops.Conv2DBackward(
			program.ptr(o.activated2), program.conv2Weight, program.ptr(o.dBranch),
			program.ptr(o.dActivated2), program.ptr(o.dConv2Weight), program.ptr(o.dConv2Bias),
			1, program.channels, program.width, program.height, conv3x3Kernel, conv3x3Kernel,
			program.channels, 1, 1, conv3x3Radius, conv3x3Radius,
		); err != nil {
			return err
		}
		if err := ops.SiLUBackward(program.ptr(o.dActivated2), program.ptr(o.norm2), program.ptr(o.dNorm2), elements); err != nil {
			return err
		}
		if err := ops.GroupNormBackward(
			program.ptr(o.hidden1), program.norm2Weight, program.ptr(o.dNorm2),
			program.ptr(o.dHidden1), program.ptr(o.dNorm2Weight), program.ptr(o.dNorm2Bias),
			1, program.channels, program.height*program.width, program.groups, program.epsilon,
		); err != nil {
			return err
		}
		if err := ops.Conv2DBackward(
			program.ptr(o.activated1), program.conv1Weight, program.ptr(o.dHidden1),
			program.ptr(o.dActivated1), program.ptr(o.dConv1Weight), program.ptr(o.dConv1Bias),
			1, program.channels, program.width, program.height, conv3x3Kernel, conv3x3Kernel,
			program.channels, 1, 1, conv3x3Radius, conv3x3Radius,
		); err != nil {
			return err
		}
		if err := ops.SiLUBackward(program.ptr(o.dActivated1), program.ptr(o.norm1), program.ptr(o.dNorm1), elements); err != nil {
			return err
		}
		if err := ops.GroupNormBackward(
			program.ptr(o.input), program.norm1Weight, program.ptr(o.dNorm1),
			program.ptr(o.dNormalizedInput), program.ptr(o.dNorm1Weight), program.ptr(o.dNorm1Bias),
			1, program.channels, program.height*program.width, program.groups, program.epsilon,
		); err != nil {
			return err
		}
		if err := ops.Add(program.ptr(o.incoming), program.ptr(o.dNormalizedInput), program.ptr(o.dInput), elements); err != nil {
			return err
		}
		return ops.DownloadF32(program.ptr(o.dBranch), result)
	})
	if err != nil {
		return nil, nil, err
	}
	read := func(offset, count int) []float32 {
		start := offset - o.dBranch
		return append([]float32(nil), result[start:start+count]...)
	}
	grads := Grads{
		program.name + ".conv1.weight": read(o.dConv1Weight, kernelElements),
		program.name + ".conv1.bias":   read(o.dConv1Bias, program.channels),
		program.name + ".conv2.weight": read(o.dConv2Weight, kernelElements),
		program.name + ".conv2.bias":   read(o.dConv2Bias, program.channels),
		program.name + ".norm1.weight": read(o.dNorm1Weight, program.channels),
		program.name + ".norm1.bias":   read(o.dNorm1Bias, program.channels),
		program.name + ".norm2.weight": read(o.dNorm2Weight, program.channels),
		program.name + ".norm2.bias":   read(o.dNorm2Bias, program.channels),
	}
	var scaleGradient float64
	for index, value := range incoming {
		scaleGradient += float64(value) * float64(trace.branch[index])
	}
	grads[program.name+".learned_residual_scale"] = []float32{float32(scaleGradient)}
	return graphImageToNCHW(read(o.dInput, elements), program.channels, program.height, program.width), grads, nil
}

func (program *resBlockVJP) Close(ctx context.Context) error {
	if program == nil {
		return nil
	}
	var result error
	if program.session != nil {
		result = errors.Join(result, program.session.Close())
		program.session = nil
	}
	result = errors.Join(result, program.allocations.Close(ctx))
	return result
}
