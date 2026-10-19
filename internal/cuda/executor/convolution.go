package executor

import (
	"errors"
	"fmt"
	"slices"

	"overgo/internal/checked"
	"overgo/internal/cuda/cudnn"
	"overgo/internal/cuda/device"
	"overgo/internal/cuda/driver"
	"overgo/internal/tensor"
)

func spatialConvolutionShape(node *tensor.Tensor) cudnn.Shape {
	input, weight := node.Inputs[0].Shape, node.Inputs[1].Shape
	attributes := node.Attrs.(tensor.Conv2DAttributes)
	return cudnn.Shape{In: int(input.Dims[0]), Out: int(node.Shape.Dims[0]), Frames: tensor.SingletonExtent, Height: int(input.Dims[tensor.PairedExtent]), Width: int(input.Dims[tensor.SingletonExtent]), Kernel: [tensor.TripleExtent]int{tensor.SingletonExtent, int(weight.Dims[tensor.SingletonExtent]), int(weight.Dims[0])}, Stride: [tensor.TripleExtent]int{tensor.SingletonExtent, int(attributes.StrideY), int(attributes.StrideX)}, Pad: [tensor.TripleExtent]int{0, int(attributes.PadTop), int(attributes.PadLeft)}}
}

func (b *blasState) convolutionStaging(state *device.State, shapes []cudnn.Shape) (uint64, error) {
	if len(shapes) == 0 {
		return 0, nil
	}
	if b.convolutionLibrary == nil {
		var err error
		b.convolutionLibrary, err = cudnn.Open(state.Stream)
		if err != nil {
			return 0, err
		}
		b.convolutionPlans = map[cudnn.Shape]cudnn.Convolution{}
	}
	for shape, plan := range b.convolutionPlans {
		if !slices.Contains(shapes, shape) {
			if err := b.convolutionLibrary.Release(plan); err != nil {
				return 0, err
			}
			delete(b.convolutionPlans, shape)
		}
	}
	var required uint64
	for _, shape := range shapes {
		plan, ok := b.convolutionPlans[shape]
		if !ok {
			var err error
			plan, err = b.convolutionLibrary.PlanSpatialWinograd(shape)
			if err != nil {
				return 0, err
			}
			b.convolutionPlans[shape] = plan
		}
		image, ok := checked.Mul64(uint64(shape.Height), uint64(shape.Width))
		if !ok {
			return 0, errors.New("convolution image size overflows")
		}
		words, ok := checked.Mul64(image, uint64(shape.In)+uint64(shape.Out))
		if !ok {
			return 0, errors.New("convolution staging size overflows")
		}
		bytes, ok := checked.Mul64(words, f32ScalarBytes)
		if !ok {
			return 0, errors.New("convolution staging byte count overflows")
		}
		bytes, err := addMemoryBytes(bytes, plan.Workspace)
		if err != nil {
			return 0, err
		}
		required = max(required, bytes)
	}
	return required, nil
}

func (b *blasState) closeConvolutions() error {
	if b.convolutionLibrary == nil {
		return nil
	}
	var err error
	for shape, plan := range b.convolutionPlans {
		err = errors.Join(err, b.convolutionLibrary.Release(plan))
		delete(b.convolutionPlans, shape)
	}
	err = errors.Join(err, b.convolutionLibrary.Close())
	b.convolutionLibrary = nil
	return err
}

func launchSpatialConvolution(state *device.State, functions functionSet, blas *blasState, node *tensor.Tensor, input, weight, bias, output driver.DevicePtr) error {
	if blas == nil || blas.convolutionLibrary == nil {
		return errors.New("winograd convolution was not prepared")
	}
	shape := spatialConvolutionShape(node)
	plan, ok := blas.convolutionPlans[shape]
	if !ok {
		return fmt.Errorf("winograd convolution shape was not prepared: %v", shape)
	}
	positions := uint32(shape.Height * shape.Width)
	channels := uint32(shape.In)
	count := channels * positions
	planarInput := blas.staging
	planarOutput := planarInput + driver.DevicePtr(uint64(count)*f32ScalarBytes)
	channelsOut := uint32(shape.Out)
	outCount := channelsOut * positions
	workspace := planarOutput + driver.DevicePtr(uint64(outCount)*f32ScalarBytes)
	if err := launch1DABI(state, functions[kernelTranspose2dF32], count, &input, &planarInput, &channels, &positions, &count); err != nil {
		return err
	}
	if !traceExternalCall(state, traceTagSpatialConvolution, uint64(planarInput), uint64(weight), uint64(planarOutput), uint64(workspace), uint64(shape.In), uint64(shape.Out), uint64(shape.Height), uint64(shape.Width)) {
		if err := blas.convolutionLibrary.Forward(plan, planarInput, weight, workspace, planarOutput); err != nil {
			return err
		}
	}
	if err := launch1DABI(state, functions[kernelTranspose2dF32], outCount, &planarOutput, &output, &positions, &channelsOut, &outCount); err != nil {
		return err
	}
	if node.Attrs.(tensor.Conv2DAttributes).HasBias {
		one := uint32(tensor.SingletonExtent)
		if err := launch1DABI(state, functions[kernelBroadcastAddF32], outCount, &output, &bias, &output, &outCount, &channelsOut, &positions, &one, &one, &channelsOut, &one, &one, &one, &channelsOut, &positions, &one); err != nil {
			return err
		}
	}
	blas.stagedNode = nil
	return nil
}
