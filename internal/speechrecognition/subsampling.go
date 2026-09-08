package speechrecognition

import (
	"context"
	"errors"
	"fmt"

	"overgo/internal/binaryschema"
	"overgo/internal/checked"
	"overgo/internal/tensor"
	"overgo/internal/tensor/dtype"
	"overgo/internal/tensor/reference"
)

// SpatialConvolutionBinding declares a spatial convolution in frequency/time
// order. Tensor storage is output/input/time/frequency; padding is left, right,
// top, bottom. ReLU follows the convolution only when explicitly selected.
type SpatialConvolutionBinding struct {
	Affine    AffineBinding `json:"affine"`
	Stride    [2]uint32     `json:"stride"`
	Padding   [4]uint32     `json:"padding"`
	Depthwise bool          `json:"depthwise"`
	ReLU      bool          `json:"relu"`
}

type spatialConvolution struct {
	binding      SpatialConvolutionBinding
	weight, bias reference.Value
}

type subsamplingState struct {
	tails   [][]float32
	started bool
}

func (c spatialConvolution) streamView(shape tensor.Shape, started bool) (spatialConvolution, tensor.Shape) {
	c.binding.Padding[3] = 0
	if started {
		shape.Dims[2] += uint64(int(c.weight.Shape.Dims[1]) - int(c.binding.Stride[1]))
		c.binding.Padding[2] = 0
	}
	return c, shape
}

func (l *loader) spatialConvolution(ctx context.Context, binding SpatialConvolutionBinding, channels int) (spatialConvolution, error) {
	shape, err := l.shape(ctx, binding.Affine.Weight, 4)
	if err != nil {
		return spatialConvolution{}, err
	}
	if binding.Stride[0] == 0 || binding.Stride[1] == 0 ||
		binding.Depthwise && (shape[1] != 1 || shape[0] != channels) || !binding.Depthwise && shape[1] != channels {
		return spatialConvolution{}, errors.New("spatial convolution: incompatible stride or channels")
	}
	weights, err := l.tensor(ctx, binding.Affine.Weight, shape...)
	if err != nil {
		return spatialConvolution{}, err
	}
	wshape, err := tensor.NewShape(uint64(shape[3]), uint64(shape[2]), uint64(shape[1]), uint64(shape[0]))
	if err != nil {
		return spatialConvolution{}, err
	}
	result := spatialConvolution{binding: binding, weight: reference.Value{Shape: wshape, Data: weights}}
	if binding.Affine.Bias != "" {
		values, err := l.tensor(ctx, binding.Affine.Bias, shape[0])
		if err != nil {
			return spatialConvolution{}, err
		}
		result.bias = reference.Value{Shape: tensor.MustShape(uint64(shape[0])), Data: values}
	}
	return result, nil
}

func (c spatialConvolution) node(shape tensor.Shape) (*tensor.Tensor, error) {
	b := tensor.NewBuilder()
	input := b.Input("input", dtype.F32, shape)
	weight := b.Input("weight", dtype.F32, c.weight.Shape)
	var bias *tensor.Tensor
	if c.bias.Data != nil {
		bias = b.Input("bias", dtype.F32, c.bias.Shape)
	}
	p, s := c.binding.Padding, c.binding.Stride
	output := b.Conv2D(input, weight, bias, s[0], s[1], p[0], p[1], p[2], p[3], c.binding.Depthwise)
	return output, b.Err()
}

// subsample reuses the tensor owner's CPU convolution, including its layout
// and accumulation policy. It does not introduce a second convolution kernel.
// The caller reserves the returned numeric peak alongside all retained state.
func subsample(ctx context.Context, layers []spatialConvolution, features []float32, frames, bands int, available uint64, stream *subsamplingState) ([]float32, int, uint64, error) {
	shape, err := tensor.NewShape(1, uint64(bands), uint64(frames))
	if err != nil {
		return nil, 0, 0, err
	}
	value, err := reference.BorrowedValue(shape, features)
	if err != nil {
		return nil, 0, 0, err
	}
	var peak uint64
	// Plan every intermediate extent before executing or allocating output.
	for _, layer := range layers {
		inputShape := shape
		if stream != nil {
			layer, inputShape = layer.streamView(shape, stream.started)
		}
		node, err := layer.node(inputShape)
		if err != nil {
			return nil, 0, 0, err
		}
		before, err := shape.Elements()
		if err != nil {
			return nil, 0, 0, err
		}
		after, err := node.Shape.Elements()
		if err != nil {
			return nil, 0, 0, err
		}
		count, ok := checked.Add64(before, after)
		if !ok {
			return nil, 0, 0, errors.New("spatial subsampling extent overflows")
		}
		if stream != nil {
			joined, err := inputShape.Elements()
			if err != nil {
				return nil, 0, 0, err
			}
			count, ok = checked.Add64(count, joined, before)
		}
		if !ok || count > available/binaryschema.Uint32Bytes {
			return nil, 0, 0, errors.New("spatial subsampling exceeds numeric byte budget")
		}
		peak = max(peak, count*binaryschema.Uint32Bytes)
		shape = node.Shape
	}
	count, err := shape.Elements()
	if err != nil || count > available/(2*binaryschema.Uint32Bytes) {
		return nil, 0, 0, errors.New("spatial subsampling layout conversion exceeds byte budget")
	}
	peak = max(peak, count*2*binaryschema.Uint32Bytes)
	for index, layer := range layers {
		if err := ctx.Err(); err != nil {
			return nil, 0, 0, err
		}
		if stream != nil {
			left := int(layer.weight.Shape.Dims[1]) - int(layer.binding.Stride[1])
			plane := int(value.Shape.Dims[0] * value.Shape.Dims[1])
			tail := make([]float32, left*plane)
			old := stream.tails[index]
			shortfall := max(0, len(tail)-len(value.Data))
			if shortfall != 0 && stream.started {
				copy(tail[:shortfall], old[len(old)-shortfall:])
			}
			copy(tail[shortfall:], value.Data[max(0, len(value.Data)-len(tail)):])
			var shape tensor.Shape
			layer, shape = layer.streamView(value.Shape, stream.started)
			if stream.started && len(old) != 0 {
				joined := make([]float32, len(old)+len(value.Data))
				copy(joined, old)
				copy(joined[len(old):], value.Data)
				value = reference.Value{Shape: shape, Data: joined}
			}
			stream.tails[index] = tail
		}
		node, err := layer.node(value.Shape)
		if err != nil {
			return nil, 0, 0, err
		}
		inputs := []reference.Value{value, layer.weight}
		if layer.bias.Data != nil {
			inputs = append(inputs, layer.bias)
		}
		value, err = reference.ExecuteOperation(node, inputs)
		if err != nil {
			return nil, 0, 0, fmt.Errorf("spatial subsampling stage %d: %w", index, err)
		}
		for i, x := range value.Data {
			if !checked.Finite32(x) {
				return nil, 0, 0, errors.New("spatial subsampling: non-finite output")
			}
			if layer.binding.ReLU {
				value.Data[i] = max(0, x)
			}
		}
	}
	if stream != nil {
		stream.started = true
	}
	channels, frequency, rows := int(shape.Dims[0]), int(shape.Dims[1]), int(shape.Dims[2])
	output := make([]float32, len(value.Data))
	for row := range rows {
		for channel := range channels {
			for band := range frequency {
				output[(row*channels+channel)*frequency+band] = value.Data[(row*frequency+band)*channels+channel]
			}
		}
	}
	return output, rows, peak, ctx.Err()
}
