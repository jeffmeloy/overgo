package model

import (
	"context"
	"errors"
	"fmt"
	"unsafe"

	"overgo/internal/cuda/device"
	"overgo/internal/cuda/driver"
	"overgo/internal/gguf"
	"overgo/internal/tensor"
	"overgo/internal/tensor/dtype"
)

// DeviceF32Weights: owns host-dequantized F32 weights in one CUDA context
// correctness bridge used before native quantized CUDA matmul
type DeviceF32Weights struct {
	*deviceTensorStore
}

func NewDeviceF32Weights(worker *device.Worker) (*DeviceF32Weights, error) {
	if worker == nil {
		return nil, errors.New("F32 device weights require a CUDA worker")
	}
	return &DeviceF32Weights{
		deviceTensorStore: newDeviceTensorStore(worker, "F32 device"),
	}, nil
}

// Load dequantizes and uploads tensors one at time; operation is
// transactional for supplied batch
func (w *DeviceF32Weights) Load(
	ctx context.Context,
	file *gguf.File,
	infos []gguf.TensorInfo,
) (returnErr error) {
	if file == nil {
		return errors.New("F32 device weights: GGUF file is nil")
	}
	return w.loadConverted(ctx, file, infos, 4, func(values []float32, _ int) []byte {
		return f32Bytes(values)
	})
}

func (w *DeviceF32Weights) Input(
	builder *tensor.Builder,
	name string,
) (*tensor.Tensor, driver.DevicePtr, error) {
	value, ok := w.Lookup(name)
	if !ok {
		return nil, 0, fmt.Errorf("F32 device tensor %q is not loaded", name)
	}
	node := builder.Input(name, dtype.F32, value.Shape)
	if err := builder.Err(); err != nil {
		return nil, 0, err
	}
	return node, value.Pointer, nil
}

// BindDeviceLayerGraphInputs: shared layer-catalog binding.
func BindDeviceLayerGraphInputs(
	builder *tensor.Builder,
	info LayerWeights,
	bind DeviceTensorBinder,
) (LayerGraphWeights, map[*tensor.Tensor]driver.DevicePtr, error) {
	if builder == nil {
		return LayerGraphWeights{}, nil, errors.New("device layer graph builder is nil")
	}
	if bind == nil {
		return LayerGraphWeights{}, nil, errors.New("device layer graph binder is nil")
	}
	feeds := make(map[*tensor.Tensor]driver.DevicePtr, 11)
	result := LayerGraphWeights{}
	if err := bindDeviceLayerGraphFields(bind, builder, &info, &result, feeds); err != nil {
		return LayerGraphWeights{}, nil, err
	}
	if err := builder.Err(); err != nil {
		return LayerGraphWeights{}, nil, err
	}
	return result, feeds, nil
}

func f32Bytes(values []float32) []byte {
	if len(values) == 0 {
		return nil
	}
	return unsafe.Slice((*byte)(unsafe.Pointer(&values[0])), len(values)*4)
}
