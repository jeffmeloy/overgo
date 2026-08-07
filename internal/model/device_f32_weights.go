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

type DeviceF32Tensor = DeviceTensor

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
	return w.load(ctx, infos, func(info gguf.TensorInfo) (DeviceTensor, error) {
		value, err := LoadHostTensor(ctx, file, info)
		if err != nil {
			return DeviceTensor{}, err
		}
		if uint64(len(value.Data)) > ^uint64(0)/4 {
			return DeviceTensor{}, fmt.Errorf("F32 device tensor %q byte size overflows", info.Name)
		}
		size := uint64(len(value.Data)) * 4
		var pointer driver.DevicePtr
		if err := w.worker.Do(ctx, func(state *device.State) error {
			var allocateErr error
			pointer, allocateErr = state.Driver.MemAlloc(size)
			if allocateErr != nil {
				return allocateErr
			}
			if copyErr := state.Driver.MemcpyHtoD(pointer, f32Bytes(value.Data)); copyErr != nil {
				_ = state.Driver.MemFree(pointer)
				pointer = 0
				return copyErr
			}
			return nil
		}); err != nil {
			return DeviceTensor{}, fmt.Errorf("upload F32 device tensor %q: %w", info.Name, err)
		}
		return DeviceTensor{
			Info:    info,
			Shape:   value.Shape,
			Pointer: pointer,
			Size:    size,
		}, nil
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

func (w *DeviceF32Weights) LayerGraphInputs(
	builder *tensor.Builder,
	info LayerWeights,
) (LayerGraphWeights, map[*tensor.Tensor]driver.DevicePtr, error) {
	return BindDeviceLayerGraphInputs(builder, info, func(
		builder *tensor.Builder,
		info gguf.TensorInfo,
	) (*tensor.Tensor, driver.DevicePtr, error) {
		return w.Input(builder, info.Name)
	})
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
	if info.FeedForwardRouter != nil &&
		((info.FeedForwardUpExperts == nil && info.FeedForwardGateUpExperts == nil) ||
			info.FeedForwardDownExperts == nil) {
		return LayerGraphWeights{}, nil, errors.New("device expert layer catalog is incomplete")
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

func (w *DeviceF32Weights) Lookup(name string) (DeviceF32Tensor, bool) {
	if w == nil {
		return DeviceF32Tensor{}, false
	}
	return w.deviceTensorStore.Lookup(name)
}

func (w *DeviceF32Weights) Count() int {
	if w == nil {
		return 0
	}
	return w.deviceTensorStore.Count()
}

func (w *DeviceF32Weights) Close() error {
	if w == nil {
		return nil
	}
	return w.deviceTensorStore.Close()
}

func f32Bytes(values []float32) []byte {
	if len(values) == 0 {
		return nil
	}
	return unsafe.Slice((*byte)(unsafe.Pointer(&values[0])), len(values)*4)
}
