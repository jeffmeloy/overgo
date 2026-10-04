package modeldevice

import (
	"context"
	"errors"
	"fmt"

	"overgo/internal/cuda/device"
	"overgo/internal/cuda/driver"
	"overgo/internal/gguf"
	"overgo/internal/model"
	"overgo/internal/tensor"
)

// DeviceTensor is valid only while used on the DeviceWeights worker thread.
type DeviceTensor struct {
	Info    gguf.TensorInfo
	Shape   tensor.Shape
	Pointer driver.DevicePtr
	Size    uint64
}

// DeviceWeights owns persistent raw GGUF tensors in one CUDA context.
type DeviceWeights struct {
	*deviceTensorStore
}

// NewDeviceWeights opens a raw tensor store on the worker.
func NewDeviceWeights(worker *device.Worker) (*DeviceWeights, error) {
	if worker == nil {
		return nil, errors.New("device weights require a CUDA worker")
	}
	return &DeviceWeights{
		deviceTensorStore: newDeviceTensorStore(worker, "device"),
	}, nil
}

// Do exposes loaded device pointers only inside the owning worker callback.
func (w *DeviceWeights) Do(
	ctx context.Context,
	function func(*device.State, map[string]DeviceTensor) error,
) error {
	if function == nil {
		return errors.New("device weights: nil callback")
	}
	if w == nil || w.deviceTensorStore == nil {
		return errors.New("device weights are closed")
	}
	w.mu.RLock()
	defer w.mu.RUnlock()
	if w.closed {
		return errors.New("device weights are closed")
	}
	return w.worker.Do(ctx, func(state *device.State) error {
		return function(state, w.tensors)
	})
}

// Input adds a graph input using the tensor's original GGUF storage type.
func (w *DeviceWeights) Input(
	builder *tensor.Builder,
	name string,
) (*tensor.Tensor, driver.DevicePtr, error) {
	if builder == nil {
		return nil, 0, errors.New("device tensor graph builder is nil")
	}
	value, ok := w.Lookup(name)
	if !ok {
		return nil, 0, fmt.Errorf("device tensor %q is not loaded", name)
	}
	shape, err := tensor.NewShape(value.Info.Shape[:value.Info.Dimensions]...)
	if err != nil {
		return nil, 0, fmt.Errorf("device tensor %q shape: %w", name, err)
	}
	node := builder.Input(name, value.Info.Type, shape)
	if err := builder.Err(); err != nil {
		return nil, 0, err
	}
	return node, value.Pointer, nil
}

// BindDeviceLayerGraphInputs binds a compiled layer catalog.
func BindDeviceLayerGraphInputs(
	builder *tensor.Builder,
	info model.LayerWeights,
	bind DeviceTensorBinder,
) (model.LayerGraphWeights, tensor.InputBindings[driver.DevicePtr], error) {
	if builder == nil {
		return model.LayerGraphWeights{}, nil, errors.New("device layer graph builder is nil")
	}
	if bind == nil {
		return model.LayerGraphWeights{}, nil, errors.New("device layer graph binder is nil")
	}
	var feeds tensor.InputBindings[driver.DevicePtr]
	result := model.LayerGraphWeights{}
	if err := bindDeviceLayerGraphFields(bind, builder, &info, &result, &feeds); err != nil {
		return model.LayerGraphWeights{}, nil, err
	}
	if err := builder.Err(); err != nil {
		return model.LayerGraphWeights{}, nil, err
	}
	return result, feeds, nil
}
