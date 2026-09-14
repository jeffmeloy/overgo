package modeldevice

import (
	"overgo/internal/cuda/driver"
	"overgo/internal/gguf"
	"overgo/internal/model"
	"overgo/internal/tensor"
)

// DeviceTensorBinder binds tensor metadata to a device graph input.
type DeviceTensorBinder func(
	*tensor.Builder,
	gguf.TensorInfo,
) (*tensor.Tensor, driver.DevicePtr, error)

func bindDeviceLayerGraphFields(
	bind DeviceTensorBinder,
	builder *tensor.Builder,
	info *model.LayerWeights,
	result *model.LayerGraphWeights,
	feeds *tensor.InputBindings[driver.DevicePtr],
) error {
	return model.BindLayerGraphInventory(info, result, func(_ model.LayerBindingSlot, tensorInfo *gguf.TensorInfo) (*tensor.Tensor, error) {
		node, pointer, err := bind(builder, *tensorInfo)
		if err != nil {
			return nil, err
		}
		feeds.Add(node, pointer)
		return node, nil
	})
}
