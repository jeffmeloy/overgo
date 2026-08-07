package model

import (
	"context"
	"encoding/binary"
	"errors"
	"fmt"

	"llamacpp2go/internal/cuda/device"
	"llamacpp2go/internal/cuda/driver"
	"llamacpp2go/internal/gguf"
	"llamacpp2go/internal/tensor"
	"llamacpp2go/internal/tensor/dtype"
)

// DeviceBF16Weights: host-converted resident decode catalog.
type DeviceBF16Weights struct {
	*deviceTensorStore
}

func NewDeviceBF16Weights(worker *device.Worker) (*DeviceBF16Weights, error) {
	if worker == nil {
		return nil, errors.New("BF16 device weights require a CUDA worker")
	}
	return &DeviceBF16Weights{deviceTensorStore: newDeviceTensorStore(worker, "BF16 device")}, nil
}

func (w *DeviceBF16Weights) Load(ctx context.Context, file *gguf.File, infos []gguf.TensorInfo) error {
	if file == nil {
		return errors.New("BF16 device weights: GGUF file is nil")
	}
	return w.load(ctx, infos, func(info gguf.TensorInfo) (DeviceTensor, error) {
		value, err := LoadHostTensor(ctx, file, info)
		if err != nil {
			return DeviceTensor{}, err
		}
		storage := make([]byte, len(value.Data)*2)
		for index, item := range value.Data {
			binary.LittleEndian.PutUint16(storage[index*2:], dtype.Float32ToBF16(item))
		}
		var pointer driver.DevicePtr
		if err := w.worker.Do(ctx, func(state *device.State) error {
			var allocErr error
			pointer, allocErr = state.Driver.MemAlloc(uint64(len(storage)))
			if allocErr != nil {
				return allocErr
			}
			if copyErr := state.Driver.MemcpyHtoD(pointer, storage); copyErr != nil {
				_ = state.Driver.MemFree(pointer)
				pointer = 0
				return copyErr
			}
			return nil
		}); err != nil {
			return DeviceTensor{}, fmt.Errorf("upload BF16 device tensor %q: %w", info.Name, err)
		}
		return DeviceTensor{Info: info, Shape: value.Shape, Pointer: pointer, Size: uint64(len(storage))}, nil
	})
}

func (w *DeviceBF16Weights) Input(builder *tensor.Builder, name string) (*tensor.Tensor, driver.DevicePtr, error) {
	value, ok := w.Lookup(name)
	if !ok {
		return nil, 0, fmt.Errorf("BF16 device tensor %q is not loaded", name)
	}
	node := builder.Input(name, dtype.BF16, value.Shape)
	return node, value.Pointer, builder.Err()
}

func (w *DeviceBF16Weights) Lookup(name string) (DeviceTensor, bool) {
	if w == nil {
		return DeviceTensor{}, false
	}
	return w.deviceTensorStore.Lookup(name)
}

func (w *DeviceBF16Weights) Close() error {
	if w == nil {
		return nil
	}
	return w.deviceTensorStore.Close()
}
