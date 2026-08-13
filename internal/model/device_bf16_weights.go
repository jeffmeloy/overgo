package model

import (
	"context"
	"encoding/binary"
	"errors"
	"fmt"

	"overgo/internal/cuda/device"
	"overgo/internal/cuda/driver"
	"overgo/internal/gguf"
	"overgo/internal/tensor"
	"overgo/internal/tensor/dtype"
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
	return w.loadConverted(ctx, file, infos, 2, func(values []float32, size int) []byte {
		storage := make([]byte, size)
		for index, item := range values {
			binary.LittleEndian.PutUint16(storage[index*2:], dtype.Float32ToBF16(item))
		}
		return storage
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
