package model

import (
	"context"
	"encoding/binary"
	"errors"
	"fmt"
	"unsafe"

	"overgo/internal/cuda/device"
	"overgo/internal/cuda/driver"
	"overgo/internal/gguf"
	"overgo/internal/tensor"
	"overgo/internal/tensor/dtype"
)

type convertedTensorEncoder func([]float32, int) []byte

// DeviceConvertedWeights owns one converted CUDA tensor catalog.
type DeviceConvertedWeights struct {
	*deviceTensorStore
	storage       dtype.Type
	bytesPerValue uint64
	encode        convertedTensorEncoder
}

func NewDeviceConvertedWeights(worker *device.Worker, storage dtype.Type) (*DeviceConvertedWeights, error) {
	if worker == nil {
		return nil, errors.New("converted device weights require a CUDA worker")
	}
	bytesPerValue, valid := storage.ScalarBytes()
	if !valid {
		return nil, fmt.Errorf("converted device weights: %s scalar storage traits are unavailable", storage)
	}
	var encode convertedTensorEncoder
	switch storage {
	case dtype.F32:
		encode = encodeF32
	case dtype.BF16:
		encode = encodeBF16
	default:
		return nil, fmt.Errorf("converted device weights: unsupported storage %s", storage)
	}
	return &DeviceConvertedWeights{
		deviceTensorStore: newDeviceTensorStore(worker, storage.String()+" device"),
		storage:           storage, bytesPerValue: bytesPerValue, encode: encode,
	}, nil
}

func (w *DeviceConvertedWeights) Load(ctx context.Context, file *gguf.File, infos []gguf.TensorInfo) error {
	if file == nil {
		return errors.New("converted device weights: GGUF file is nil")
	}
	return w.loadConverted(ctx, file, infos, w.bytesPerValue, w.encode)
}

func (w *DeviceConvertedWeights) Input(builder *tensor.Builder, name string) (*tensor.Tensor, driver.DevicePtr, error) {
	value, ok := w.Lookup(name)
	if !ok {
		return nil, 0, fmt.Errorf("converted device tensor %q is not loaded", name)
	}
	node := builder.Input(name, w.storage, value.Shape)
	return node, value.Pointer, builder.Err()
}

func encodeF32(values []float32, _ int) []byte {
	if len(values) == tensor.FirstOffset {
		return nil
	}
	byteCount := len(values) * int(unsafe.Sizeof(values[tensor.FirstOffset]))
	return unsafe.Slice((*byte)(unsafe.Pointer(&values[tensor.FirstOffset])), byteCount)
}

func encodeBF16(values []float32, size int) []byte {
	storage := make([]byte, 0, size)
	for _, value := range values {
		storage = binary.LittleEndian.AppendUint16(storage, dtype.Float32ToBF16(value))
	}
	return storage
}
