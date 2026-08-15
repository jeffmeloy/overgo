package model

import (
	"context"
	"errors"
	"fmt"
	"math"

	"overgo/internal/cuda/device"
	"overgo/internal/cuda/driver"
	"overgo/internal/gguf"
	"overgo/internal/tensor"
)

const defaultWeightChunkSize = 16 << 20
const weightUploadBufferCount = 2

type weightUploadChunk struct {
	offset uint64
	data   []byte
	err    error
}

// DeviceTensor: valid only while used on DeviceWeights' worker thread
type DeviceTensor struct {
	Info    gguf.TensorInfo
	Shape   tensor.Shape
	Pointer driver.DevicePtr
	Size    uint64
}

// DeviceWeights: owns persistent raw GGUF tensors in one CUDA context
type DeviceWeights struct {
	*deviceTensorStore
}

func NewDeviceWeights(worker *device.Worker) (*DeviceWeights, error) {
	if worker == nil {
		return nil, errors.New("device weights require a CUDA worker")
	}
	return &DeviceWeights{
		deviceTensorStore: newDeviceTensorStore(worker, "device"),
	}, nil
}

// Load: streams tensors from GGUF into persistent CUDA allocations
func (w *DeviceWeights) Load(
	ctx context.Context,
	file *gguf.File,
	tensors []gguf.TensorInfo,
) (returnErr error) {
	if file == nil {
		return errors.New("device weights: GGUF file is nil")
	}
	var uploadBuffers [weightUploadBufferCount][]byte
	return w.load(ctx, tensors, func(info gguf.TensorInfo) (DeviceTensor, error) {
		if info.Size == 0 {
			return DeviceTensor{}, fmt.Errorf("device tensor %q has zero size", info.Name)
		}
		var pointer driver.DevicePtr
		if err := w.worker.Do(ctx, func(state *device.State) error {
			var allocErr error
			pointer, allocErr = state.Driver.MemAlloc(info.Size)
			return allocErr
		}); err != nil {
			return DeviceTensor{}, fmt.Errorf("allocate tensor %q: %w", info.Name, err)
		}
		if err := w.streamTensor(ctx, file, info, pointer, &uploadBuffers); err != nil {
			_ = w.worker.Do(context.Background(), func(state *device.State) error {
				return state.Driver.MemFree(pointer)
			})
			return DeviceTensor{}, err
		}
		return DeviceTensor{Info: info, Pointer: pointer}, nil
	})
}

func (w *DeviceWeights) streamTensor(
	ctx context.Context,
	file *gguf.File,
	info gguf.TensorInfo,
	pointer driver.DevicePtr,
	uploadBuffers *[weightUploadBufferCount][]byte,
) error {
	if info.Size <= defaultWeightChunkSize {
		if cap((*uploadBuffers)[0]) < int(info.Size) {
			(*uploadBuffers)[0] = make([]byte, int(info.Size))
		}
		data := (*uploadBuffers)[0][:int(info.Size)]
		if err := file.ReadTensorRange(info, 0, data); err != nil {
			return fmt.Errorf("read tensor %q at 0: %w", info.Name, err)
		}
		if err := w.worker.Do(ctx, func(state *device.State) error {
			return state.Driver.MemcpyHtoD(pointer, data)
		}); err != nil {
			return fmt.Errorf("upload tensor %q at 0: %w", info.Name, err)
		}
		return nil
	}
	chunkSize := uint64(defaultWeightChunkSize)
	pipelineCtx, cancel := context.WithCancel(ctx)
	defer cancel()
	free := make(chan []byte, weightUploadBufferCount)
	ready := make(chan weightUploadChunk, weightUploadBufferCount)
	done := make(chan struct{})
	for index := range weightUploadBufferCount {
		if cap((*uploadBuffers)[index]) < int(chunkSize) {
			(*uploadBuffers)[index] = make([]byte, int(chunkSize))
		}
		free <- (*uploadBuffers)[index][:int(chunkSize)]
	}
	go func() {
		defer close(done)
		defer close(ready)
		for offset := uint64(0); offset < info.Size; {
			var buffer []byte
			select {
			case buffer = <-free:
			case <-pipelineCtx.Done():
				return
			}
			length := min(chunkSize, info.Size-offset)
			data := buffer[:int(length)]
			if err := file.ReadTensorRange(info, offset, data); err != nil {
				select {
				case ready <- weightUploadChunk{offset: offset, err: err}:
				case <-pipelineCtx.Done():
				}
				return
			}
			select {
			case ready <- weightUploadChunk{offset: offset, data: data}:
				offset += length
			case <-pipelineCtx.Done():
				return
			}
		}
	}()
	for chunk := range ready {
		if chunk.err != nil {
			cancel()
			<-done
			return fmt.Errorf("read tensor %q at %d: %w", info.Name, chunk.offset, chunk.err)
		}
		if uint64(pointer) > math.MaxUint64-chunk.offset {
			cancel()
			<-done
			return fmt.Errorf("device pointer for tensor %q overflows", info.Name)
		}
		destination := pointer + driver.DevicePtr(chunk.offset)
		if err := w.worker.Do(ctx, func(state *device.State) error {
			return state.Driver.MemcpyHtoD(destination, chunk.data)
		}); err != nil {
			cancel()
			<-done
			return fmt.Errorf("upload tensor %q at %d: %w", info.Name, chunk.offset, err)
		}
		select {
		case free <- chunk.data[:int(chunkSize)]:
		case <-pipelineCtx.Done():
		}
	}
	<-done
	return ctx.Err()
}

// Do exposes loaded device pointers only inside owning worker callback
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

// Input: adds graph input using tensor's original GGUF storage type
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
