package model

import (
	"context"
	"errors"
	"fmt"
	"math"
	"sync"

	"llamacpp2go/internal/cuda/device"
	"llamacpp2go/internal/cuda/driver"
	"llamacpp2go/internal/gguf"
	"llamacpp2go/internal/tensor"
)

const defaultWeightChunkSize = 16 << 20

// DeviceTensor is valid only while used on DeviceWeights' worker thread.
type DeviceTensor struct {
	Info    gguf.TensorInfo
	Pointer driver.DevicePtr
}

// DeviceWeights owns persistent raw GGUF tensors in one CUDA context.
type DeviceWeights struct {
	worker *device.Worker

	mu      sync.RWMutex
	tensors map[string]DeviceTensor
	closed  bool
}

func NewDeviceWeights(worker *device.Worker) (*DeviceWeights, error) {
	if worker == nil {
		return nil, errors.New("device weights require a CUDA worker")
	}
	return &DeviceWeights{
		worker:  worker,
		tensors: make(map[string]DeviceTensor),
	}, nil
}

// Load streams tensors from GGUF into persistent CUDA allocations.
func (w *DeviceWeights) Load(
	ctx context.Context,
	file *gguf.File,
	tensors []gguf.TensorInfo,
) (returnErr error) {
	if file == nil {
		return errors.New("device weights: GGUF file is nil")
	}
	w.mu.Lock()
	defer w.mu.Unlock()
	if w.closed {
		return errors.New("device weights are closed")
	}
	var added []string
	defer func() {
		if returnErr == nil {
			return
		}
		var pointers []driver.DevicePtr
		for _, name := range added {
			pointers = append(pointers, w.tensors[name].Pointer)
			delete(w.tensors, name)
		}
		_ = w.worker.Do(context.Background(), func(state *device.State) error {
			for _, pointer := range pointers {
				_ = state.Driver.MemFree(pointer)
			}
			return nil
		})
	}()
	for _, info := range tensors {
		if _, exists := w.tensors[info.Name]; exists {
			return fmt.Errorf("device tensor %q is already loaded", info.Name)
		}
		if info.Size == 0 {
			return fmt.Errorf("device tensor %q has zero size", info.Name)
		}
		if err := ctx.Err(); err != nil {
			return err
		}
		var pointer driver.DevicePtr
		if err := w.worker.Do(ctx, func(state *device.State) error {
			var allocErr error
			pointer, allocErr = state.Driver.MemAlloc(info.Size)
			return allocErr
		}); err != nil {
			return fmt.Errorf("allocate tensor %q: %w", info.Name, err)
		}
		if err := w.streamTensor(ctx, file, info, pointer); err != nil {
			_ = w.worker.Do(context.Background(), func(state *device.State) error {
				return state.Driver.MemFree(pointer)
			})
			return err
		}
		w.tensors[info.Name] = DeviceTensor{Info: info, Pointer: pointer}
		added = append(added, info.Name)
	}
	return nil
}

func (w *DeviceWeights) streamTensor(
	ctx context.Context,
	file *gguf.File,
	info gguf.TensorInfo,
	pointer driver.DevicePtr,
) error {
	chunkSize := uint64(defaultWeightChunkSize)
	if info.Size < chunkSize {
		chunkSize = info.Size
	}
	chunk := make([]byte, int(chunkSize))
	for offset := uint64(0); offset < info.Size; {
		if err := ctx.Err(); err != nil {
			return err
		}
		length := chunkSize
		if remaining := info.Size - offset; remaining < length {
			length = remaining
		}
		data := chunk[:int(length)]
		if err := file.ReadTensorRange(info, offset, data); err != nil {
			return fmt.Errorf("read tensor %q at %d: %w", info.Name, offset, err)
		}
		if uint64(pointer) > math.MaxUint64-offset {
			return fmt.Errorf("device pointer for tensor %q overflows", info.Name)
		}
		destination := pointer + driver.DevicePtr(offset)
		if err := w.worker.Do(ctx, func(state *device.State) error {
			return state.Driver.MemcpyHtoD(destination, data)
		}); err != nil {
			return fmt.Errorf("upload tensor %q at %d: %w", info.Name, offset, err)
		}
		offset += length
	}
	return nil
}

// Do exposes loaded device pointers only inside the owning worker callback.
func (w *DeviceWeights) Do(
	ctx context.Context,
	function func(*device.State, map[string]DeviceTensor) error,
) error {
	if function == nil {
		return errors.New("device weights: nil callback")
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

func (w *DeviceWeights) Lookup(name string) (DeviceTensor, bool) {
	if w == nil {
		return DeviceTensor{}, false
	}
	w.mu.RLock()
	defer w.mu.RUnlock()
	if w.closed {
		return DeviceTensor{}, false
	}
	value, ok := w.tensors[name]
	return value, ok
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

func (w *DeviceWeights) Count() int {
	if w == nil {
		return 0
	}
	w.mu.RLock()
	defer w.mu.RUnlock()
	return len(w.tensors)
}

// Close frees every loaded tensor in the owning CUDA context.
func (w *DeviceWeights) Close() error {
	if w == nil {
		return nil
	}
	w.mu.Lock()
	defer w.mu.Unlock()
	if w.closed {
		return nil
	}
	w.closed = true
	var pointers []driver.DevicePtr
	for _, tensor := range w.tensors {
		pointers = append(pointers, tensor.Pointer)
	}
	clear(w.tensors)
	return w.worker.Do(context.Background(), func(state *device.State) error {
		var errs []error
		for _, pointer := range pointers {
			if err := state.Driver.MemFree(pointer); err != nil {
				errs = append(errs, err)
			}
		}
		return errors.Join(errs...)
	})
}
