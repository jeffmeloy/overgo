package model

import (
	"context"
	"errors"
	"fmt"
	"sync"
	"unsafe"

	"llamacpp2go/internal/cuda/device"
	"llamacpp2go/internal/cuda/driver"
	"llamacpp2go/internal/gguf"
	"llamacpp2go/internal/tensor"
	"llamacpp2go/internal/tensor/dtype"
)

type DeviceF32Tensor struct {
	Info    gguf.TensorInfo
	Shape   tensor.Shape
	Pointer driver.DevicePtr
	Size    uint64
}

// DeviceF32Weights: owns host-dequantized F32 weights in one CUDA context
// correctness bridge used before native quantized CUDA matmul
type DeviceF32Weights struct {
	worker *device.Worker

	mu      sync.RWMutex
	tensors map[string]DeviceF32Tensor
	closed  bool
}

func NewDeviceF32Weights(worker *device.Worker) (*DeviceF32Weights, error) {
	if worker == nil {
		return nil, errors.New("F32 device weights require a CUDA worker")
	}
	return &DeviceF32Weights{
		worker:  worker,
		tensors: make(map[string]DeviceF32Tensor),
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
	w.mu.Lock()
	defer w.mu.Unlock()
	if w.closed {
		return errors.New("F32 device weights are closed")
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
	for _, info := range infos {
		if _, exists := w.tensors[info.Name]; exists {
			return fmt.Errorf("F32 device tensor %q is already loaded", info.Name)
		}
		value, err := LoadHostTensor(ctx, file, info)
		if err != nil {
			return err
		}
		if uint64(len(value.Data)) > ^uint64(0)/4 {
			return fmt.Errorf("F32 device tensor %q byte size overflows", info.Name)
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
			return fmt.Errorf("upload F32 device tensor %q: %w", info.Name, err)
		}
		w.tensors[info.Name] = DeviceF32Tensor{
			Info:    info,
			Shape:   value.Shape,
			Pointer: pointer,
			Size:    size,
		}
		added = append(added, info.Name)
	}
	return nil
}

func (w *DeviceF32Weights) Lookup(name string) (DeviceF32Tensor, bool) {
	if w == nil {
		return DeviceF32Tensor{}, false
	}
	w.mu.RLock()
	defer w.mu.RUnlock()
	if w.closed {
		return DeviceF32Tensor{}, false
	}
	value, ok := w.tensors[name]
	return value, ok
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
	if builder == nil {
		return LayerGraphWeights{}, nil, errors.New("F32 device layer graph builder is nil")
	}
	if info.FeedForwardRouter != nil &&
		((info.FeedForwardUpExperts == nil && info.FeedForwardGateUpExperts == nil) ||
			info.FeedForwardDownExperts == nil) {
		return LayerGraphWeights{}, nil, errors.New("F32 device expert layer catalog is incomplete")
	}
	feeds := make(map[*tensor.Tensor]driver.DevicePtr, 11)
	result := LayerGraphWeights{}
	if err := bindDeviceLayerGraphFields(w, builder, &info, &result, feeds); err != nil {
		return LayerGraphWeights{}, nil, err
	}
	if err := builder.Err(); err != nil {
		return LayerGraphWeights{}, nil, err
	}
	return result, feeds, nil
}

func (w *DeviceF32Weights) Count() int {
	if w == nil {
		return 0
	}
	w.mu.RLock()
	defer w.mu.RUnlock()
	return len(w.tensors)
}

func (w *DeviceF32Weights) Close() error {
	if w == nil {
		return nil
	}
	w.mu.Lock()
	defer w.mu.Unlock()
	if w.closed {
		return nil
	}
	w.closed = true
	pointers := make([]driver.DevicePtr, 0, len(w.tensors))
	for _, value := range w.tensors {
		pointers = append(pointers, value.Pointer)
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

func f32Bytes(values []float32) []byte {
	if len(values) == 0 {
		return nil
	}
	return unsafe.Slice((*byte)(unsafe.Pointer(&values[0])), len(values)*4)
}
