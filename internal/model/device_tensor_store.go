package model

import (
	"context"
	"errors"
	"fmt"
	"sync"

	"overgo/internal/checked"
	"overgo/internal/cuda/device"
	"overgo/internal/cuda/driver"
	"overgo/internal/gguf"
)

// deviceTensorStore: shared CUDA tensor ownership.
type deviceTensorStore struct {
	worker *device.Worker
	kind   string

	mu      sync.RWMutex
	tensors map[string]DeviceTensor
	closed  bool
}

func newDeviceTensorStore(worker *device.Worker, kind string) *deviceTensorStore {
	return &deviceTensorStore{
		worker: worker, kind: kind, tensors: make(map[string]DeviceTensor),
	}
}

func (s *deviceTensorStore) load(
	ctx context.Context,
	infos []gguf.TensorInfo,
	upload func(gguf.TensorInfo) (DeviceTensor, error),
) (returnErr error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.closed {
		return fmt.Errorf("%s weights are closed", s.kind)
	}
	added := make([]string, 0, len(infos))
	defer func() {
		if returnErr == nil {
			return
		}
		pointers := make([]driver.DevicePtr, 0, len(added))
		for _, name := range added {
			pointers = append(pointers, s.tensors[name].Pointer)
			delete(s.tensors, name)
		}
		_ = s.worker.Do(context.Background(), func(state *device.State) error {
			for _, pointer := range pointers {
				_ = state.Driver.MemFree(pointer)
			}
			return nil
		})
	}()
	for _, info := range infos {
		if _, exists := s.tensors[info.Name]; exists {
			return fmt.Errorf("%s tensor %q is already loaded", s.kind, info.Name)
		}
		if err := ctx.Err(); err != nil {
			return err
		}
		value, err := upload(info)
		if err != nil {
			return err
		}
		s.tensors[info.Name] = value
		added = append(added, info.Name)
	}
	return nil
}

func (s *deviceTensorStore) loadConverted(
	ctx context.Context,
	file *gguf.File,
	infos []gguf.TensorInfo,
	bytesPerValue uint64,
	encode func([]float32, int) []byte,
) error {
	return s.load(ctx, infos, func(info gguf.TensorInfo) (DeviceTensor, error) {
		value, err := LoadHostTensor(ctx, file, info)
		if err != nil {
			return DeviceTensor{}, err
		}
		byteCount, ok := checked.Bytes(uint64(len(value.Data)), bytesPerValue)
		if !ok {
			return DeviceTensor{}, fmt.Errorf("%s tensor %q byte size overflows", s.kind, info.Name)
		}
		storageSize, ok := checked.Int(byteCount)
		if !ok {
			return DeviceTensor{}, fmt.Errorf("%s tensor %q exceeds host address space", s.kind, info.Name)
		}
		storage := encode(value.Data, storageSize)
		if len(storage) != storageSize {
			return DeviceTensor{}, fmt.Errorf("%s tensor %q encoding size differs", s.kind, info.Name)
		}
		var pointer driver.DevicePtr
		if err := s.worker.Do(ctx, func(state *device.State) error {
			var allocateErr error
			pointer, allocateErr = state.Driver.MemAlloc(byteCount)
			if allocateErr != nil {
				return allocateErr
			}
			if copyErr := state.Driver.MemcpyHtoD(pointer, storage); copyErr != nil {
				_ = state.Driver.MemFree(pointer)
				return copyErr
			}
			return nil
		}); err != nil {
			return DeviceTensor{}, fmt.Errorf("upload %s tensor %q: %w", s.kind, info.Name, err)
		}
		return DeviceTensor{Info: info, Shape: value.Shape, Pointer: pointer, Size: byteCount}, nil
	})
}

func (s *deviceTensorStore) Lookup(name string) (DeviceTensor, bool) {
	if s == nil {
		return DeviceTensor{}, false
	}
	s.mu.RLock()
	defer s.mu.RUnlock()
	if s.closed {
		return DeviceTensor{}, false
	}
	value, ok := s.tensors[name]
	return value, ok
}

func (s *deviceTensorStore) Close() error {
	if s == nil {
		return nil
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.closed {
		return nil
	}
	s.closed = true
	pointers := make([]driver.DevicePtr, 0, len(s.tensors))
	for _, value := range s.tensors {
		pointers = append(pointers, value.Pointer)
	}
	clear(s.tensors)
	return s.worker.Do(context.Background(), func(state *device.State) error {
		var errs []error
		for _, pointer := range pointers {
			if err := state.Driver.MemFree(pointer); err != nil {
				errs = append(errs, err)
			}
		}
		return errors.Join(errs...)
	})
}
