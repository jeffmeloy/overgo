package model

import (
	"context"
	"errors"
	"fmt"
	"sync"

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

func (s *deviceTensorStore) Count() int {
	if s == nil {
		return 0
	}
	s.mu.RLock()
	defer s.mu.RUnlock()
	return len(s.tensors)
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
