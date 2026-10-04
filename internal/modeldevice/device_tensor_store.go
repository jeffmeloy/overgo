package modeldevice

import (
	"context"
	"errors"
	"sync"

	"overgo/internal/cuda/device"
	"overgo/internal/cuda/driver"
)

// deviceMemory is the device surface a load needs: allocate, copy host bytes
// in, free. The worker is the only implementation outside tests.
type deviceMemory interface {
	alloc(ctx context.Context, size uint64) (driver.DevicePtr, error)
	write(ctx context.Context, destination driver.DevicePtr, data []byte) error
	free(ctx context.Context, pointers []driver.DevicePtr) error
}

type workerMemory struct{ worker *device.Worker }

func (m workerMemory) alloc(ctx context.Context, size uint64) (driver.DevicePtr, error) {
	var pointer driver.DevicePtr
	err := m.worker.Do(ctx, func(state *device.State) error {
		var allocErr error
		pointer, allocErr = state.Driver.MemAlloc(size)
		return allocErr
	})
	return pointer, err
}

func (m workerMemory) write(ctx context.Context, destination driver.DevicePtr, data []byte) error {
	return m.worker.Do(ctx, func(state *device.State) error {
		return state.Driver.MemcpyHtoD(destination, data)
	})
}

func (m workerMemory) free(ctx context.Context, pointers []driver.DevicePtr) error {
	return m.worker.Do(ctx, func(state *device.State) error {
		var errs []error
		for _, pointer := range pointers {
			if err := state.Driver.MemFree(pointer); err != nil {
				errs = append(errs, err)
			}
		}
		return errors.Join(errs...)
	})
}

// deviceTensorStore: shared CUDA tensor ownership.
type deviceTensorStore struct {
	worker *device.Worker
	mem    deviceMemory
	kind   string

	mu      sync.RWMutex
	tensors map[string]DeviceTensor
	closed  bool
}

func newDeviceTensorStore(worker *device.Worker, kind string) *deviceTensorStore {
	return &deviceTensorStore{
		worker: worker, mem: workerMemory{worker}, kind: kind, tensors: make(map[string]DeviceTensor),
	}
}

// Lookup returns the loaded tensor by name.
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

// Close frees every loaded tensor.
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
	return s.mem.free(context.Background(), pointers)
}
