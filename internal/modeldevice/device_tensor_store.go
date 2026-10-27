package modeldevice

import (
	"context"
	"errors"
	"fmt"
	"sync"

	"overgo/internal/cuda/device"
	"overgo/internal/cuda/driver"
	"overgo/internal/gguf"
)

// deviceMemory is the device surface a load needs: allocate, copy host bytes
// in, free. The worker is the only implementation outside tests.
type deviceMemory interface {
	alloc(ctx context.Context, size uint64) (driver.DevicePtr, error)
	write(ctx context.Context, destination driver.DevicePtr, data []byte) error
	free(ctx context.Context, pointers []driver.DevicePtr) error
}

// hostMappedMemory is pinned host memory the card maps, which a mapped route places a tensor in.
type hostMappedMemory interface {
	hostAlloc(ctx context.Context, size uint64) ([]byte, driver.DevicePtr, error)
	freeHosted(ctx context.Context, data [][]byte) error
}

type workerMemory struct{ worker *device.Worker }

func (m workerMemory) hostAlloc(ctx context.Context, size uint64) ([]byte, driver.DevicePtr, error) {
	var host []byte
	var pointer driver.DevicePtr
	err := m.worker.Do(ctx, func(state *device.State) error {
		var allocErr error
		host, pointer, allocErr = state.Driver.MemHostAlloc(size)
		return allocErr
	})
	return host, pointer, err
}

func (m workerMemory) freeHosted(ctx context.Context, data [][]byte) error {
	return m.worker.Do(ctx, func(state *device.State) error {
		var errs []error
		for _, host := range data {
			errs = append(errs, state.Driver.MemFreeHost(host))
		}
		return errors.Join(errs...)
	})
}

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
	// hosted is the host bytes of the tensors held in pinned host memory the card maps (Loader.Mapped).
	hosted map[string][]byte
	closed bool
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

// place allocates a tensor, copies its bytes in and publishes it under its name; a failure frees the allocation.
func (s *deviceTensorStore) place(ctx context.Context, info gguf.TensorInfo, data []byte) error {
	pointer, err := s.mem.alloc(ctx, uint64(len(data)))
	if err != nil {
		return err
	}
	release := func(err error) error {
		return errors.Join(err, s.mem.free(context.WithoutCancel(ctx), []driver.DevicePtr{pointer}))
	}
	if err := s.mem.write(ctx, pointer, data); err != nil {
		return release(err)
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if _, exists := s.tensors[info.Name]; exists || s.closed {
		return release(fmt.Errorf("%s tensor %q cannot be placed", s.kind, info.Name))
	}
	s.tensors[info.Name] = DeviceTensor{Info: info, Pointer: pointer}
	return nil
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
	var hosted [][]byte
	for name, value := range s.tensors {
		if host, mapped := s.hosted[name]; mapped {
			hosted = append(hosted, host)
			continue
		}
		pointers = append(pointers, value.Pointer)
	}
	clear(s.tensors)
	clear(s.hosted)
	err := s.mem.free(context.Background(), pointers)
	if len(hosted) != 0 {
		err = errors.Join(err, s.mem.(hostMappedMemory).freeHosted(context.Background(), hosted))
	}
	return err
}
