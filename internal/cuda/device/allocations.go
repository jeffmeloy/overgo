package device

import (
	"context"
	"errors"

	"overgo/internal/cuda/driver"
)

// AllocationSet: ordered worker-owned device allocations.
type AllocationSet struct {
	worker   *Worker
	pointers []driver.DevicePtr
}

func NewAllocationSet(worker *Worker) AllocationSet {
	return AllocationSet{worker: worker}
}

func (a *AllocationSet) Allocate(ctx context.Context, bytes uint64) (driver.DevicePtr, error) {
	if a == nil || a.worker == nil {
		return 0, errors.New("CUDA allocation set is unavailable")
	}
	var pointer driver.DevicePtr
	err := a.worker.Do(ctx, func(state *State) error {
		var err error
		pointer, err = state.Driver.MemAlloc(bytes)
		return err
	})
	if err == nil {
		a.pointers = append(a.pointers, pointer)
	}
	return pointer, err
}

func (a *AllocationSet) Upload(ctx context.Context, payload []byte) (driver.DevicePtr, error) {
	if a == nil || a.worker == nil {
		return 0, errors.New("CUDA allocation set is unavailable")
	}
	var pointer driver.DevicePtr
	err := a.worker.Do(ctx, func(state *State) error {
		var err error
		pointer, err = state.Driver.MemAlloc(uint64(len(payload)))
		if err != nil {
			return err
		}
		if err := state.Driver.MemcpyHtoD(pointer, payload); err != nil {
			return errors.Join(err, state.Driver.MemFree(pointer))
		}
		return nil
	})
	if err == nil {
		a.pointers = append(a.pointers, pointer)
	}
	return pointer, err
}

func (a *AllocationSet) Close(ctx context.Context) error {
	if a == nil || a.worker == nil {
		return errors.New("CUDA allocation set is unavailable")
	}
	if len(a.pointers) == 0 {
		return nil
	}
	err := a.worker.Do(ctx, func(state *State) error {
		var result error
		for _, pointer := range a.pointers {
			result = errors.Join(result, state.Driver.MemFree(pointer))
		}
		return result
	})
	a.pointers = nil
	return err
}
