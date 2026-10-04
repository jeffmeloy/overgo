//go:build windows

package devicemath

import (
	"context"
	"fmt"

	"overgo/internal/checked"
	"overgo/internal/cuda/device"
	"overgo/internal/cuda/driver"
)

// LayerTensorOffsets are one pre-norm layer's tensor offsets
// tensors within a flat device buffer (the same offsets serve the weight buffer and
// the gradient buffer, which share a layout). The caller (densecausal) computes
// these from its sorted optimizer plan; the resident stack resolves them against the
// flat weight/grad base pointers.
type LayerTensorOffsets struct {
	InLN, PostLN, Q, K, V, O, Gate, Up, Down int
	QBias, KBias, VBias                      int
	AttnBias                                 bool
}

// LayerTrainVectors carries host-updated layer vectors.
type LayerTrainVectors struct {
	InLN, PostLN        []float32
	QBias, KBias, VBias []float32
}

func ptrAt(base driver.DevicePtr, elems int) driver.DevicePtr {
	return base + driver.DevicePtr(uint64(elems)*f32Bytes)
}

// ResidentPtr returns the device pointer elems f32 elements into a flat resident
// buffer -- the caller-facing offset helper for addressing sub-tensors of dW/dG/dM.
func ResidentPtr(base driver.DevicePtr, elems int) driver.DevicePtr {
	return ptrAt(base, elems)
}

// ResidentSlice names a host buffer and the element offset of its device home
// within a flat resident buffer, for batched partial transfers.
type ResidentSlice struct {
	ElemOffset int
	Data       []float32
}

// AllocResidentF32 allocates a persistent device f32 buffer of count elements and
// either uploads init (len must equal count) or zeroes it (init nil). The pointer
// survives across worker.Do calls until FreeResident releases it -- the backbone of
// across-step weight/momentum residency.
func AllocResidentF32(worker *device.Worker, count int, init []float32) (driver.DevicePtr, error) {
	if count <= 0 {
		return 0, fmt.Errorf("AllocResidentF32: count %d must be positive", count)
	}
	if init != nil && len(init) != count {
		return 0, fmt.Errorf("AllocResidentF32: init len %d != count %d", len(init), count)
	}
	bytes, ok := checked.Bytes(uint64(count), f32Bytes)
	if !ok {
		return 0, fmt.Errorf("AllocResidentF32: size overflow for %d elements", count)
	}
	var ptr driver.DevicePtr
	err := worker.Do(context.Background(), func(state *device.State) error {
		p, err := state.Driver.MemAlloc(bytes)
		if err != nil {
			return err
		}
		if init != nil {
			if err := state.Driver.MemcpyHtoD(p, driver.Bytes(init)); err != nil {
				_ = state.Driver.MemFree(p)
				return err
			}
		} else if err := state.Driver.MemsetD32Async(p, 0, uint64(count), state.Stream); err != nil {
			_ = state.Driver.MemFree(p)
			return err
		}
		if err := state.Driver.StreamSynchronize(state.Stream); err != nil {
			_ = state.Driver.MemFree(p)
			return err
		}
		ptr = p
		return nil
	})
	if err != nil {
		return 0, err
	}
	return ptr, nil
}

// FreeResident releases persistent device buffers allocated by AllocResidentF32.
func FreeResident(worker *device.Worker, ptrs ...driver.DevicePtr) error {
	return worker.Do(context.Background(), func(state *device.State) error {
		var err error
		for _, p := range ptrs {
			if p != 0 {
				if e := state.Driver.MemFree(p); e != nil && err == nil {
					err = e
				}
			}
		}
		return err
	})
}

// ReadResident copies the named element ranges of a flat resident buffer back to
// their host slices in one session (device-to-host).
func ReadResident(worker *device.Worker, base driver.DevicePtr, slices ...ResidentSlice) error {
	return worker.Do(context.Background(), func(state *device.State) error {
		if err := state.Driver.StreamSynchronize(state.Stream); err != nil {
			return err
		}
		for _, sl := range slices {
			if err := state.Driver.MemcpyDtoH(driver.Bytes(sl.Data), ptrAt(base, sl.ElemOffset)); err != nil {
				return err
			}
		}
		return nil
	})
}
