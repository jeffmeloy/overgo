//go:build windows

package device

import (
	"testing"

	"overgo/internal/cuda/driver"
	cudatest "overgo/internal/cuda/testutil"
)

func TestUnavailableDeviceAllowsRecovery(t *testing.T) {
	cudatest.Require(t)
	if cudatest.MeasurementProcess(t, DefaultOrdinal()) {
		return
	}
	library, err := driver.Open()
	if err != nil {
		t.Fatal(err)
	}
	defer library.Close()
	if err := library.Init(); err != nil {
		t.Fatal(err)
	}
	count, err := library.DeviceCount()
	if err != nil || count <= 0 {
		t.Fatal("missing admitted device", count, err)
	}
	// CUDA ordinals end at count-1. Derive an unavailable ordinal from the
	// actual host instead of assuming a particular number of installed GPUs.
	worker, err := New(count)
	if worker != nil {
		_ = worker.Close()
		t.Fatal("unavailable device returned a worker")
	}
	if err == nil {
		t.Fatal("unavailable device was accepted")
	}
	worker, err = New(DefaultOrdinal())
	if err != nil {
		t.Fatal("valid device failed after unavailable-device rejection", err)
	}
	var owner *driver.Library
	if err := worker.Do(t.Context(), func(state *State) error {
		owner = state.Driver
		pointer, err := state.Driver.MemAlloc(1)
		if err != nil {
			return err
		}
		return state.Driver.MemFree(pointer)
	}); err != nil {
		_ = worker.Close()
		t.Fatal(err)
	}
	if err := worker.Close(); err != nil {
		t.Fatal(err)
	}
	if stats := owner.MemoryStats(); stats.CurrentBytes != 0 {
		t.Fatal("recovered device retained allocation", stats.CurrentBytes)
	}
}
