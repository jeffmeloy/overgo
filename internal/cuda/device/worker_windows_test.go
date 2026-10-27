//go:build windows

package device

import (
	"bytes"
	"errors"
	cudatest "overgo/internal/cuda/testutil"
	"sync"
	"sync/atomic"
	"testing"
)

func TestWorkerMemoryRoundTrip(t *testing.T) {
	cudatest.Require(t)

	worker, err := New(0)
	if err != nil {
		t.Fatal(err)
	}
	defer worker.Close()

	source := []byte("overgo CUDA memory round trip")
	destination := make([]byte, len(source))
	var duringBytes uint64
	err = worker.Do(t.Context(), func(state *State) error {
		pointer, allocErr := state.Driver.MemAlloc(uint64(len(source)))
		if allocErr != nil {
			return allocErr
		}
		defer state.Driver.MemFree(pointer)
		copyPointer, allocErr := state.Driver.MemAlloc(uint64(len(source)))
		if allocErr != nil {
			return allocErr
		}
		defer state.Driver.MemFree(copyPointer)
		duringBytes = state.Driver.MemoryStats().CurrentBytes
		if copyErr := state.Driver.MemcpyHtoD(pointer, source); copyErr != nil {
			return copyErr
		}
		if copyErr := state.Driver.MemcpyDtoD(
			copyPointer,
			pointer,
			uint64(len(source)),
		); copyErr != nil {
			return copyErr
		}
		if copyErr := state.Driver.MemcpyDtoH(destination, copyPointer); copyErr != nil {
			return copyErr
		}
		return state.Driver.StreamSynchronize(state.Stream)
	})
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(destination, source) {
		t.Fatalf("round trip = %q, want %q", destination, source)
	}
	stats, err := worker.MemoryStats(t.Context())
	if err != nil {
		t.Fatal(err)
	}
	if duringBytes != 2*uint64(len(source)) ||
		stats.CurrentBytes != 0 ||
		stats.PeakBytes < uint64(len(source)) ||
		stats.Allocations != 0 {
		t.Fatalf("memory stats during=%d after=%+v", duringBytes, stats)
	}
	execution, err := worker.ExecutionStats(t.Context())
	if err != nil {
		t.Fatal(err)
	}
	if execution.HostToDeviceCopies != 1 ||
		execution.HostToDeviceBytes != uint64(len(source)) ||
		execution.DeviceToHostCopies != 1 ||
		execution.DeviceToHostBytes != uint64(len(source)) ||
		execution.DeviceToDeviceCopies != 1 ||
		execution.DeviceToDeviceBytes != uint64(len(source)) ||
		execution.StreamSynchronizations != 1 {
		t.Fatalf("execution stats = %+v", execution)
	}
}

// TestSpinningWorkerRunsHandedRequests: a spinning worker runs the requests two goroutines hand it, one at a time on
// its thread, each answering its own error, and serves its channel again once it stops spinning.
func TestSpinningWorkerRunsHandedRequests(t *testing.T) {
	cudatest.Require(t)
	worker, err := New(0)
	if err != nil {
		t.Fatal(err)
	}
	defer worker.Close()
	if err := worker.Spin(t.Context(), true); err != nil {
		t.Fatal(err)
	}
	const requests = 1000
	failed := errors.New("the request's own failure")
	var running, ran atomic.Int64
	var group sync.WaitGroup
	for caller := range 2 {
		group.Go(func() {
			for index := range requests {
				err := worker.Do(t.Context(), func(state *State) error {
					if running.Add(1) != 1 || state == nil {
						t.Error("two requests ran at once, or without the worker's state")
					}
					running.Add(-1)
					ran.Add(1)
					if index%2 == caller {
						return failed
					}
					return nil
				})
				if (index%2 == caller) != errors.Is(err, failed) {
					t.Errorf("request %d of caller %d answered %v", index, caller, err)
					return
				}
			}
		})
	}
	group.Wait()
	if err := worker.Spin(t.Context(), false); err != nil {
		t.Fatal(err)
	}
	if err := worker.Do(t.Context(), func(*State) error { ran.Add(1); return nil }); err != nil || ran.Load() != 2*requests+1 {
		t.Fatalf("after the spin %d requests ran: %v", ran.Load(), err)
	}
}
