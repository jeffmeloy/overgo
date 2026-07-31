//go:build windows

package device

import (
	"bytes"
	"context"
	"os"
	"testing"
)

func TestWorkerMemoryRoundTrip(t *testing.T) {
	if os.Getenv("LLAMACPP2GO_CUDA_TEST") == "" {
		t.Skip("set LLAMACPP2GO_CUDA_TEST=1 to run CUDA integration tests")
	}

	worker, err := New(0)
	if err != nil {
		t.Fatal(err)
	}
	defer worker.Close()

	source := []byte("llamacpp2go CUDA memory round trip")
	destination := make([]byte, len(source))
	var duringBytes uint64
	err = worker.Do(context.Background(), func(state *State) error {
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
	stats, err := worker.MemoryStats(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if duringBytes != 2*uint64(len(source)) ||
		stats.CurrentBytes != 0 ||
		stats.PeakBytes < uint64(len(source)) ||
		stats.Allocations != 0 {
		t.Fatalf("memory stats during=%d after=%+v", duringBytes, stats)
	}
	execution, err := worker.ExecutionStats(context.Background())
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
