//go:build windows

// Package driver_test verifies the public CUDA driver boundary.
package driver_test

import (
	"testing"

	"overgo/internal/cuda/driver"
	cudatest "overgo/internal/cuda/testutil"
)

func TestDriverIntegration(t *testing.T) {
	cudatest.Require(t)

	lib, err := driver.Open()
	if err != nil {
		t.Fatal(err)
	}
	defer lib.Close()

	if err := lib.Init(); err != nil {
		t.Fatal(err)
	}
	version, err := lib.DriverVersion()
	if err != nil {
		t.Fatal(err)
	}
	if version <= 0 {
		t.Fatalf("invalid driver version %d", version)
	}
	count, err := lib.DeviceCount()
	if err != nil {
		t.Fatal(err)
	}
	if count < 1 {
		t.Fatal("no CUDA devices found")
	}
	info, err := lib.DeviceInfo(0)
	if err != nil {
		t.Fatal(err)
	}
	if info.Name == "" || info.TotalMemoryBytes == 0 {
		t.Fatalf("incomplete device info: %+v", info)
	}
}

// TestStreamQueryReportsCompletedWork holds StreamQuery to cuStreamQuery: a
// stream with nothing queued has completed its work, and so has one whose
// queued copy was waited for; neither is an error.
func TestStreamQueryReportsCompletedWork(t *testing.T) {
	cudatest.Require(t)
	lib, err := driver.Open()
	if err != nil {
		t.Fatal(err)
	}
	defer lib.Close()
	if err := lib.Init(); err != nil {
		t.Fatal(err)
	}
	device, err := lib.Device(0)
	if err != nil {
		t.Fatal(err)
	}
	context, err := lib.ContextCreate(device, 0)
	if err != nil {
		t.Fatal(err)
	}
	defer lib.ContextDestroy(context)
	stream, err := lib.StreamCreate(0)
	if err != nil {
		t.Fatal(err)
	}
	defer lib.StreamDestroy(stream)
	if done, err := lib.StreamQuery(stream); !done || err != nil {
		t.Fatalf("an idle stream's query = %t, %v; want completed", done, err)
	}
	const bytes = 1 << 20
	buffer, err := lib.MemAlloc(bytes)
	if err != nil {
		t.Fatal(err)
	}
	defer lib.MemFree(buffer)
	if err := lib.MemcpyHtoDAsync(buffer, make([]byte, bytes), stream); err != nil {
		t.Fatal(err)
	}
	if _, err := lib.StreamQuery(stream); err != nil {
		t.Fatalf("a working stream's query: %v", err)
	}
	if err := lib.StreamSynchronize(stream); err != nil {
		t.Fatal(err)
	}
	if done, err := lib.StreamQuery(stream); !done || err != nil {
		t.Fatalf("a synchronized stream's query = %t, %v; want completed", done, err)
	}
}
