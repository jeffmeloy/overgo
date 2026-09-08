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
