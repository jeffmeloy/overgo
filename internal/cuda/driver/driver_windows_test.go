//go:build windows

package driver

import (
	"os"
	"testing"
)

func TestDriverIntegration(t *testing.T) {
	if os.Getenv("LLAMACPP2GO_CUDA_TEST") == "" {
		t.Skip("set LLAMACPP2GO_CUDA_TEST=1 to run CUDA integration tests")
	}

	lib, err := Open()
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
