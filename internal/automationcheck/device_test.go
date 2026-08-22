package automationcheck

import (
	"context"
	"slices"
	"strings"
	"testing"
)

func TestDeviceCheck(t *testing.T) {
	var calls []string
	check := DeviceCheck(t.TempDir(), []string{"internal/cuda/kernel/load.go"}, recordingCommand(&calls))
	planned, err := Plan([]Check{check}, DeviceImpact([]string{"internal/cuda/kernel/load.go"}))
	if err != nil || len(planned) != 1 {
		t.Fatalf("plan = %+v, %v", planned, err)
	}
	if _, err := Run(context.Background(), planned[0]); err != nil {
		t.Fatal(err)
	}
	if len(calls) != 1 || !strings.Contains(calls[0], "./cmd/device-lane -paths internal/cuda/kernel/load.go") {
		t.Fatalf("calls = %v", calls)
	}
}

func TestDeviceResource(t *testing.T) {
	resources := DeviceCheck(".", nil, recordingCommand(new([]string))).Descriptor.Resources
	if len(resources) != 1 || resources[0].Name != "device" || !resources[0].Exclusive {
		t.Fatalf("resources = %+v", resources)
	}
}

func TestDeviceApplicability(t *testing.T) {
	for _, path := range []string{"kernels/cuda/a.cu", "internal/cuda/executor/run.go", "internal/optimizer/step_cuda_windows.go"} {
		if !slices.Equal(DeviceImpact([]string{path}), []Fact{deviceImpact}) {
			t.Errorf("%s did not require device evidence", path)
		}
	}
	if impact := DeviceImpact([]string{"internal/model/config.go"}); len(impact) != 0 {
		t.Fatalf("host-only impact = %v", impact)
	}
}
