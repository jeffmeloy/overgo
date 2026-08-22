package automationcheck

import (
	"context"
	"os"
	"path/filepath"
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
		if !slices.Equal(DeviceImpact([]string{path}).Facts, []Fact{deviceImpact}) {
			t.Errorf("%s did not require device evidence", path)
		}
	}
	if impact := DeviceImpact([]string{"internal/model/config.go"}); len(impact.Facts) != 0 || len(impact.Exclusions) != 1 {
		t.Fatalf("host-only impact = %v", impact)
	}
}

func TestDeviceFunctionImpact(t *testing.T) {
	root := t.TempDir()
	if err := os.MkdirAll(filepath.Join(root, "kernels"), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.MkdirAll(filepath.Join(root, "internal", "owner"), 0o755); err != nil {
		t.Fatal(err)
	}
	manifest := `{"modules":[{"source":"kernels/cuda/owned.cu","asset":"internal/cuda/kernel/owned.ptx","functions":["owned_kernel"]}]}`
	if err := os.WriteFile(filepath.Join(root, "kernels", "manifest.json"), []byte(manifest), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(root, "internal", "owner", "launch.go"), []byte("package owner\nconst launch = \"owned_kernel\"\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	plan, err := DevicePlan(root, []string{"kernels/cuda/owned.cu"})
	if err != nil || plan.Full || !slices.Equal(plan.Functions, []string{"owned_kernel"}) || !slices.Equal(plan.Packages, []string{"./internal/owner"}) {
		t.Fatalf("device plan = %+v, %v", plan, err)
	}
}

func TestDeviceConservativeFallback(t *testing.T) {
	root := t.TempDir()
	if err := os.MkdirAll(filepath.Join(root, "kernels"), 0o755); err != nil {
		t.Fatal(err)
	}
	manifest := `{"modules":[{"source":"kernels/cuda/unowned.cu","functions":["unowned_kernel"]}]}`
	if err := os.WriteFile(filepath.Join(root, "kernels", "manifest.json"), []byte(manifest), 0o644); err != nil {
		t.Fatal(err)
	}
	for _, changed := range []string{"kernels/cuda/unknown.cu", "kernels/cuda/unowned.cu"} {
		plan, err := DevicePlan(root, []string{changed})
		if err != nil || !plan.Full || plan.Reason == "" {
			t.Fatalf("fallback for %s = %+v, %v", changed, plan, err)
		}
	}
}
