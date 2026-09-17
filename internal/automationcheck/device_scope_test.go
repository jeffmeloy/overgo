package automationcheck

import (
	"os"
	"path/filepath"
	"slices"
	"testing"
)

func TestDeviceDeclaredConsumers(t *testing.T) {
	root := t.TempDir()
	files := map[string]string{
		"kernels/manifest.json":                         `{"modules":[{"source":"kernels/cuda/owned.cu","asset":"internal/cuda/kernel/owned.ptx","functions":["owned_kernel"]}]}`,
		"internal/cuda/executor/launch_cuda_windows.go": "package executor\nconst launch = \"owned_kernel\"\n",
		"internal/modeldevice/model.go":                 "package modeldevice\nimport _ \"overgo/internal/cuda/executor\"\n",
		"internal/inference/runner.go":                  "package inference\nimport _ \"overgo/internal/modeldevice\"\n",
		"internal/optimizer/host_test.go":               "package optimizer\nimport _ \"overgo/internal/cuda/executor\"\n",
		"internal/projector/host.go":                    "package projector\n",
	}
	for path, content := range files {
		path = filepath.Join(root, filepath.FromSlash(path))
		if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	for _, path := range []string{
		"kernels/cuda/owned.cu", "internal/cuda/kernel/owned.ptx",
		"internal/cuda/executor/launch_cuda_windows.go", "internal/modeldevice/model.go", "internal/inference/runner.go",
	} {
		t.Run(path, func(t *testing.T) {
			plan, err := DevicePlan(root, []string{path})
			if err != nil || plan.Full || !slices.Contains(plan.Packages, "./internal/inference") {
				t.Fatalf("device consumers = %+v, %v", plan, err)
			}
			if slices.Contains(plan.Packages, "./internal/optimizer") || slices.Contains(plan.Packages, "./internal/projector") {
				t.Fatalf("non-production consumer selected: %+v", plan)
			}
			assertInferenceDeviceScope(t, DeviceScopes(plan))
		})
	}
	for _, path := range []string{"internal/cuda/executor/unknown.go", "kernels/cuda/unknown.cu", "kernels/manifest.json"} {
		plan, err := DevicePlan(root, []string{path})
		if err != nil || !plan.Full {
			t.Fatalf("fallback = %+v, %v", plan, err)
		}
		assertInferenceDeviceScope(t, DeviceScopes(plan))
	}
	plan, err := DevicePlan(root, []string{"internal/unrelated/host.go"})
	if err != nil || plan.Full || len(plan.Packages) != 0 {
		t.Fatalf("unrelated change = %+v, %v", plan, err)
	}
}

func TestDeviceFullScopeOwnsInference(t *testing.T) {
	plan, err := DevicePlan(t.TempDir(), nil)
	if err != nil || !plan.Full {
		t.Fatalf("full plan = %+v, %v", plan, err)
	}
	assertInferenceDeviceScope(t, DeviceScopes(plan))
	if !slices.Contains(DeviceLanePackages(), "internal/inference") {
		t.Fatal("impact ownership omitted inference")
	}
	full := DeviceScopes(plan)
	for _, scope := range full {
		if len(scope.Tests) > 0 {
			scope.Tests[0] = "mutated caller copy"
		}
	}
	assertInferenceDeviceScope(t, DeviceScopes(plan))
}

func assertInferenceDeviceScope(t *testing.T, scopes []DeviceTestScope) {
	t.Helper()
	for _, scope := range scopes {
		if scope.Package != "./internal/inference" {
			continue
		}
		for _, name := range []string{"TestRecurrentCheckpoint", "TestNextNMTPDeviceSpeculationLossless", "TestHermeticCUDAContinuousCacheParity"} {
			if !slices.Contains(scope.Tests, name) {
				t.Fatalf("inference contract %s absent from %+v", name, scope)
			}
		}
		if scope.Run != "" {
			t.Fatalf("inference uses a broad test pattern: %+v", scope)
		}
		return
	}
	t.Fatal("inference device scope missing")
}
