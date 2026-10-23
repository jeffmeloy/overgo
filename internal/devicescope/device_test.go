package devicescope

import (
	"os"
	"path/filepath"
	"slices"
	"testing"
)

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
	plan, err := PlanFor(root, []string{"kernels/cuda/owned.cu"})
	if err != nil || plan.Full || !slices.Equal(plan.Functions, []string{"owned_kernel"}) || !slices.Equal(plan.Packages, []string{"./internal/owner"}) {
		t.Fatalf("device plan = %+v, %v", plan, err)
	}
}

// TestPlanForReachesImportingTests holds that a Go change plans each lane
// package whose test binary reaches it: a recursive scope's subpackage by its
// test files' imports, through a production import, and an exact scope whose
// test files import it once one of them declares a test the lane runs, while
// a package's test-only import reaches nothing beyond it.
func TestPlanForReachesImportingTests(t *testing.T) {
	root := t.TempDir()
	for path, content := range map[string]string{
		"kernels/manifest.json":                     `{"modules":[]}`,
		"internal/modeldevice/expert_residency.go":  "package modeldevice\n",
		"internal/bridge/bridge.go":                 "package bridge\n\nimport _ \"overgo/internal/modeldevice\"\n",
		"internal/other/other.go":                   "package other\n",
		"internal/other/other_test.go":              "package other\n\nimport _ \"overgo/internal/modeldevice\"\n",
		"internal/cuda/executor/executor.go":        "package executor\n",
		"internal/cuda/executor/executor_test.go":   "package executor\n\nimport _ \"overgo/internal/bridge\"\n",
		"internal/cuda/driver/driver_test.go":       "package driver\n\nimport _ \"overgo/internal/other\"\n",
		"internal/cuda/driver/testdata/fixture.go":  "package fixture\n\nimport _ \"overgo/internal/modeldevice\"\n",
		"internal/inference/inference_test.go":      "package inference\n\nimport _ \"overgo/internal/modeldevice\"\n",
		"internal/inference/span_test.go":           "package inference\n\nfunc TestDeviceDecodeSpan() {}\n",
		"internal/optimizer/host_test.go":           "package optimizer\n\nimport _ \"overgo/internal/modeldevice\"\n\nfunc TestHost() {}\n",
		"internal/projector/projector.go":           "package projector\n",
		"internal/projector/projector_unit_test.go": "package projector\n\nimport _ \"overgo/internal/other\"\n",
	} {
		full := filepath.Join(root, filepath.FromSlash(path))
		if err := os.MkdirAll(filepath.Dir(full), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(full, []byte(content), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	plan, err := PlanFor(root, []string{"internal/modeldevice/expert_residency.go"})
	if want := []string{"./internal/cuda/executor", "./internal/inference"}; err != nil || plan.Full || !slices.Equal(plan.Packages, want) {
		t.Fatalf("device plan = %+v, %v; want packages %v", plan, err, want)
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
		plan, err := PlanFor(root, []string{changed})
		if err != nil || !plan.Full || plan.Reason == "" {
			t.Fatalf("fallback for %s = %+v, %v", changed, plan, err)
		}
	}
}
