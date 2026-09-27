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
