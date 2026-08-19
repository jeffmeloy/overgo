package main

import (
	"os"
	"slices"
	"testing"
)

func TestDeviceLaneSelectionAndReporting(t *testing.T) {
	working, err := os.Getwd()
	if err != nil {
		t.Fatal(err)
	}
	if err := os.Chdir("../.."); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = os.Chdir(working) })
	paths := splitPaths("internal/optimizer/muon_step_cuda_windows.go,kernels/cuda/vector_add.cu,internal/optimizer/removed_cuda_windows.go")
	steps := deviceSteps(paths)
	if len(steps) != 2 || !slices.Equal(steps[0], []string{"go", "run", "./cmd/cuda-smoke"}) {
		t.Fatalf("scoped steps = %v", steps)
	}
	want := []string{"go", "test", "-p=1", "-timeout=20m", "./internal/cuda/...", "./internal/optimizer", "-count=1"}
	if !slices.Equal(steps[1], want) {
		t.Fatalf("selected packages = %v, want %v", steps[1], want)
	}
	if got := deviceScopeLabel(paths); got != "changed-paths(3)" {
		t.Fatalf("scope label = %q", got)
	}
	if got := deviceScopeLabel(nil); got != "full" || len(deviceSteps(nil)) != 3 {
		t.Fatalf("full scope = %q steps=%v", got, deviceSteps(nil))
	}
	for _, step := range deviceSteps(nil)[1:] {
		if !slices.Contains(step, "-p=1") {
			t.Fatalf("device test step is not serialized: %v", step)
		}
		if !slices.Contains(step, "-timeout=20m") {
			t.Fatalf("device test step lacks real-suite timeout: %v", step)
		}
	}
}
