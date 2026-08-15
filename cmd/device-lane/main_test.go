package main

import (
	"slices"
	"testing"
)

func TestDeviceLaneSelectionAndReporting(t *testing.T) {
	paths := splitPaths("internal/optimizer/update_cuda_windows.go,kernels/attention.cu,internal/optimizer/other.go")
	steps := deviceSteps(paths)
	if len(steps) != 2 || !slices.Equal(steps[0], []string{"go", "run", "./cmd/cuda-smoke"}) {
		t.Fatalf("scoped steps = %v", steps)
	}
	want := []string{"go", "test", "./internal/cuda/...", "./internal/optimizer", "-count=1"}
	if !slices.Equal(steps[1], want) {
		t.Fatalf("selected packages = %v, want %v", steps[1], want)
	}
	if got := deviceScopeLabel(paths); got != "changed-paths(3)" {
		t.Fatalf("scope label = %q", got)
	}
	if got := deviceScopeLabel(nil); got != "full" || len(deviceSteps(nil)) != 3 {
		t.Fatalf("full scope = %q steps=%v", got, deviceSteps(nil))
	}
}
