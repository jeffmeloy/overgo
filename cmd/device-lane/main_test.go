package main

import (
	"os"
	"slices"
	"strings"
	"testing"
	"time"
)

func TestDeviceLaneProgress(t *testing.T) {
	line := deviceProgress([]string{"go", "test", "./internal/cuda/..."}, 1, 3, 2*time.Second)
	for _, field := range []string{"phase=2/3", "heartbeat=running", "elapsed=2.0s", "go test ./internal/cuda/..."} {
		if !strings.Contains(line, field) {
			t.Fatalf("progress %q lacks %q", line, field)
		}
	}
}

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
	full := deviceSteps(nil)
	if len(full) != 3 {
		t.Fatalf("full steps=%v", full)
	}
	for _, step := range full[1:] {
		if !slices.Contains(step, "-p=1") {
			t.Fatalf("device test step is not serialized: %v", step)
		}
		if !slices.Contains(step, "-timeout=20m") {
			t.Fatalf("device test step lacks real-suite timeout: %v", step)
		}
	}
}

func TestDeviceLaneRunsChangedTestPackage(t *testing.T) {
	working, err := os.Getwd()
	if err != nil {
		t.Fatal(err)
	}
	if err := os.Chdir("../.."); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = os.Chdir(working) })
	steps := deviceSteps([]string{"internal/adaptiveparity/gemma4_fp8_leadership_windows_test.go"})
	want := []string{
		"go", "test", "-p=1", "-timeout=20m", "./internal/adaptiveparity", "-count=1",
	}
	if len(steps) != 2 || !slices.Equal(steps[1], want) {
		t.Fatalf("selected test step = %v, want %v", steps, want)
	}
}
