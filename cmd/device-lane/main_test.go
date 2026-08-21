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
	steps, err := deviceSteps(paths)
	if err != nil {
		t.Fatal(err)
	}
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
	full, err := deviceSteps(nil)
	if err != nil {
		t.Fatal(err)
	}
	if got := deviceScopeLabel(nil); got != "full" || len(full) != 3 {
		t.Fatalf("full scope = %q steps=%v", got, full)
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

func TestDeviceLaneSelectsChangedTestsWithoutRunningUnrelatedArtifacts(t *testing.T) {
	working, err := os.Getwd()
	if err != nil {
		t.Fatal(err)
	}
	if err := os.Chdir("../.."); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = os.Chdir(working) })
	steps, err := deviceSteps([]string{"internal/adaptiveparity/gemma4_fp8_leadership_windows_test.go"})
	if err != nil {
		t.Fatal(err)
	}
	want := []string{
		"go", "test", "-p=1", "-timeout=20m", "-run", "^(TestGemma4FP8Leadership)$",
		"./internal/adaptiveparity", "-count=1",
	}
	if len(steps) != 2 || !slices.Equal(steps[1], want) {
		t.Fatalf("selected test step = %v, want %v", steps, want)
	}
}
