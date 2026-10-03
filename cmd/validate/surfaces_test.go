package main

import (
	"testing"

	"overgo/internal/recipe"
)

// TestSurfaceMoves holds the rule a commit-bound record expires by: a kernel
// or module change moves a surface whatever its packages, a change in one of
// its package directories moves it, and a documentation or test change, or
// one outside its directories, moves nothing.
func TestSurfaceMoves(t *testing.T) {
	t.Parallel()
	training := []string{"internal/hostmath", "internal/trainingworkflow"}
	for _, changed := range [][]string{{"kernels/cuda/ops_f32.cu"}, {"go.mod"}, {"docs/notes.md", "internal/hostmath/hostmath.go"}} {
		if moved := surfaceMoves(training, changed); !moved.Affected || moved.Reason == "" {
			t.Errorf("%v left the surface unmoved: %+v", changed, moved)
		}
	}
	for _, changed := range [][]string{{"docs/MEDIA_REPORT.md", "internal/hostmath/hostmath_test.go"}, {"internal/hostmathx/a.go", "internal/inference/runner.go"}} {
		if moved := surfaceMoves(training, changed); moved.Affected {
			t.Errorf("%v moved the surface: %s", changed, moved.Reason)
		}
	}
}

// TestKindValidation proves validation dispatch is by declared task, generic
// across models: inference earns a guard, a measured serving attempt and the
// accuracy benchmarks, an image-gen task its generation proof, and an unknown
// task nothing.
func TestKindValidation(t *testing.T) {
	t.Parallel()
	if got := len(kindValidation(recipe.TaskInference)); got != 5 {
		t.Fatalf("inference cells=%d want 5", got)
	}
	if got := len(kindValidation(recipe.TaskImageGen)); got != 1 {
		t.Fatalf("image-gen cells=%d want 1", got)
	}
	if got := len(kindValidation(recipe.TaskTraining)); got != 1 {
		t.Fatalf("training cells=%d want 1", got)
	}
	if kindValidation(recipe.Task("nonexistent-task")) != nil {
		t.Fatal("unknown task produced validation cells")
	}
}
