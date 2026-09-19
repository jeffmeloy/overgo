package main

import (
	"testing"

	"overgo/internal/recipe"
)

// TestSurfaceAffected proves the commit-change-driven core: a kernel/module
// change moves every surface, a documentation or test change moves none, a
// hostmath change moves the training surface (this row's own premise), and an
// inference-source change moves the inference and evaluation surfaces.
func TestSurfaceAffected(t *testing.T) {
	root := "../.."
	ctx := t.Context()

	all, err := surfaceAffected(ctx, root, []string{"kernels/cuda/ops_f32.cu"})
	if err != nil {
		t.Fatal(err)
	}
	if len(all) != len(modalitySurfaces) {
		t.Fatalf("surfaces reported=%d want=%d", len(all), len(modalitySurfaces))
	}
	for id, rec := range all {
		if !rec.Affected {
			t.Fatalf("kernel change left %s unaffected", id)
		}
	}

	none, err := surfaceAffected(ctx, root, []string{"docs/MEDIA_REPORT.md", "internal/inference/runner_test.go"})
	if err != nil {
		t.Fatal(err)
	}
	for id, rec := range none {
		if rec.Affected {
			t.Fatalf("documentation/test change moved %s: %s", id, rec.Reason)
		}
	}

	host, err := surfaceAffected(ctx, root, []string{"internal/hostmath/hostmath.go"})
	if err != nil {
		t.Fatal(err)
	}
	if !host["training"].Affected {
		t.Fatal("hostmath change left the training surface unaffected")
	}

	inference, err := surfaceAffected(ctx, root, []string{"internal/inference/runner.go"})
	if err != nil {
		t.Fatal(err)
	}
	if !inference["inference"].Affected || !inference["evaluation"].Affected {
		t.Fatalf("inference change: inference=%+v evaluation=%+v", inference["inference"], inference["evaluation"])
	}
}

// TestKindValidation proves validation dispatch is by declared task, generic
// across models: text inference earns a guard and the accuracy benchmarks, an
// image-gen task its generation proof, and an unknown task nothing.
func TestKindValidation(t *testing.T) {
	if got := len(kindValidation(recipe.TaskInference)); got != 4 {
		t.Fatalf("inference cells=%d want 4", got)
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
