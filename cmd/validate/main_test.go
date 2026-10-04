package main

import (
	"errors"
	"io/fs"
	"os"
	"path/filepath"
	"testing"

	"overgo/internal/recipe"
)

// TestKindValidation proves validation dispatch is by declared task, generic
// across models: inference earns a guard, a measured serving attempt and the
// accuracy benchmarks, an image-gen task its generation proof, a thought
// bank's generation the specialized proof its acceptance covers, and an
// unknown task nothing.
func TestKindValidation(t *testing.T) {
	t.Parallel()
	if got := len(kindValidation(recipe.TaskInference)); got != 5 {
		t.Fatalf("inference cells=%d want 5", got)
	}
	if got := len(kindValidation(recipe.TaskImageGen)); got != 1 {
		t.Fatalf("image-gen cells=%d want 1", got)
	}
	if got := kindValidation(recipe.TaskGeneration); len(got) != 1 || got[0] != "specialized-proof" {
		t.Fatalf("generation cells=%+v want one specialized proof", got)
	}
	if got := len(kindValidation(recipe.TaskTraining)); got != 1 {
		t.Fatalf("training cells=%d want 1", got)
	}
	if kindValidation(recipe.Task("nonexistent-task")) != nil {
		t.Fatal("unknown task produced validation cells")
	}
}

// TestCompatibilityCommandIsGone holds the fold of cmd/compatibility into
// validate: the model and media reports, the media samples, the capability
// census and the model-verification records it served are validate's flags,
// and its acceptance suite runs as validate's, so no second command remains.
func TestCompatibilityCommandIsGone(t *testing.T) {
	t.Parallel()
	if _, err := os.Stat(filepath.Join("..", "compatibility")); !errors.Is(err, fs.ErrNotExist) {
		t.Fatalf("cmd/compatibility is present (%v); its reports, samples, census and verification records are validate's", err)
	}
}
