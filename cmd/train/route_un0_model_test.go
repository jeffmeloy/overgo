//go:build modeltest

package main

import (
	"os"
	"overgo/internal/artifact"
	"overgo/internal/dataroot"
	"overgo/internal/testskip"
	"overgo/internal/testutil"
	"overgo/internal/trainingprogram"
	"overgo/internal/trainingworkflow"
	"path/filepath"
	"testing"
)

// TestUn0RoutePublishesCheckpoint trains the real Un-0 checkpoint's scratch bootstrap through the
// oscillatorimage route and holds it to descending and publishing a
// checkpoint that trained from the bootstrap, not from Un-0's weights.
func TestUn0RoutePublishesCheckpoint(t *testing.T) {
	testskip.Short(t, "trains the real Un-0 checkpoint")
	repository := testutil.RepoRoot(t)
	roots, err := dataroot.Resolve(repository)
	if err != nil {
		t.Fatal(err)
	}
	model := roots.ResolveModelPath("Un-0")
	if _, err := os.Stat(filepath.Join(model, "config.json")); err != nil {
		testskip.NotApplicable(t, "the Un-0 checkpoint is absent at "+model)
	}
	out := filepath.Join(t.TempDir(), "checkpoint")
	if err := runRoute(t.Context(), []string{"oscillatorimage", "-model", model, "-steps", "2",
		"-stimulus", filepath.Join(repository, "docs", "verification", "un0-training-dataset.txt"),
		"-store", filepath.Join(t.TempDir(), "store"), "-out", out}); err != nil {
		t.Fatalf("oscillatorimage route on Un-0: %v", err)
	}
	checkpoint, err := trainingprogram.LoadCheckpoint(out)
	if err != nil {
		t.Fatalf("the trained Un-0 bootstrap's checkpoint: %v", err)
	}
	weights, err := trainingworkflow.IdentifyModelWeights(model)
	if err != nil {
		t.Fatal(err)
	}
	file, err := os.Open(weights)
	if err != nil {
		t.Fatal(err)
	}
	un0, _, err := artifact.Identify(artifact.KindModel, file)
	if closeErr := file.Close(); err != nil || closeErr != nil || checkpoint.Model == un0 {
		t.Fatalf("the trained bootstrap names %s as trained from (%v, %v), want its scratch construction, not Un-0's weights %s", checkpoint.Model, err, closeErr, un0)
	}
}
