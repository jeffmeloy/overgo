package main

import (
	"bytes"
	"os"
	"path/filepath"
	"reflect"
	"strconv"
	"strings"
	"testing"

	cudatest "overgo/internal/cuda/testutil"
	"overgo/internal/dataroot"
	"overgo/internal/testcheck"
	"overgo/internal/testskip"
	"overgo/internal/testutil"
	"overgo/internal/trainingprogram"
)

// TestRouteAdapterResumesItsCheckpoint trains Gemma E4B's layer-0 adapter through the adapter route for two updates
// at once, and for one that a second run resumes to two: the resumed run's checkpoint holds the weights the
// uninterrupted run's does, byte for byte, and its optimizer state and stream. A checkpoint that took every update
// asked for is refused.
func TestRouteAdapterResumesItsCheckpoint(t *testing.T) {
	cudatest.Require(t)
	roots := testcheck.Must(dataroot.Resolve(testutil.RepoRoot(t)))(t)
	model := filepath.Join(roots.Checkpoints, "overgo-hfconvert", "gemma-4-E4B-it-bf16.gguf")
	if _, err := os.Stat(model); err != nil {
		testskip.NotApplicable(t, "Gemma E4B's converted GGUF is absent: "+model)
	}
	scratch := t.TempDir()
	dataset := filepath.Join(scratch, "text.txt")
	testcheck.NoError(t, os.WriteFile(dataset, []byte("The adapter learns one sentence, and then it learns it again."), 0o600))
	store := t.TempDir()
	train := func(out string, steps int, more ...string) error {
		return runRoute(t.Context(), append([]string{"adapter", "-model", model, "-dataset", dataset, "-layer", "0",
			"-steps", strconv.Itoa(steps), "-store", store, "-out", out}, more...))
	}
	whole, part, resumed := filepath.Join(scratch, "whole"), filepath.Join(scratch, "part"), filepath.Join(scratch, "resumed")
	for _, run := range []struct {
		out   string
		steps int
		more  []string
	}{{whole, 2, nil}, {part, 1, nil}, {resumed, 2, []string{"-resume", part}}} {
		testcheck.NoError(t, train(run.out, run.steps, run.more...))
	}
	wholeWeights := testcheck.Must(os.ReadFile(filepath.Join(whole, trainingprogram.CheckpointWeights)))(t)
	resumedWeights := testcheck.Must(os.ReadFile(filepath.Join(resumed, trainingprogram.CheckpointWeights)))(t)
	if !bytes.Equal(wholeWeights, resumedWeights) {
		t.Fatal("the resumed run's weights differ from the uninterrupted run's")
	}
	wholeCheckpoint := testcheck.Must(trainingprogram.LoadCheckpoint(whole))(t)
	resumedCheckpoint := testcheck.Must(trainingprogram.LoadCheckpoint(resumed))(t)
	if !reflect.DeepEqual(wholeCheckpoint.Optimizer, resumedCheckpoint.Optimizer) || wholeCheckpoint.Stream != resumedCheckpoint.Stream {
		t.Fatal("the resumed run's optimizer state or stream differs from the uninterrupted run's")
	}
	if err := train(filepath.Join(scratch, "again"), 2, "-resume", whole); err == nil || !strings.Contains(err.Error(), "took 2 updates") {
		t.Fatalf("resuming a checkpoint that took every update asked for = %v, want a refusal", err)
	}
}
