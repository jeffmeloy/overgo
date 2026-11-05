package main

import (
	"os"
	"path/filepath"
	"testing"
	"time"

	"overgo/internal/artifact"
	"overgo/internal/clioptions"
	"overgo/internal/jsonfile"
	"overgo/internal/overgodb"
	"overgo/internal/plan"
	"overgo/internal/recipe"
	"overgo/internal/testcheck"
	"overgo/internal/testutil"
)

// TestARunHashesEachModelFileOnce holds a measurement run to reading a model's bytes once: the first phase to bind a
// cell hashes its files and keeps the digests in the run's record, and a later phase binds the cell from them without
// reading the files, so bytes changed under it are not seen. A run begun again is a record of its own, so it hashes them
// again and sees the change. A file that could not be hashed keeps no digest.
func TestARunHashesEachModelFileOnce(t *testing.T) {
	// Serial: it names the measurement run's record in the process environment.
	store, err := overgodb.Create(filepath.Join(t.TempDir(), "overgodb-store"))
	testcheck.NoError(t, err)
	defer store.Close()
	weights := filepath.Join(t.TempDir(), "model.gguf")
	testcheck.NoError(t, os.WriteFile(weights, []byte("weights"), 0o600))
	model := creditModel(t, store, weights)
	bound := func() string {
		t.Helper()
		cells := []ModelValidation{{
			Model: model, ModelName: "hashed", Kind: recipe.TaskInference, Validation: guardValidation,
			Recipe: testutil.ArtifactID(t, artifact.KindRecipe, "hashed"), Location: weights,
		}}
		testcheck.NoError(t, bindCells(t.Context(), testutil.RepoRoot(t), store, cells, "fixture-execution"))
		return cells[0].Consumed
	}
	record := filepath.Join(t.TempDir(), "measurement_run.json")
	begin := func() {
		t.Helper()
		started := time.Now()
		if err := jsonfile.Write(record, plan.MeasurementRun{Commit: "measured", Process: os.Getpid(), Started: started,
			Phases: []plan.MeasurementPhase{{Name: "preflight", Started: started}}}, clioptions.PrivateFileMode); err != nil {
			t.Fatal(err)
		}
	}
	t.Setenv(clioptions.MeasurementRunEnvironment, record)

	begin()
	first := bound()
	var run plan.MeasurementRun
	testcheck.NoError(t, jsonfile.Decode(record, &run))
	if len(run.Digests) == 0 {
		t.Fatal("the first phase to bind the cell kept no digest in the run's record")
	}
	testcheck.NoError(t, os.WriteFile(weights, []byte("changed"), 0o600))
	if later := bound(); later != first {
		t.Fatal("a later phase of the run read the model's file again")
	}
	begin()
	if again := bound(); again == first {
		t.Fatal("a run begun again bound the cell from the run before's digests")
	}
	keepDigests(map[string]string{"unhashed.gguf": "not a digest"})
	testcheck.NoError(t, jsonfile.Decode(record, &run))
	if _, kept := run.Digests["unhashed.gguf"]; kept {
		t.Fatal("a file that could not be hashed kept a digest")
	}
}
