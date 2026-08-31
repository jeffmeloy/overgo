package trainingworkflow

import (
	"path/filepath"
	"strings"
	"testing"

	"overgo/internal/runrecord"
)

func TestBoundedTrainingEvidencePreservesFacts(t *testing.T) {
	stubExecutableCodeCommit(t, strings.Repeat("ef", 20))
	root, model, dataset, store, recipeID := moeWorkflowFixture(t)
	result, err := Execute(t.Context(), Request{
		Repository: store, Observations: store, Recipe: recipeID,
		ModelDirectory: model, DatasetPath: dataset,
		OutputDirectory: filepath.Join(root, "bounded-evidence"), Steps: 2, Host: true,
	})
	if err != nil {
		t.Fatal(err)
	}
	coverage, chunk, err := runrecord.RequireMoERouterObservationCoverage(
		t.Context(), store, result.RouterObservationCoverage,
	)
	if err != nil {
		t.Fatal(err)
	}
	if len(result.RouterObservations) != 1 || result.RouterObservations[0] != chunk.ID ||
		coverage.ObservationCount != uint64(len(chunk.Observations)) || len(chunk.Observations) != 2 {
		t.Fatalf("bounded router evidence = result=%v coverage=%+v observations=%d", result.RouterObservations, coverage, len(chunk.Observations))
	}
	for index, observation := range chunk.Observations {
		if observation.Step != uint64(index) || observation.Run != coverage.Run ||
			observation.Dataset != result.Checkpoint.Dataset || observation.Split != result.Checkpoint.Split ||
			observation.Checkpoint != result.Checkpoint.ID() || observation.Policy != result.Checkpoint.RunPlan {
			t.Fatalf("router observation %d lost an exact fact: %+v", index, observation)
		}
		if found, err := store.HasContent(t.Context(), observation.ID); err != nil || found {
			t.Fatalf("per-step router document retained: found=%t err=%v", found, err)
		}
	}
}

func TestTrainingEvidenceJournalAmplificationBound(t *testing.T) {
	stubExecutableCodeCommit(t, strings.Repeat("fa", 20))
	root, model, dataset, store, recipeID := moeWorkflowFixture(t)
	request := Request{
		Repository: store, Observations: store, Recipe: recipeID,
		ModelDirectory: model, DatasetPath: dataset, Steps: 1, Host: true,
		OutputDirectory: filepath.Join(root, "warm-evidence"),
	}
	if _, err := Execute(t.Context(), request); err != nil {
		t.Fatal(err)
	}
	_, before := store.Head()
	request.Steps = 2
	request.OutputDirectory = filepath.Join(root, "measured-evidence")
	result, err := Execute(t.Context(), request)
	if err != nil {
		t.Fatal(err)
	}
	_, after := store.Head()
	if after != before+1 {
		t.Fatalf("two-step evidence advanced journal by %d commits, want one atomic publication", after-before)
	}
	coverage, chunk, err := runrecord.RequireMoERouterObservationCoverage(t.Context(), store, result.RouterObservationCoverage)
	if err != nil || coverage.ObservationCount != 2 || len(chunk.Observations) != 2 {
		t.Fatalf("amplification evidence = coverage=%+v observations=%d err=%v", coverage, len(chunk.Observations), err)
	}
}
