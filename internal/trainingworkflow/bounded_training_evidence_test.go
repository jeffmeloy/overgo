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
	// A run commits its checkpoint record, its session evidence and its
	// verification whatever its steps: a step adds no commit.
	commits := func(steps int, output string) (Result, uint64) {
		t.Helper()
		_, before := store.Head()
		request.Steps, request.OutputDirectory = steps, filepath.Join(root, output)
		result, err := Execute(t.Context(), request)
		if err != nil {
			t.Fatal(err)
		}
		_, after := store.Head()
		return result, after - before
	}
	_, one := commits(1, "one-step-evidence")
	result, two := commits(2, "two-step-evidence")
	if two != one {
		t.Fatalf("two-step evidence advanced the journal by %d commits, a one-step run by %d: want steps to add none", two, one)
	}
	coverage, chunk, err := runrecord.RequireMoERouterObservationCoverage(t.Context(), store, result.RouterObservationCoverage)
	if err != nil || coverage.ObservationCount != 2 || len(chunk.Observations) != 2 {
		t.Fatalf("amplification evidence = coverage=%+v observations=%d err=%v", coverage, len(chunk.Observations), err)
	}
}
