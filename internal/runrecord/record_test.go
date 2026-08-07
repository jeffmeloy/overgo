package runrecord

import (
	"context"
	"math"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/repodb"
)

func TestRunAndEvaluationRoundTrip(t *testing.T) {
	recipeID := fixtureID(t, artifact.KindRecipe, "recipe")
	datasetID := fixtureID(t, artifact.KindDataset, "dataset")
	inputID := fixtureID(t, artifact.KindFile, "input")
	outputID := fixtureID(t, artifact.KindOutput, "output")
	run, err := NewRun(recipeID, OutcomeSucceeded, []artifact.ID{inputID}, []artifact.ID{outputID}, "")
	if err != nil {
		t.Fatal(err)
	}
	content, err := run.ContentBytes()
	if err != nil {
		t.Fatal(err)
	}
	parsedRun, err := ParseRun(content)
	if err != nil || parsedRun.ID != run.ID {
		t.Fatalf("run round trip = (%+v, %v)", parsedRun, err)
	}
	evaluation, err := NewEvaluation(recipeID, run.ID, datasetID, []Metric{
		{Name: "latency", Value: 12.5, Unit: "ms", Direction: DirectionMinimize},
		{Name: "quality", Value: 0.9, Direction: DirectionMaximize},
	})
	if err != nil {
		t.Fatal(err)
	}
	evaluationContent, err := evaluation.ContentBytes()
	if err != nil {
		t.Fatal(err)
	}
	parsedEvaluation, err := ParseEvaluation(evaluationContent)
	if err != nil || parsedEvaluation.ID != evaluation.ID || len(evaluation.Lineage()) != 3 {
		t.Fatalf("evaluation round trip = (%+v, %v)", parsedEvaluation, err)
	}
	if len(run.Lineage()) != 3 {
		t.Fatalf("run lineage = %+v", run.Lineage())
	}
}

func TestRunAndEvaluationPersistWithLineage(t *testing.T) {
	ctx := context.Background()
	store, err := repodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	recipeID := fixtureID(t, artifact.KindRecipe, "recipe")
	datasetID := fixtureID(t, artifact.KindDataset, "dataset")
	inputID := fixtureID(t, artifact.KindFile, "input")
	outputID := fixtureID(t, artifact.KindOutput, "output")
	if _, err := store.Commit(ctx, artifact.Batch{Key: "fixture/facts", Artifacts: []artifact.Descriptor{
		{ID: recipeID}, {ID: datasetID}, {ID: inputID}, {ID: outputID},
	}}); err != nil {
		t.Fatal(err)
	}
	run, err := NewRun(recipeID, OutcomeSucceeded, []artifact.ID{inputID}, []artifact.ID{outputID}, "")
	if err != nil {
		t.Fatal(err)
	}
	batch, err := run.Batch("fixture/run")
	if err != nil {
		t.Fatal(err)
	}
	if _, err := store.Commit(ctx, batch); err != nil {
		t.Fatal(err)
	}
	evaluation, err := NewEvaluation(recipeID, run.ID, datasetID, []Metric{{
		Name: "quality", Value: 1, Direction: DirectionMaximize,
	}})
	if err != nil {
		t.Fatal(err)
	}
	batch, err = evaluation.Batch("fixture/evaluation")
	if err != nil {
		t.Fatal(err)
	}
	if _, err := store.Commit(ctx, batch); err != nil {
		t.Fatal(err)
	}
	parents, err := store.Parents(ctx, evaluation.ID)
	if err != nil || len(parents) != 3 {
		t.Fatalf("evaluation parents = (%+v, %v)", parents, err)
	}
}

func TestRunAndEvaluationRejectInvalidFacts(t *testing.T) {
	recipeID := fixtureID(t, artifact.KindRecipe, "recipe")
	datasetID := fixtureID(t, artifact.KindDataset, "dataset")
	runID := fixtureID(t, artifact.KindRun, "run")
	if _, err := NewRun(recipeID, OutcomeSucceeded, nil, nil, ""); err == nil {
		t.Fatal("successful run without output accepted")
	}
	if _, err := NewRun(recipeID, OutcomeFailed, nil, nil, ""); err == nil {
		t.Fatal("failed run without failure code accepted")
	}
	if _, err := NewEvaluation(recipeID, runID, datasetID, []Metric{{
		Name: "quality", Value: math.NaN(), Direction: DirectionMaximize,
	}}); err == nil {
		t.Fatal("non-finite metric accepted")
	}
}

func fixtureID(t *testing.T, kind artifact.Kind, value string) artifact.ID {
	t.Helper()
	id, err := artifact.IdentifyBytes(kind, []byte(value))
	if err != nil {
		t.Fatal(err)
	}
	return id
}
