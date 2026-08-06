package runrecord_test

import (
	"context"
	"testing"

	"llamacpp2go/internal/artifact"
	"llamacpp2go/internal/modelrecipe"
	"llamacpp2go/internal/recipe"
	"llamacpp2go/internal/repodb"
	"llamacpp2go/internal/runrecord"
	"llamacpp2go/internal/workflowrecipe"
)

func TestEvaluationPromotesWorkflowRecipe(t *testing.T) {
	ctx := context.Background()
	store, err := repodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	modelID := testID(t, artifact.KindModel, "model")
	tokenizerID := testID(t, artifact.KindTokenizer, "tokenizer")
	datasetID := testID(t, artifact.KindDataset, "dataset")
	outputID := testID(t, artifact.KindOutput, "output")
	if _, err := store.Commit(ctx, artifact.Batch{Key: "fixture/workflow/facts", Artifacts: []artifact.Descriptor{
		{ID: modelID}, {ID: tokenizerID}, {ID: datasetID}, {ID: outputID},
	}}); err != nil {
		t.Fatal(err)
	}
	definition, err := workflowrecipe.Generation(workflowrecipe.Bindings{
		Model: modelID, Tokenizer: tokenizerID,
	}, recipe.PlacementHost)
	if err != nil {
		t.Fatal(err)
	}
	if _, _, err := modelrecipe.PublishCandidate(
		ctx, store, "fixture/workflow/candidate", definition,
	); err != nil {
		t.Fatal(err)
	}
	if _, _, err := modelrecipe.Transition(
		ctx, store, "fixture/workflow/validated", definition, recipe.StatusValidated, nil, nil,
	); err != nil {
		t.Fatal(err)
	}
	run, err := runrecord.NewRun(
		definition.ID, runrecord.OutcomeSucceeded, nil, []artifact.ID{outputID}, "",
	)
	if err != nil {
		t.Fatal(err)
	}
	batch, err := run.Batch("fixture/workflow/run")
	if err != nil {
		t.Fatal(err)
	}
	if _, err := store.Commit(ctx, batch); err != nil {
		t.Fatal(err)
	}
	evaluation, err := runrecord.NewEvaluation(
		definition.ID, run.ID, datasetID,
		[]runrecord.Metric{{Name: "quality", Value: 1, Direction: runrecord.DirectionMaximize}},
	)
	if err != nil {
		t.Fatal(err)
	}
	batch, err = evaluation.Batch("fixture/workflow/evaluation")
	if err != nil {
		t.Fatal(err)
	}
	if _, err := store.Commit(ctx, batch); err != nil {
		t.Fatal(err)
	}
	if _, _, err := modelrecipe.Transition(
		ctx, store, "fixture/workflow/active", definition, recipe.StatusActive,
		[]artifact.ID{evaluation.ID}, nil,
	); err != nil {
		t.Fatal(err)
	}
	active, ok, err := modelrecipe.Active(ctx, store, modelID, recipe.TaskGeneration)
	if err != nil || !ok || active.ID != definition.ID {
		t.Fatalf("active workflow = (%s, %v, %v)", active.ID, ok, err)
	}
}

func testID(t *testing.T, kind artifact.Kind, value string) artifact.ID {
	t.Helper()
	id, err := artifact.IdentifyBytes(kind, []byte(value))
	if err != nil {
		t.Fatal(err)
	}
	return id
}
