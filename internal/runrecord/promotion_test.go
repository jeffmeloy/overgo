package runrecord_test

import (
	"context"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/modelrecipe"
	"overgo/internal/modelrecipetest"
	"overgo/internal/recipe"
	"overgo/internal/repodb"
	"overgo/internal/runrecord"
	"overgo/internal/testutil"
	"overgo/internal/workflowrecipe"
)

func TestEvaluationPromotesWorkflowRecipe(t *testing.T) {
	ctx := context.Background()
	store, err := repodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	modelID := testutil.ArtifactID(t, artifact.KindModel, "model")
	tokenizerID := testutil.ArtifactID(t, artifact.KindTokenizer, "tokenizer")
	datasetID := testutil.ArtifactID(t, artifact.KindDataset, "dataset")
	if _, err := store.Commit(ctx, artifact.Batch{Key: "fixture/workflow/facts", Artifacts: []artifact.Descriptor{
		{ID: modelID}, {ID: tokenizerID}, {ID: datasetID},
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
	verification, err := modelrecipetest.PublishVerification(
		ctx, store, "fixture/workflow/verification", definition.ID,
	)
	if err != nil {
		t.Fatal(err)
	}
	evaluation, err := runrecord.NewEvaluation(
		definition.ID, verification.Run, datasetID,
		[]runrecord.Metric{{Name: "quality", Value: 1, Direction: runrecord.DirectionMaximize}},
	)
	if err != nil {
		t.Fatal(err)
	}
	batch, err := evaluation.Batch("fixture/workflow/evaluation")
	if err != nil {
		t.Fatal(err)
	}
	if _, err := store.Commit(ctx, batch); err != nil {
		t.Fatal(err)
	}
	if _, _, err := modelrecipe.ActivateVerified(
		ctx, store, "fixture/workflow/active", definition, verification,
		[]artifact.ID{evaluation.ID}, nil,
	); err != nil {
		t.Fatal(err)
	}
	active, ok, err := modelrecipe.Active(ctx, store, modelID, recipe.TaskGeneration)
	if err != nil || !ok || active.ID != definition.ID {
		t.Fatalf("active workflow = (%s, %v, %v)", active.ID, ok, err)
	}
}
