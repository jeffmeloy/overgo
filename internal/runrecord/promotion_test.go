package runrecord_test

import (
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/modelrecipe"
	"overgo/internal/modelrecipetest"
	"overgo/internal/overgodbtest"
	"overgo/internal/recipe"
	"overgo/internal/runrecord"
	"overgo/internal/testcheck"
	"overgo/internal/testutil"
)

func TestEvaluationPromotesCapabilityRecipe(t *testing.T) {
	ctx := t.Context()
	store := overgodbtest.Open(t, t.TempDir())
	modelID := testutil.ArtifactID(t, artifact.KindModel, "model")
	datasetID := testutil.ArtifactID(t, artifact.KindDataset, "dataset")
	if _, err := store.Commit(ctx, artifact.Batch{Key: "fixture/workflow/facts", Artifacts: []artifact.Descriptor{
		{ID: modelID}, {ID: datasetID},
	}}); err != nil {
		t.Fatal(err)
	}
	definition := testcheck.Must(modelrecipe.CapabilityDefinition(recipe.TaskForecast, modelID))(t)
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
	verification := testcheck.Must(modelrecipetest.PublishVerification(
		ctx, store, "fixture/workflow/verification", definition.ID,
	))(t)
	evaluation := testcheck.Must(runrecord.NewEvaluation(
		definition.ID, verification.Run, datasetID,
		[]runrecord.Metric{{Name: "quality", Value: 1, Direction: runrecord.DirectionMaximize}},
	))(t)
	batch := testcheck.Must(evaluation.Batch("fixture/workflow/evaluation"))(t)
	if _, err := store.Commit(ctx, batch); err != nil {
		t.Fatal(err)
	}
	if _, _, err := modelrecipe.ActivateVerified(
		ctx, store, "fixture/workflow/active", definition, verification,
		recipe.EvidenceVerified, "workflow fixture activation", []artifact.ID{evaluation.ID}, nil,
	); err != nil {
		t.Fatal(err)
	}
	active, ok, err := modelrecipe.ActiveRecord(ctx, store, modelID, recipe.TaskForecast)
	if err != nil || !ok || active.Definition.ID != definition.ID {
		t.Fatalf("active workflow = (%s, %v, %v)", active.Definition.ID, ok, err)
	}
}
