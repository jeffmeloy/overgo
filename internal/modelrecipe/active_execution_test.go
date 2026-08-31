package modelrecipe

import (
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/overgodb"
	"overgo/internal/recipe"
	"overgo/internal/testutil"
)

// TestResolveActiveExecutionServesInference pins the smoke-matrix
// regression of 2026-08-24: resolving a model's active inference
// execution must succeed — inference executes through the compiled
// runner, but its ordered program still resolves through the module
// catalog, because selection needs the program for its definition,
// resource, and policy bindings. Before the fix every recipe-authorized
// serve refused with "unsupported runtime task inference".
func TestResolveActiveExecutionServesInference(t *testing.T) {
	ctx := t.Context()
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	modelID := testutil.ArtifactID(t, artifact.KindModel, "active-execution-model")
	if _, err := store.Commit(ctx, artifact.Batch{
		Key:       "fixture/active-execution/model",
		Artifacts: []artifact.Descriptor{{ID: modelID, Size: 4096}},
	}); err != nil {
		t.Fatal(err)
	}
	definition, err := inferenceFixture(modelID, recipe.PlacementHost, DecodeSessionRequest)
	if err != nil {
		t.Fatal(err)
	}
	if _, _, err := PublishCandidate(ctx, store, "fixture/active-execution/candidate", definition); err != nil {
		t.Fatal(err)
	}
	if err := ActivateCapability(
		ctx, store, definition,
		publishVerification(t, store, definition.ID, "fixture/active-execution/verification"),
		recipe.EvidenceVerified, "active execution fixture",
	); err != nil {
		t.Fatal(err)
	}
	execution, err := ResolveActiveExecution(ctx, store, modelID, recipe.TaskInference, SessionWarm)
	if err != nil {
		t.Fatalf("active inference execution refused: %v", err)
	}
	program := execution.Program
	if program.Definition().Task != recipe.TaskInference || program.Definition().ID != definition.ID {
		t.Fatalf("resolved program = %+v", program.Definition())
	}
}
