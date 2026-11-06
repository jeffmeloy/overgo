package trainingworkflow

import (
	"path/filepath"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/overgodbtest"
	"overgo/internal/testcheck"
)

// TestObserverRepeatsEnvironmentCommit pins rerun idempotency: two
// sessions on the same machine derive the identical environment record,
// and the second session's observation must publish against the stored
// environment instead of refusing on a no-op batch.
func TestObserverRepeatsEnvironmentCommit(t *testing.T) {
	store := overgodbtest.Create(t, filepath.Join(t.TempDir(), "repodb"))
	model := testcheck.Must(artifact.IdentifyBytes(artifact.KindModel, []byte("model")))(t)
	recipeID := testcheck.Must(artifact.IdentifyBytes(artifact.KindRecipe, []byte("recipe")))(t)
	ctx := t.Context()
	if _, err := store.Commit(ctx, artifact.Batch{
		Key:       "test/observer-idempotency/authority",
		Artifacts: []artifact.Descriptor{{ID: model}, {ID: recipeID}},
	}); err != nil {
		t.Fatal(err)
	}
	for session := range 2 {
		observer := testcheck.Must(NewObserver(store, true))(t)
		if _, err := observer.Finish(ctx, model, recipeID, nil, 1); err != nil {
			t.Fatalf("session %d observation: %v", session, err)
		}
	}
}
