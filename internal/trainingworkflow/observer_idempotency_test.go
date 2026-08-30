package trainingworkflow

import (
	"context"
	"path/filepath"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/overgodb"
)

// TestObserverRepeatsEnvironmentCommit pins rerun idempotency: two
// sessions on the same machine derive the identical environment record,
// and the second session's observation must publish against the stored
// environment instead of refusing on a no-op batch.
func TestObserverRepeatsEnvironmentCommit(t *testing.T) {
	store, err := overgodb.Open(filepath.Join(t.TempDir(), "repodb"))
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	model, err := artifact.IdentifyBytes(artifact.KindModel, []byte("model"))
	if err != nil {
		t.Fatal(err)
	}
	recipeID, err := artifact.IdentifyBytes(artifact.KindRecipe, []byte("recipe"))
	if err != nil {
		t.Fatal(err)
	}
	ctx := context.Background()
	if _, err := store.Commit(ctx, artifact.Batch{
		Key:       "test/observer-idempotency/authority",
		Artifacts: []artifact.Descriptor{{ID: model}, {ID: recipeID}},
	}); err != nil {
		t.Fatal(err)
	}
	for session := 0; session < 2; session++ {
		observer, err := NewObserver(store, true)
		if err != nil {
			t.Fatal(err)
		}
		if _, err := observer.Finish(ctx, model, recipeID, nil, 1); err != nil {
			t.Fatalf("session %d observation: %v", session, err)
		}
	}
}
