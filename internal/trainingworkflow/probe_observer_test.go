package trainingworkflow

import (
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/testutil"
)

func TestProbeObserverExposesSessionLifecycle(t *testing.T) {
	observer, err := NewProbeObserver(nil, true)
	if err != nil {
		t.Fatal(err)
	}
	model := testutil.ArtifactID(t, artifact.KindModel, "training-model")
	recipeID := testutil.ArtifactID(t, artifact.KindRecipe, "training-recipe")
	if err := observer.Admit(t.Context(), model, recipeID); err != nil {
		t.Fatal(err)
	}
	snapshot := observer.SessionSnapshot()
	if snapshot.Name != "training" || snapshot.Active != 1 || snapshot.Entries != 1 || snapshot.Loads != 1 {
		t.Fatalf("admitted training session = %+v", snapshot)
	}
	if err := observer.lease.Release(); err != nil {
		t.Fatal(err)
	}
	snapshot = observer.SessionSnapshot()
	if snapshot.Active != 0 || snapshot.Entries != 0 || snapshot.Retirements != 1 {
		t.Fatalf("released training session = %+v", snapshot)
	}
	if err := observer.director.Close(t.Context()); err != nil {
		t.Fatal(err)
	}
}
