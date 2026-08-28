package workflowruntime

import (
	"context"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/artifact/repositorytest"
	"overgo/internal/overgodb"
	"overgo/internal/runrecord"
	"overgo/internal/testutil"
)

// TestOperationalTransitionPublication proves this surface converts
// operational state through the one shared door: a TransitionRecovery event
// publishes exactly one canonical transition batch, never a raw
// signal.
func TestOperationalTransitionPublication(t *testing.T) {
	ctx := context.Background()
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = store.Close() })
	subject := testutil.ArtifactBytesID(t, artifact.KindRecipe, []byte("workflowruntime-transition-subject"))
	authority := testutil.ArtifactBytesID(t, artifact.KindEvidence, []byte("workflowruntime-transition-authority"))
	if _, err := store.Commit(ctx, artifact.Batch{
		Key:       "workflowruntime/transition-authorities",
		Artifacts: []artifact.Descriptor{{ID: subject}, {ID: authority}},
	}); err != nil {
		t.Fatal(err)
	}
	counting := &repositorytest.CountingRepository{Repository: store}
	if _, err := runrecord.PublishOperationalTransition(ctx, counting, runrecord.OperationalTransition{
		Kind: runrecord.TransitionRecovery, Subject: subject, Authority: authority,
		Note: "stage recovered from an interrupted execution",
	}); err != nil {
		t.Fatal(err)
	}
	if counting.Commits != 1 {
		t.Fatalf("transition publication used %d commits, want 1", counting.Commits)
	}
}
