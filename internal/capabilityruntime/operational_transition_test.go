package capabilityruntime

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
// operational state through the one shared door: a TransitionCapabilityChange event
// publishes exactly one canonical transition batch, never a raw
// signal.
func TestOperationalTransitionPublication(t *testing.T) {
	ctx := context.Background()
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = store.Close() })
	subject := testutil.ArtifactBytesID(t, artifact.KindProfile, []byte("capabilityruntime-transition-subject"))
	authority := testutil.ArtifactBytesID(t, artifact.KindEvidence, []byte("capabilityruntime-transition-authority"))
	if _, err := store.Commit(ctx, artifact.Batch{
		Key:       "capabilityruntime/transition-authorities",
		Artifacts: []artifact.Descriptor{{ID: subject}, {ID: authority}},
	}); err != nil {
		t.Fatal(err)
	}
	counting := &repositorytest.CountingRepository{Repository: store}
	if _, err := runrecord.PublishOperationalTransition(ctx, counting, runrecord.OperationalTransition{
		Kind: runrecord.TransitionCapabilityChange, Subject: subject, Authority: authority,
		Note: "peer capability catalog changed under refresh",
	}); err != nil {
		t.Fatal(err)
	}
	if counting.Commits != 1 {
		t.Fatalf("transition publication used %d commits, want 1", counting.Commits)
	}
}
