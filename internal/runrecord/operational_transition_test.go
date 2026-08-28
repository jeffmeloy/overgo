package runrecord

import (
	"context"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/artifact/repositorytest"
	"overgo/internal/overgodb"
	"overgo/internal/testutil"
)

// TestOperationalTransitionPublication holds the conversion door to
// its contract: a typed transition publishes as one atomic batch with
// subject, authority, and evidence lineage; unknown kinds and unbound
// subjects refuse; and a raw coordination-shaped schema refuses at
// the store boundary itself.
func TestOperationalTransitionPublication(t *testing.T) {
	ctx := context.Background()
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = store.Close() })
	subject := testutil.ArtifactBytesID(t, artifact.KindRun, []byte("transition-subject"))
	authority := testutil.ArtifactBytesID(t, artifact.KindEvidence, []byte("transition-authority"))
	if _, err := store.Commit(ctx, artifact.Batch{
		Key:       "transition/authorities",
		Artifacts: []artifact.Descriptor{{ID: subject}, {ID: authority}},
	}); err != nil {
		t.Fatal(err)
	}
	counting := &repositorytest.CountingRepository{Repository: store}
	transition, err := PublishOperationalTransition(ctx, counting, OperationalTransition{
		Kind: TransitionRecovery, Subject: subject, Authority: authority,
		Note: "runtime reconnected before the reconnect grace expired",
	})
	if err != nil {
		t.Fatal(err)
	}
	if counting.Commits != 1 {
		t.Fatalf("transition publication used %d commits, want 1", counting.Commits)
	}
	if len(transition.Lineage()) != 2 {
		t.Fatalf("transition lineage = %+v", transition.Lineage())
	}

	if _, err := PublishOperationalTransition(ctx, store, OperationalTransition{
		Kind: "chatter", Subject: subject, Authority: authority, Note: "noise",
	}); err == nil {
		t.Fatal("unknown transition kind admitted")
	}

	signal := artifact.Descriptor{
		ID:        testutil.ArtifactBytesID(t, artifact.KindEvidence, []byte("heartbeat-signal")),
		Schema:    "overgo/agent-heartbeat/v1",
		MediaType: "application/json",
	}
	if _, err := store.Commit(ctx, artifact.Batch{
		Key: "transition/raw-signal", Artifacts: []artifact.Descriptor{signal},
	}); err == nil || !strings.Contains(err.Error(), "operational signal") {
		t.Fatalf("raw heartbeat advanced the head: %v", err)
	}
}
