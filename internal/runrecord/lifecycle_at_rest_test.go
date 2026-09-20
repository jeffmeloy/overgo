package runrecord

import (
	"strings"
	"testing"
	"time"

	"overgo/internal/artifact"
	"overgo/internal/overgodb"
	"overgo/internal/testutil"
)

func TestGateLifecycleAtRestAdmission(t *testing.T) {
	ctx := t.Context()
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	environment := testutil.ArtifactID(t, artifact.KindEvidence, "at-rest-environment")
	result := testutil.ArtifactID(t, artifact.KindEvidence, "at-rest-result")
	testutil.PublishArtifact(t, store, environment)
	prepared, err := NewGatePreparation(
		"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
		environment, time.Unix(100, 0),
	)
	if err != nil {
		t.Fatal(err)
	}
	preparedContent, err := prepared.Content()
	if err != nil {
		t.Fatal(err)
	}
	if _, err := store.Commit(ctx, artifact.Batch{
		Key: "test/at-rest/prepared", Contents: []artifact.Content{preparedContent},
		Lineage: prepared.Lineage(),
		Aliases: []artifact.AliasBinding{{Name: GateLifecycleCurrentAlias, Target: prepared.ID}},
	}); err != nil {
		t.Fatal(err)
	}

	admit := GateLifecycleAtRest(store)
	if err := admit(overgodb.StoreLocalAliasPrefix+"unknown/current", prepared.ID); err == nil ||
		!strings.Contains(err.Error(), "unknown store-local authority") {
		t.Fatalf("unknown authority error = %v", err)
	}
	if err := admit(GateLifecycleCurrentAlias, prepared.ID); err == nil ||
		!strings.Contains(err.Error(), "not at rest") {
		t.Fatalf("prepared admission error = %v", err)
	}

	testutil.PublishArtifact(t, store, result)
	finalized, err := NewGateFinalization(
		prepared, "0123456789abcdef0123456789abcdef01234567", result, OutcomeSucceeded,
	)
	if err != nil {
		t.Fatal(err)
	}
	finalizedContent, err := finalized.Content()
	if err != nil {
		t.Fatal(err)
	}
	if _, err := store.Commit(ctx, artifact.Batch{
		Key: "test/at-rest/finalized", Contents: []artifact.Content{finalizedContent},
		Lineage: finalized.Lineage(),
		Aliases: []artifact.AliasBinding{{
			Name: GateLifecycleCurrentAlias, Target: finalized.ID, Previous: artifact.IDPointer(prepared.ID),
		}},
	}); err != nil {
		t.Fatal(err)
	}
	if err := admit(GateLifecycleCurrentAlias, finalized.ID); err != nil {
		t.Fatalf("finalized admission refused: %v", err)
	}
}

// TestGateLaneObligationAtRestAdmission proves the deferred lane obligation
// crosses a store rewrite in every state but running: a pending or resolved
// obligation is at rest, a running one is a lane runner still writing receipts.
func TestGateLaneObligationAtRestAdmission(t *testing.T) {
	ctx := t.Context()
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	preparation := testutil.ArtifactID(t, artifact.KindEvidence, "lane-at-rest-preparation")
	result := testutil.ArtifactID(t, artifact.KindEvidence, "lane-at-rest-result")
	outcome := testutil.ArtifactID(t, artifact.KindEvidence, "lane-at-rest-outcome")
	now := time.Unix(200, 0)
	pending, err := NewGateLaneObligation(
		"0123456789abcdef0123456789abcdef01234567", preparation, result,
		[]string{"device", "webui-lane"}, []string{"internal/gate/run.go"}, now,
	)
	if err != nil {
		t.Fatal(err)
	}
	commit := func(obligation GateLaneObligation, previous *artifact.ID) {
		t.Helper()
		batch, err := obligation.Batch(previous)
		if err != nil {
			t.Fatal(err)
		}
		if _, err := store.Commit(ctx, batch); err != nil {
			t.Fatal(err)
		}
	}
	commit(pending, nil)
	admit := GateLifecycleAtRest(store)
	if err := admit(GateLaneObligationAlias, pending.ID); err != nil {
		t.Fatalf("pending obligation refused: %v", err)
	}
	running, err := pending.Transition(LaneObligationRunning, artifact.ID{}, now.Add(time.Second))
	if err != nil {
		t.Fatal(err)
	}
	commit(running, artifact.IDPointer(pending.ID))
	if err := admit(GateLaneObligationAlias, running.ID); err == nil ||
		!strings.Contains(err.Error(), "not at rest") {
		t.Fatalf("running obligation admission error = %v", err)
	}
	passed, err := running.Transition(LaneObligationPassed, outcome, now.Add(2*time.Second))
	if err != nil {
		t.Fatal(err)
	}
	commit(passed, artifact.IDPointer(running.ID))
	if err := admit(GateLaneObligationAlias, passed.ID); err != nil {
		t.Fatalf("passed obligation refused: %v", err)
	}
}
