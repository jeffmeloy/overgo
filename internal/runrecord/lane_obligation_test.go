package runrecord

import (
	"strings"
	"testing"
	"time"

	"overgo/internal/artifact"
	"overgo/internal/overgodb"
	"overgo/internal/testutil"
)

// TestGateLaneObligation holds the deferred-lane record: a pending
// obligation is published under the current alias, its states form a chain
// of immutable documents, the chain admits only the declared transitions,
// and a passing gate result may carry deferred steps.
func TestGateLaneObligation(t *testing.T) {
	t.Parallel()
	preparation := testutil.ArtifactID(t, artifact.KindEvidence, "lane obligation preparation")
	result := testutil.ArtifactID(t, artifact.KindEvidence, "lane obligation gate result")
	outcome := testutil.ArtifactID(t, artifact.KindEvidence, "lane run result")
	commit := strings.Repeat("c", 40)
	now := time.Unix(1_700_000_000, 0)
	pending, err := NewGateLaneObligation(commit, preparation, result, []string{"webui-lane", "test-device", "device"}, []string{"internal/gate/run.go", "docs/plan.json"}, now)
	if err != nil {
		t.Fatal(err)
	}
	if pending.State != LaneObligationPending || pending.Checks[0] != "device" || pending.Paths[0] != "docs/plan.json" || pending.Resolved() {
		t.Fatalf("pending obligation = %+v", pending)
	}
	if _, err := NewGateLaneObligation(commit, preparation, result, nil, pending.Paths, now); err == nil {
		t.Fatal("an obligation without checks was admitted")
	}
	if _, err := pending.Transition(LaneObligationPassed, outcome, now); err == nil {
		t.Fatal("pending moved straight to passed")
	}
	running, err := pending.Transition(LaneObligationRunning, artifact.ID{}, now.Add(time.Second))
	if err != nil || running.Previous != pending.ID || running.ID == pending.ID {
		t.Fatalf("running transition = %+v %v", running, err)
	}
	if _, err := pending.Transition(LaneObligationRunning, outcome, now); err == nil {
		t.Fatal("an open state accepted an outcome")
	}
	failed, err := running.Transition(LaneObligationFailed, outcome, now.Add(2*time.Second))
	if err != nil || failed.Outcome != outcome || failed.Resolved() {
		t.Fatalf("failed transition = %+v %v", failed, err)
	}
	superseded, err := failed.Transition(LaneObligationSuperseded, outcome, now.Add(3*time.Second))
	if err != nil || !superseded.Resolved() {
		t.Fatalf("superseded transition = %+v %v", superseded, err)
	}
	if _, err := superseded.Transition(LaneObligationRunning, artifact.ID{}, now); err == nil {
		t.Fatal("a resolved obligation reopened")
	}

	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	if _, found, err := CurrentGateLaneObligation(t.Context(), store); err != nil || found {
		t.Fatalf("empty store reported an obligation: found=%v %v", found, err)
	}
	batch, err := pending.Batch(nil)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := store.Commit(t.Context(), batch); err != nil {
		t.Fatal(err)
	}
	current, found, err := CurrentGateLaneObligation(t.Context(), store)
	if err != nil || !found || current.ID != pending.ID {
		t.Fatalf("current obligation = %+v found=%v %v", current, found, err)
	}
	previous := pending.ID
	batch, err = running.Batch(&previous)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := store.Commit(t.Context(), batch); err != nil {
		t.Fatal(err)
	}
	stale, err := failed.Batch(&previous)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := store.Commit(t.Context(), stale); err == nil {
		t.Fatal("a stale CAS alias move was admitted")
	}
	current, _, err = CurrentGateLaneObligation(t.Context(), store)
	if err != nil || current.ID != running.ID || current.State != LaneObligationRunning {
		t.Fatalf("current after running = %+v %v", current, err)
	}
	read, found, err := gateLaneObligationCodec.Read(t.Context(), store, pending.ID)
	if err != nil || !found || read.State != LaneObligationPending {
		t.Fatalf("read pending = %+v found=%v %v", read, found, err)
	}

	recipe := testutil.ArtifactID(t, artifact.KindRecipe, "lane obligation recipe")
	environment := testutil.ArtifactID(t, artifact.KindEvidence, "lane obligation environment")
	steps := []GateStep{
		{Name: "build", Phase: PhaseBuild, Outcome: StepSucceeded, DurationNS: 1},
		{Name: "test-device", Phase: PhaseTest, Outcome: StepDeferred, DurationNS: 1},
		{Name: "commit", Phase: PhasePackage, Outcome: StepSucceeded, DurationNS: 1},
	}
	if _, err := NewGateRecord(recipe, environment, commit, OutcomeSucceeded, "", 3, steps); err != nil {
		t.Fatalf("a successful gate refused a deferred step: %v", err)
	}
	steps[1].Outcome = StepFailed
	if _, err := NewGateRecord(recipe, environment, commit, OutcomeSucceeded, "", 3, steps); err == nil {
		t.Fatal("a successful gate accepted a failed step")
	}
}
