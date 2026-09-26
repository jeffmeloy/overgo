package gate

import (
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
	"time"

	"overgo/internal/artifact"
	"overgo/internal/automationcheck"
	"overgo/internal/overgodb"
	"overgo/internal/runrecord"
	"overgo/internal/testutil"
)

// TestNextGateResumesUnfinishedLanes holds the next gate to resolving the
// lanes a landed commit owes, never refusing on their bookkeeping: an open
// obligation no runner holds -- its runner ended before recording an outcome,
// with or without a stale locator naming a live process -- is debt the gate
// runs inline, and a live runner is awaited on its own lock, so the gate
// reads the outcome that runner records rather than the state it began from.
func TestNextGateResumesUnfinishedLanes(t *testing.T) {
	t.Parallel()
	repo := t.TempDir()
	if err := os.Mkdir(filepath.Join(repo, "tmp"), 0o755); err != nil {
		t.Fatal(err)
	}
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	commit := strings.Repeat("d", 40)
	now := time.Now()
	pending, err := runrecord.NewGateLaneObligation(commit,
		testutil.ArtifactID(t, artifact.KindEvidence, "resumption preparation"),
		testutil.ArtifactID(t, artifact.KindEvidence, "resumption result"),
		[]string{automationcheck.WebUICheckName}, []string{"docs/plan.json"}, now)
	if err != nil {
		t.Fatal(err)
	}
	publish := func(state runrecord.GateLaneObligation, previous *artifact.ID) {
		t.Helper()
		batch, err := state.Batch(previous)
		if err != nil {
			t.Fatal(err)
		}
		if _, err := store.CommitAs(t.Context(), gateProducer, batch); err != nil {
			t.Fatal(err)
		}
	}
	publish(pending, nil)
	if debt, err := requireLaneObligationsResolved(t.Context(), repo, store); err != nil || debt == nil || debt.ID != pending.ID {
		t.Fatalf("pending obligation without a runner: debt=%v err=%v", debt, err)
	}
	locator := gateLanesLocator{Version: artifact.InitialDocumentVersion, State: runrecord.LaneObligationRunning, Obligation: pending.ID, CodeCommit: commit, PID: os.Getpid(), Updated: now}
	if err := writeJSON(repo, gateLanesLocatorFile, locator, 0o644); err != nil {
		t.Fatal(err)
	}
	if debt, err := requireLaneObligationsResolved(t.Context(), repo, store); err != nil || debt == nil || debt.ID != pending.ID {
		t.Fatalf("stale locator with a live unrelated PID blocked recovery: debt=%v err=%v", debt, err)
	}

	// A runner holds the lanes, records their outcome, then releases them.
	guard, err := acquireLaneRunner(repo)
	if err != nil {
		t.Fatal(err)
	}
	answered := make(chan waitedGate, 1)
	started := make(chan struct{})
	go func() {
		close(started)
		debt, err := requireLaneObligationsResolved(t.Context(), repo, store)
		answered <- waitedGate{err: err, afterRunner: debt == nil}
	}()
	<-started
	for range 64 {
		runtime.Gosched()
	}
	running, err := pending.Transition(runrecord.LaneObligationRunning, artifact.ID{}, now.Add(time.Second))
	if err != nil {
		t.Fatal(err)
	}
	publish(running, &pending.ID)
	passed, err := running.Transition(runrecord.LaneObligationPassed,
		testutil.ArtifactID(t, artifact.KindEvidence, "resumption lanes"), now.Add(2*time.Second))
	if err != nil {
		t.Fatal(err)
	}
	publish(passed, &running.ID)
	if err := guard.Close(); err != nil {
		t.Fatal(err)
	}
	if gate := <-answered; gate.err != nil || !gate.afterRunner {
		t.Fatalf("the next gate answered before the runner it waited on recorded its outcome: %+v", gate)
	}
}
