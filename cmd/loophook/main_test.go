package main

import (
	"os"
	"path/filepath"
	"slices"
	"testing"
)

// TestStopDecision pins the pure Stop verdict: any valve allows; otherwise
// orphaned turn-created work blocks first, and ANY turn under an open plan
// row blocks with the continue verdict -- committed progress is necessary,
// never sufficient (owner rule: the loop runs until the plan is empty; the
// sanctioned pause is a recorded user stop, which arms the freshStop valve).
func TestStopDecision(t *testing.T) {
	cases := []struct {
		name                                                          string
		stopHookActive, freshStop, gateRunning, planComplete, dirtyGo bool
		want                                                          string
	}{
		{"retry valve", true, false, false, false, true, stopAllow},
		{"fresh recorded stop", false, true, false, false, true, stopAllow},
		{"gate in flight", false, false, true, false, true, stopAllow},
		// loop-hardening-7 accepted residual: during the gate's go-run COMPILE
		// phase gateRunning is momentarily false, so a wait-turn BLOCKS on dirty
		// .go (safe direction); the next attempt's retry valve allows.
		{"gate compile-window (pre-retry)", false, false, false, false, true, stopOrphan},
		{"plan complete", false, false, false, true, true, stopAllow},
		{"uncommitted go", false, false, false, false, true, stopOrphan},
		{"committed progress, open rows remain", false, false, false, false, false, stopContinue},
		{"answered prompt, no progress", false, false, false, false, false, stopContinue},
		{"open row, idle turn", false, false, false, false, false, stopContinue},
	}
	for _, c := range cases {
		got := stopDecision(c.stopHookActive, c.freshStop, c.gateRunning, c.planComplete, c.dirtyGo)
		if got != c.want {
			t.Errorf("%s: stopDecision=%q want %q", c.name, got, c.want)
		}
	}
}

// TestBoundedRequestScopesAnswerNotTurn pins the owner rule: the bounded
// marker classifies the ANSWER's scope and is consumed without rendering a
// verdict — a bounded turn with no committed progress still draws the
// continue block, and only a valve (e.g. a recorded user stop) allows it.
func TestBoundedRequestScopesAnswerNotTurn(t *testing.T) {
	marker := filepath.Join(t.TempDir(), "bounded")
	if err := os.WriteFile(marker, []byte("bounded\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if !consumeBoundedRequest(marker) || fileExists(marker) {
		t.Fatal("bounded request marker was not consumed")
	}
	if got := stopDecision(false, false, false, false, false); got != stopContinue {
		t.Fatalf("bounded no-progress turn = %q, want the continue block", got)
	}
	if got := stopDecision(false, true, false, false, false); got != stopAllow {
		t.Fatalf("recorded user stop = %q, want allow", got)
	}
	if !boundedRequestText("automation plan completed?") || !boundedRequestText("summarize work completed") || !boundedRequestText("should we add a doc check?") {
		t.Fatal("bounded question/status request was not classified")
	}
	if boundedRequestText("can you implement all remaining tasks?") || boundedRequestText("continue until done") {
		t.Fatal("action or continuation request was classified as bounded")
	}
}

// TestHeadProgressBaseline pins the progress check's fail-open direction: a
// missing or legacy (headless) snapshot cannot prove the turn made no
// progress, so it must allow rather than harass.
func TestHeadProgressBaseline(t *testing.T) {
	if _, ok := readTurnBaseFrom([]byte(`[{"path":"a.go"}]`)); ok {
		t.Fatal("legacy facts-only snapshot parsed as a head baseline")
	}
	base, ok := readTurnBaseFrom([]byte(`{"head":"abc123","facts":[]}`))
	if !ok || base.Head != "abc123" {
		t.Fatalf("snapshot with head failed to parse: %+v ok=%v", base, ok)
	}
}

// TestStopIgnoresPreexistingDirt pins the turn-scoped dirt verdict: dirt already
// present at the turn-start snapshot is another lane's parked work and never
// blocks this turn's end; only turn-created dirt is an orphaning obligation.
// A missing snapshot fails SAFE -- every dirty path blocks, the pre-snapshot
// behavior -- so a lost marker errs toward committing, never toward orphaning.
func TestStopIgnoresPreexistingDirt(t *testing.T) {
	parked := []dirtyFact{
		{Path: "internal/discovery/servable.go", WorktreeStatus: "M", WorkIdentity: "first"},
		{Path: "internal/modelrecipe/lifecycle.go", WorktreeStatus: "M", WorkIdentity: "second"},
	}

	if created := turnCreatedDirt(parked, parked); len(created) != 0 {
		t.Fatalf("pre-existing dirt reported as turn-created: %v", created)
	}
	if got := stopDecision(false, false, true, false, len(turnCreatedDirt(parked, parked)) > 0); got != stopAllow {
		t.Fatalf("progressed turn with only parked parallel-lane dirt = %q, want allow", got)
	}

	current := append(slices.Clone(parked), dirtyFact{Path: "cmd/loophook/main.go", WorktreeStatus: "M", WorkIdentity: "third"})
	created := turnCreatedDirt(parked, current)
	if len(created) != 1 || created[0].Path != "cmd/loophook/main.go" {
		t.Fatalf("turn-created dirt = %v, want the new path only", created)
	}
	if got := stopDecision(false, false, false, false, len(created) > 0); got != stopOrphan {
		t.Fatalf("turn that created dirt = %q, want the orphan block", got)
	}

	// Missing snapshot: nil base treats every current path as turn-created.
	if created := turnCreatedDirt(nil, parked); len(created) != len(parked) {
		t.Fatalf("missing snapshot must fail safe toward blocking, got %v", created)
	}

	changed := slices.Clone(parked)
	changed[0].WorkIdentity = "changed-this-turn"
	if created := turnCreatedDirt(parked, changed); len(created) != 1 || created[0].Path != parked[0].Path {
		t.Fatalf("modified parked path was not turn-created dirt: %v", created)
	}
}
