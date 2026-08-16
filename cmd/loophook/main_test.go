package main

import (
	"os"
	"path/filepath"
	"testing"
)

// TestStopDecision pins the pure Stop verdict: any valve allows; otherwise the
// turn-end is blocked iff work is orphaned (uncommitted work or an armed dispatch
// marker). Crucially, a committed-but-not-dispatched turn (markerArmed, clean
// tree) BLOCKS -- that is the milestone-stop the old progress-based gate let
// through.
func TestStopDecision(t *testing.T) {
	cases := []struct {
		name                                                                  string
		stopHookActive, freshStop, gateRunning, planComplete, dirtyGo, marker bool
		wantBlock                                                             bool
	}{
		{"retry valve", true, false, false, false, true, true, false},
		{"fresh stop", false, true, false, false, true, true, false},
		{"gate in flight", false, false, true, false, true, true, false},
		// loop-hardening-7 accepted residual: during the gate's go-run COMPILE
		// phase gateRunning is momentarily false, so a wait-turn BLOCKS on dirty
		// .go (safe direction). Recovery is the "retry valve" row above -- the
		// next attempt carries stop_hook_active and allows. This pair pins the
		// self-healing that makes the compile-gap safe without a stale-marker fix.
		{"gate compile-window (pre-retry)", false, false, false, false, true, true, true},
		{"plan complete", false, false, false, true, true, true, false},
		{"clean, nothing armed", false, false, false, false, false, false, false},
		{"uncommitted go", false, false, false, false, true, false, true},
		{"milestone-stop: committed, not dispatched", false, false, false, false, false, true, true},
		{"both orphans", false, false, false, false, true, true, true},
	}
	for _, c := range cases {
		got := stopDecision(c.stopHookActive, false, c.freshStop, c.gateRunning, c.planComplete, c.dirtyGo, c.marker)
		if got != c.wantBlock {
			t.Errorf("%s: stopDecision=%v want %v", c.name, got, c.wantBlock)
		}
	}
}

func TestBoundedRequestCompletionDoesNotRecordStop(t *testing.T) {
	marker := filepath.Join(t.TempDir(), "bounded")
	if err := os.WriteFile(marker, []byte("bounded\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	bounded := consumeBoundedRequest(marker)
	if !bounded || fileExists(marker) {
		t.Fatal("bounded request marker was not consumed")
	}
	if stopDecision(false, bounded, false, false, false, true, true) {
		t.Fatal("bounded answer was blocked by continuation state")
	}
	if !boundedRequestText("automation plan completed?") || !boundedRequestText("summarize work completed") || !boundedRequestText("should we add a doc check?") {
		t.Fatal("bounded question/status request was not classified")
	}
	if boundedRequestText("can you implement all remaining tasks?") || boundedRequestText("continue until done") {
		t.Fatal("action or continuation request was classified as bounded")
	}
}

func TestLoopDirtySnapshot(t *testing.T) {
	for _, path := range []string{
		"internal/model/model.go",
		"internal/model/architecture_profiles.json",
		"internal/server/webui/index.html",
		"docs/design.md",
	} {
		paths, err := dirtyPaths([]byte("?? " + path + "\x00"))
		if err != nil {
			t.Fatalf("%s: %v", path, err)
		}
		if len(paths) != 1 || paths[0] != path {
			t.Errorf("%s was not treated as meaningful dirty work: %v", path, paths)
		}
	}
	paths, err := dirtyPaths(nil)
	if err != nil || len(paths) != 0 {
		t.Fatalf("clean status = (%v, %v), want (none, nil)", paths, err)
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
	if stopDecision(false, false, false, false, false, len(turnCreatedDirt(parked, parked)) > 0, false) {
		t.Fatal("bounded turn with only parked parallel-lane dirt was blocked")
	}

	current := append(append([]dirtyFact{}, parked...), dirtyFact{Path: "cmd/loophook/main.go", WorktreeStatus: "M", WorkIdentity: "third"})
	created := turnCreatedDirt(parked, current)
	if len(created) != 1 || created[0].Path != "cmd/loophook/main.go" {
		t.Fatalf("turn-created dirt = %v, want the new path only", created)
	}
	if !stopDecision(false, false, false, false, false, len(created) > 0, false) {
		t.Fatal("turn that created dirt was allowed to end without committing")
	}

	// Missing snapshot: nil base treats every current path as turn-created.
	if created := turnCreatedDirt(nil, parked); len(created) != len(parked) {
		t.Fatalf("missing snapshot must fail safe toward blocking, got %v", created)
	}

	changed := append([]dirtyFact{}, parked...)
	changed[0].WorkIdentity = "changed-this-turn"
	if created := turnCreatedDirt(parked, changed); len(created) != 1 || created[0].Path != parked[0].Path {
		t.Fatalf("modified parked path was not turn-created dirt: %v", created)
	}
}
