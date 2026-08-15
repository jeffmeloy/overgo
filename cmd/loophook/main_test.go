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
		dirty, err := dirtyPlannedScope([]byte("?? " + path + "\x00"))
		if err != nil {
			t.Fatalf("%s: %v", path, err)
		}
		if !dirty {
			t.Errorf("%s was not treated as meaningful dirty work", path)
		}
	}
	dirty, err := dirtyPlannedScope(nil)
	if err != nil || dirty {
		t.Fatalf("clean status = (%v, %v), want (false, nil)", dirty, err)
	}
}
