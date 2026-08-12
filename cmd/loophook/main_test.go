package main

import "testing"

// TestStopDecision pins the pure Stop verdict: any valve allows; otherwise the
// turn-end is blocked iff work is orphaned (uncommitted .go or an armed dispatch
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
		{"plan complete", false, false, false, true, true, true, false},
		{"clean, nothing armed", false, false, false, false, false, false, false},
		{"uncommitted go", false, false, false, false, true, false, true},
		{"milestone-stop: committed, not dispatched", false, false, false, false, false, true, true},
		{"both orphans", false, false, false, false, true, true, true},
	}
	for _, c := range cases {
		got := stopDecision(c.stopHookActive, c.freshStop, c.gateRunning, c.planComplete, c.dirtyGo, c.marker)
		if got != c.wantBlock {
			t.Errorf("%s: stopDecision=%v want %v", c.name, got, c.wantBlock)
		}
	}
}
