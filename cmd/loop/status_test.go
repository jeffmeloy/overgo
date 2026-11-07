package main

import (
	"math"
	"path/filepath"
	"slices"
	"testing"
	"time"

	"overgo/internal/jsonfile"
	"overgo/internal/landing"
	"overgo/internal/plan"
	"overgo/internal/processlock"
	"overgo/internal/testcheck"
)

// TestTheDriverWritesItsStatus holds what the owner reads in place of asking for progress: the decisions waiting on
// the owner's answer, those that change a standing rule or are an act no session takes and are unanswered, and the
// week's pace from the gate's runs: commits a day, the share of commit attempts refused, a preflight no attempt, and
// the hours with no run beyond an hour's gap, a run older than the week left out.
func TestTheDriverWritesItsStatus(t *testing.T) {
	t.Parallel()
	now := time.Date(2026, time.November, 7, 18, 0, 0, 0, time.UTC)
	at := func(hours float64) time.Time { return now.Add(-time.Duration(hours * float64(time.Hour))) }
	run := func(started, ended float64, outcome string) map[string]any {
		return map[string]any{"reference": "row/do", "started": at(started), "ended": at(ended), "outcome": outcome}
	}
	root := t.TempDir()
	testcheck.NoError(t, processlock.EnsureStateDirectory(root))
	runs := []map[string]any{run(200, 199.9, "landed"), run(50, 49.9, "landed"), run(48, 47.9, "rules"), run(47, 46.95, "preflight"), run(10, 9.9, "landed")}
	testcheck.NoError(t, jsonfile.Write(filepath.Join(root, processlock.StateDirectory, "landing_waits.json"), runs, startupFileMode))
	pace := testcheck.Must(landing.WeekPace(root, now))(t)
	// From the week's start to the first run, between the runs, and from the last to now, each gap past an hour.
	idle := (168 - 50) + (49.9 - 48) + (47.9 - 10) + 9.9
	if math.Abs(pace.LandingsPerDay-2.0/7) > 1e-9 || math.Abs(pace.RefusedShare-1.0/3) > 1e-9 || math.Abs(pace.IdleHours-idle) > 1e-6 {
		t.Fatalf("the week's pace = %+v; want 2 commits over 7 days, 1 of 3 attempts refused and %.2f idle hours", pace, idle)
	}
	document := plan.Plan{Decisions: []plan.Decision{
		{ID: "push", Question: "Push the release?", OwnersAct: true},
		{ID: "rule", Question: "Change the rule?", Doctrine: true},
		{ID: "loader", Question: "Which loader?", Recommendation: "the view"},
		{ID: "delete", Question: "Delete the duplicate?", OwnersAct: true, Answer: "keep it"},
	}}
	status := statusOf(now, "row / do: the work", "abc123 a commit", document, pace)
	if want := []string{"push: Push the release?", "rule: Change the rule?"}; !slices.Equal(status.WaitingForOwner, want) || status.Week != pace || status.Dispatched != "row / do: the work" {
		t.Fatalf("the status = %+v; want the two unanswered decisions waiting for the owner, %v, and the pace", status, want)
	}
}
