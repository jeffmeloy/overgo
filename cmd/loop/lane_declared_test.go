package main

import (
	"strings"
	"testing"
	"time"

	"overgo/internal/landing"
	"overgo/internal/plan"
	"overgo/internal/testcheck"
)

// TestTheStatusReadsTheDeclaredLanesAlone holds the status to each of the cycle's declared lanes (lane-passes/do,
// lane-record/do, lanes-record): a lane run whole by its own result, a lane run test by test by its packages' tests',
// each how many passed and failed and when the newest was taken, or that it never ran; a result no lane holds goes
// unread, where a fixture's lanes had been printed.
func TestTheStatusReadsTheDeclaredLanesAlone(t *testing.T) {
	t.Parallel()
	root := t.TempDir()
	newest := time.Now()
	testcheck.NoError(t, plan.RecordTestResults(root, plan.TestResult{Commit: "c1", Ended: newest.Add(-time.Hour)}, plan.LaneTest("needle-reference")))
	testcheck.NoError(t, plan.RecordTestResults(root, plan.TestResult{Commit: "c2", Ended: newest}, "overgo/internal/inference TestA", "overgo/cmd/loop TestB"))
	testcheck.NoError(t, plan.RecordTestResults(root, plan.TestResult{Commit: "c2", Failed: true, Ended: newest}, "overgo/internal/latentimage TestC"))
	testcheck.NoError(t, plan.RecordTestResults(root, plan.TestResult{Commit: "c2", Ended: newest}, plan.LaneTest("passes")))
	said := map[string]string{}
	for _, standing := range laneStandings(root) {
		said[standing.Lane] = standing.String()
	}
	if len(said) != len(landing.CycleLanes()) || said["passes"] != "" {
		t.Fatalf("the status reads %v; want the %d declared lanes and no other", said, len(landing.CycleLanes()))
	}
	if !strings.HasPrefix(said["models"], "lane models: 2 passed, 1 failed; the newest at c2") ||
		!strings.HasPrefix(said["needle-reference"], "lane needle-reference: 1 passed, 0 failed; the newest at c1") || said["sequences"] != "lane sequences: never run" {
		t.Fatalf("the status says %q, %q and %q", said["models"], said["needle-reference"], said["sequences"])
	}
}
