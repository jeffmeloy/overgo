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
// lane-record/do): the commit it last passed at and the trunk's commits since, how its last run ended, that it never
// passed, or that it never ran; a name the record holds that no lane is goes unread, where a fixture's lanes had been
// printed.
func TestTheStatusReadsTheDeclaredLanesAlone(t *testing.T) {
	t.Parallel()
	root := t.TempDir()
	gitFixture(t, root, "init", "-q")
	gitFixture(t, root, "commit", "-q", "--allow-empty", "-m", "the lanes passed here")
	passed := gitFixture(t, root, "rev-parse", "HEAD")
	gitFixture(t, root, "commit", "-q", "--allow-empty", "-m", "one")
	gitFixture(t, root, "commit", "-q", "--allow-empty", "-m", "two")
	now := time.Now()
	testcheck.NoError(t, plan.RecordLaneRun(root, "audio", plan.LaneRun{Commit: passed, Passed: true, Ended: now}))
	testcheck.NoError(t, plan.RecordLaneRun(root, "models", plan.LaneRun{Commit: passed, Ended: now}))
	testcheck.NoError(t, plan.RecordLaneRun(root, "passes", plan.LaneRun{Commit: passed, Passed: true, Ended: now}))
	said := map[string]string{}
	for _, standing := range laneStandings(t.Context(), root) {
		said[standing.Lane] = standing.String()
	}
	if len(said) != len(landing.CycleLanes()) || said["passes"] != "" {
		t.Fatalf("the status reads %v; want the %d declared lanes and no other", said, len(landing.CycleLanes()))
	}
	if !strings.HasPrefix(said["audio"], "lane audio: passed at "+passed[:12]+", 2 commits since; last run passed") ||
		!strings.HasPrefix(said["models"], "lane models: never passed; last run failed at "+passed[:12]) || said["sequences"] != "lane sequences: never run" {
		t.Fatalf("the status says %q, %q and %q", said["audio"], said["models"], said["sequences"])
	}
}
