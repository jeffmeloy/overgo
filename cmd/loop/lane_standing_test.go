package main

import (
	"strings"
	"testing"
	"time"

	"overgo/internal/plan"
	"overgo/internal/testcheck"
)

// TestTheStatusSaysWhereEachLaneLastPassed holds the status to each cycle lane's standing (lane-passes/do): the commit
// it last passed at and the trunk's commits since, and how its last run ended; a lane that never passed says so.
func TestTheStatusSaysWhereEachLaneLastPassed(t *testing.T) {
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
	standings := laneStandings(t.Context(), root)
	if len(standings) != 2 {
		t.Fatalf("the status reads %d lanes, want 2", len(standings))
	}
	audio, models := standings[0].String(), standings[1].String()
	if !strings.HasPrefix(audio, "lane audio: passed at "+passed[:12]+", 2 commits since; last run passed") ||
		!strings.HasPrefix(models, "lane models: never passed; last run failed at "+passed[:12]) {
		t.Fatalf("the status says %q and %q", audio, models)
	}
}
