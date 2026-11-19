package main

import (
	"os/exec"
	"path/filepath"
	"testing"

	"overgo/internal/plan"
	"overgo/internal/testcheck"
	"overgo/internal/testevidence"
)

// TestADirectLaneRunRecordsEachLanesPass holds a run of the cycle's lanes to keeping each lane's end where it is told
// to, with the commit it ran at (lane-passes/do, lane-record/do): a lane that passes in a worktree of the trunk is
// recorded in the trunk's checkout as its latest run and its latest pass, and a later run that fails is its latest run
// with that pass kept.
func TestADirectLaneRunRecordsEachLanesPass(t *testing.T) {
	t.Parallel()
	trunk, head := fixtureRepository(t)
	worktree := filepath.Join(t.TempDir(), "lanes")
	add := exec.CommandContext(t.Context(), "git", "worktree", "add", "-q", "--detach", worktree, head)
	add.Dir = trunk
	if output, err := add.CombinedOutput(); err != nil {
		t.Fatalf("git worktree add: %v: %s", err, output)
	}
	kept := trunkCheckout(t.Context(), worktree)
	testcheck.NoError(t, keepLaneRun(t.Context(), kept, worktree, "audio", true, nil, testevidence.GoTestReport{}))
	testcheck.NoError(t, keepLaneRun(t.Context(), kept, worktree, "audio", false, nil, testevidence.GoTestReport{}))
	record := plan.ReadLaneRuns(trunk)["audio"]
	if record.Last.Commit != head || record.Last.Passed || record.Passed.Commit != head || !record.Passed.Passed || record.Passed.Ended.IsZero() {
		t.Fatalf("the trunk keeps the lane as %+v; want its failed latest run and its pass, both at %s", record, head)
	}
	if kept := plan.ReadLaneRuns(worktree); len(kept) != 0 {
		t.Fatalf("the worktree the lanes ran in keeps %+v; want the trunk's checkout to keep them", kept)
	}
}
