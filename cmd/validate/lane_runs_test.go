package main

import (
	"io"
	"os/exec"
	"path/filepath"
	"testing"

	"overgo/internal/plan"
	"overgo/internal/testcheck"
)

// TestADirectLaneRunRecordsEachLanesPass holds a run of the cycle's lanes to keeping each test's result where it is told
// to, with the commit it ran at (lane-passes/do, lane-record/do, lanes-record): tests that pass and fail in a worktree of
// the trunk are kept in the trunk's checkout, each with that commit and whether it failed, and the worktree keeps none.
func TestADirectLaneRunRecordsEachLanesPass(t *testing.T) {
	t.Parallel()
	trunk, head := fixtureRepository(t)
	worktree := filepath.Join(t.TempDir(), "lanes")
	add := exec.CommandContext(t.Context(), "git", "worktree", "add", "-q", "--detach", worktree, head)
	add.Dir = trunk
	if output, err := add.CombinedOutput(); err != nil {
		t.Fatalf("git worktree add: %v: %s", err, output)
	}
	keep := testcheck.Must(resultKeeper(t.Context(), io.Discard, worktree, trunkCheckout(t.Context(), worktree), true))(t)
	keep(false, "overgo/internal/audioparity TestPasses")
	keep(true, "overgo/internal/audioparity TestFails")
	results := plan.ReadTestResults(trunk)
	passed, failed := results["overgo/internal/audioparity TestPasses"], results["overgo/internal/audioparity TestFails"]
	if passed.Commit != head || passed.Failed || failed.Commit != head || !failed.Failed || passed.Ended.IsZero() {
		t.Fatalf("the trunk keeps %+v and %+v; want a pass and a failure, both at %s", passed, failed, head)
	}
	if kept := plan.ReadTestResults(worktree); len(kept) != 0 {
		t.Fatalf("the worktree the lanes ran in keeps %+v; want the trunk's checkout to keep them", kept)
	}
}
