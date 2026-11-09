//go:build windows && modeltest

package main

import (
	"context"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"overgo/internal/clioptions"
	"overgo/internal/gitauthority"
	"overgo/internal/testcheck"
	"overgo/internal/testutil"
)

// replayFixtureSteps are landed steps the replay is held to: a landing that added only its test, whose check passes at
// its parent once the test is in place and so is no task, and the smallest change among the landings since 2026-11-01
// whose check runs one host test (4 lines beside its test).
var replayFixtureSteps = []string{"fault-finding/agent-lifecycles", "empty-added-token/refuse"}

// TestLandedStepsReplayForAWorker holds the replay to what it records, by a worker whose outcome is known: one that
// writes the landing's own change, which the step's check, failing at its parent, must then pass. Each step is checked
// out at the commit it landed on with the landing's tests in place, and a step whose check passes there is skipped as
// no task. Each check at an old commit compiles anew, about 100 s on 2026-11-08, so the test holds three within go
// test's ten minutes. (A worker that fails is on record from Qwen3.8-27B's replay through the -replay path: 40
// replies and 677.7 s on recipe-repairs/declared-penalty, 39 of them file reads, and its check still failed.)
func TestLandedStepsReplayForAWorker(t *testing.T) {
	ctx := t.Context()
	root := testutil.RepoRoot(t)
	landed := testcheck.Must(landedSteps(ctx, root, "2026-11-01"))(t)
	checkout := filepath.Join(t.TempDir(), "replay")
	t.Cleanup(func() { testcheck.NoError(t, gitWrite(root, "worktree", "remove", "--force", checkout)) })
	for _, reference := range replayFixtureSteps {
		at := slices.IndexFunc(landed, func(step replayed) bool { return step.reference == reference })
		if at < 0 {
			t.Fatalf("no step %s among the %d landed since 2026-11-01 with one test as their check", reference, len(landed))
		}
		step := landed[at]
		for _, worker := range []replayWorker{landingWorker(root, step)} {
			record := testcheck.Must(replay(ctx, root, checkout, step, worker, 1))(t)
			t.Logf("%s by %s: passed %t in %d attempts, %d calls, %d ms; skipped %q", reference, record.Worker, record.Passed, record.Attempts, record.Calls, record.WallMS, record.Skipped)
			switch {
			case reference == replayFixtureSteps[0]:
				if record.Skipped == "" {
					t.Errorf("%s added only its test, and its replay put it to %s: %+v", reference, worker.name, record)
				}
			case record.Skipped != "" || record.Attempts != 1 || record.WallMS <= 0 || record.Passed != (worker.name == "landing") || record.Passed == (record.Failure != ""):
				t.Errorf("%s replayed by %s as %+v, want one attempt that passes exactly when the worker wrote the landing's change", reference, worker.name, record)
			}
		}
	}
}

// landingWorker writes the change that landed step, every file it changed but its tests, as the worker whose answer is
// known to pass the step's check.
func landingWorker(root string, step replayed) replayWorker {
	return replayWorker{name: "landing", work: func(ctx context.Context, checkout string, task replayTask) (localWork, error) {
		changed, err := gitauthority.Query(ctx, root, "diff", "--name-only", "--diff-filter=AM", step.parent, step.landed)
		if err != nil {
			return localWork{}, err
		}
		var work localWork
		for path := range strings.FieldsSeq(string(changed)) {
			if slices.Contains(task.Tests, path) {
				continue
			}
			data, err := gitauthority.Query(ctx, root, "show", step.landed+":"+path)
			if err != nil {
				return work, err
			}
			if err := os.WriteFile(filepath.Join(checkout, filepath.FromSlash(path)), data, clioptions.OutputFileMode); err != nil {
				return work, err
			}
			work.Calls++
		}
		return work, nil
	}}
}
