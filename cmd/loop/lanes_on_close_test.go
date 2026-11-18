package main

import (
	"bytes"
	"errors"
	"io"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"overgo/internal/plan"
	"overgo/internal/processlock"
	"overgo/internal/testcheck"
)

// TestARowsClosingRunsTheLanesAtItsCommit holds a row's closing commit to running the cycle's lanes at that commit
// (lanes-on-close): a landing that leaves its row open runs none; one that closes it names its commit and starts a
// runner, which runs the lanes in the lanes checkout moved to that commit; a close while they run starts no second
// runner, and the runner runs the lanes at it once its run ends, a run that failed being said; a runner started while
// another holds the lanes runs nothing. Nothing had run the lanes while the plan held rows, and two of five failed 66
// commits after their last run.
func TestARowsClosingRunsTheLanesAtItsCommit(t *testing.T) {
	t.Parallel()
	root := t.TempDir()
	gitFixture(t, root, "init", "-q", "-b", "master")
	commitPlan := func(rows ...string) string {
		document := plan.Plan{Campaign: "lanes", Lane: "master-lead"}
		for _, row := range rows {
			document.Items = append(document.Items, plan.Item{ID: row, Title: "Row " + row, Steps: []plan.Step{{ID: "do", Title: "Do " + row, Verify: "go test ./" + row}}})
		}
		testcheck.NoError(t, os.MkdirAll(filepath.Join(root, "docs"), 0o755))
		testcheck.NoError(t, plan.Save(filepath.Join(root, filepath.FromSlash(plan.Path)), document))
		gitFixture(t, root, "add", plan.Path)
		gitFixture(t, root, "commit", "-q", "-m", "plan of "+strings.Join(rows, ", "))
		return gitFixture(t, root, "rev-parse", "HEAD")
	}
	commitPlan("kept", "closing", "later")
	first := commitPlan("kept", "later")
	second := commitPlan("kept")

	starts := 0
	start := func() error { starts++; return nil }
	said := testcheck.Must(lanesOnClose(root, "kept/do", first, start))(t)
	if said != "" || starts != 0 {
		t.Fatalf("a landing that left its row open said %q and started %d runners; want neither", said, starts)
	}
	said = testcheck.Must(lanesOnClose(root, "closing/do", first, start))(t)
	if starts != 1 || !strings.Contains(said, "closing closed") || !strings.Contains(said, first[:12]) {
		t.Fatalf("the landing that closed a row said %q and started %d runners; want one, at its commit", said, starts)
	}

	checkout := testcheck.Must(filepath.Abs(filepath.Join(root, processlock.StateDirectory, lanesCheckout)))(t)
	var ran []string
	var output bytes.Buffer
	run := func(at string) error {
		if at != checkout {
			t.Errorf("the lanes ran in %s; want the lanes checkout %s", at, checkout)
		}
		ran = append(ran, gitFixture(t, at, "rev-parse", "HEAD"))
		if len(ran) > 1 {
			return nil
		}
		// While the lanes run: a second runner runs nothing, and a close starts none.
		testcheck.NoError(t, runLanesDue(root, io.Discard, func(string) error { return errors.New("a second runner ran the lanes") }))
		closing := testcheck.Must(lanesOnClose(root, "later/do", second, start))(t)
		if starts != 1 || !strings.Contains(closing, "once the run going there ends") {
			t.Errorf("a close while the lanes ran said %q and started %d runners in all; want the one, and its run after", closing, starts)
		}
		return errors.New("the device lane did not pass")
	}
	testcheck.NoError(t, runLanesDue(root, &output, run))
	if !slices.Equal(ran, []string{first, second}) {
		t.Fatalf("the lanes ran at %q; want the closing commits %q, in turn", ran, []string{first, second})
	}
	if !strings.Contains(output.String(), "did not pass: the device lane did not pass") || strings.Count(output.String(), "the cycle's lanes run at") != 2 {
		t.Fatalf("the runner said:\n%s\nwant each run's start, and the first's failure", output.String())
	}
	output.Reset()
	testcheck.NoError(t, runLanesDue(root, &output, func(string) error { return errors.New("the lanes ran again at a commit they ran at") }))
	if output.Len() != 0 {
		t.Fatalf("a runner with no newer commit due said:\n%s\nwant nothing run", output.String())
	}
}
