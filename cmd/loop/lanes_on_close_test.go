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

	"overgo/internal/dataroot"
	"overgo/internal/plan"
	"overgo/internal/processlock"
	"overgo/internal/testcheck"
)

// TestARowsClosingRunsTheLanesAtItsCommit holds a row's closing commit to running the cycle's lanes at that commit
// (lanes-on-close): a landing that leaves its row open runs none; one that closes it names its commit and starts a
// runner, which runs the lanes in the lanes checkout moved to that commit; a close while they run starts no second
// runner, and the runner, once its run ends, starts the next, built from the trunk, which runs the lanes at it (each run
// supervised by its own commit's runner, lanes-runner-current), a run that failed being said; a runner started while
// another holds the lanes runs nothing. Nothing had run the lanes while the plan held rows, and two of five failed 66
// commits after their last run. Each run finds the trunk's data roots in its checkout, as the tests it runs read them:
// the replay test found none in the lanes checkout (lanes-roots).
func TestARowsClosingRunsTheLanesAtItsCommit(t *testing.T) {
	t.Parallel()
	root := t.TempDir()
	gitFixture(t, root, "init", "-q", "-b", "master")
	const roots = `{"models": "C:/fixture/models"}`
	testcheck.NoError(t, os.WriteFile(filepath.Join(root, dataroot.ConfigFile), []byte(roots), 0o600))
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
		if held, err := os.ReadFile(filepath.Join(at, dataroot.ConfigFile)); err != nil || string(held) != roots {
			t.Errorf("the lanes checkout holds data roots %q, %v; want the trunk's %q", held, err, roots)
		}
		if len(ran) > 1 {
			return nil
		}
		// While the lanes run: a second runner runs nothing, and a close starts none.
		testcheck.NoError(t, runLanesDue(root, io.Discard, func(string) error { return errors.New("a second runner ran the lanes") }, start))
		closing := testcheck.Must(lanesOnClose(root, "later/do", second, start))(t)
		if starts != 1 || !strings.Contains(closing, "once the run going there ends") {
			t.Errorf("a close while the lanes ran said %q and started %d runners in all; want the one, and its run after", closing, starts)
		}
		return errors.New("the device lane did not pass")
	}
	testcheck.NoError(t, runLanesDue(root, &output, run, start))
	if !slices.Equal(ran, []string{first}) || starts != 2 {
		t.Fatalf("the first runner ran the lanes at %q and %d runners were started in all; want it to run its own commit %s alone and start the next for the later one",
			ran, starts, first)
	}
	if !strings.Contains(output.String(), "did not pass: the device lane did not pass") || !strings.Contains(output.String(), "the runner built from the trunk runs it") {
		t.Fatalf("the runner said:\n%s\nwant its run's failure, and the later commit left to the runner it started", output.String())
	}
	// The runner it started, built from the trunk, runs the later commit.
	testcheck.NoError(t, runLanesDue(root, &output, run, start))
	if !slices.Equal(ran, []string{first, second}) || starts != 2 {
		t.Fatalf("the lanes ran at %q, %d runners started; want the closing commits %q, in turn, each by its own runner", ran, starts, []string{first, second})
	}
	output.Reset()
	testcheck.NoError(t, runLanesDue(root, &output, func(string) error { return errors.New("the lanes ran again at a commit they ran at") }, start))
	if output.Len() != 0 {
		t.Fatalf("a runner with no newer commit due said:\n%s\nwant nothing run", output.String())
	}
}
