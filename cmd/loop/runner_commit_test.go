package main

import (
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"runtime/debug"
	"testing"

	"overgo/internal/clioptions"
	"overgo/internal/dataroot"
	"overgo/internal/jsonfile"
	"overgo/internal/processlock"
	"overgo/internal/testcheck"
)

// TestTheLanesRunnerIsBuiltFromTheCommitItRuns holds the lanes runner to naming the commit it was built from
// (runner-from-commit/do). In a fixture repository whose runner prints "committed" and the commit it was told, and
// whose checkout holds an uncommitted edit printing "uncommitted", the runner built prints "committed" and the trunk's
// commit. A build so told is named by that commit alone, even where go stamped it modified, while a build told nothing
// keeps go's stamp. buildChecker had built the committed code all along, but go stamps a build from the main checkout
// above the checker worktree, whose .git is a file go does not take for a root, so lanes.log said the runner was built
// "with its tree's changes": the main checkout's, which the runner did not hold.
func TestTheLanesRunnerIsBuiltFromTheCommitItRuns(t *testing.T) {
	t.Parallel()
	root := t.TempDir()
	runner := func(says string) string {
		return "package main\n\nimport \"fmt\"\n\nvar builtCommit string\n\nfunc main() { fmt.Print(\"" + says + " \" + builtCommit) }\n"
	}
	for path, content := range map[string]string{"go.mod": "module fixture\n\ngo 1.26\n", "cmd/loop/main.go": runner("committed")} {
		testcheck.NoError(t, os.MkdirAll(filepath.Dir(filepath.Join(root, path)), 0o755))
		testcheck.NoError(t, os.WriteFile(filepath.Join(root, path), []byte(content), 0o644))
	}
	gitFixture(t, root, "init", "-q", "-b", "master")
	gitFixture(t, root, "add", ".")
	gitFixture(t, root, "commit", "-q", "-m", "fixture")
	head := gitFixture(t, root, "rev-parse", "HEAD")
	testcheck.NoError(t, os.WriteFile(filepath.Join(root, "cmd/loop/main.go"), []byte(runner("uncommitted")), 0o644))
	built := filepath.Join(t.TempDir(), "loop.exe")
	if refusal, err := buildChecker(root, built, "./cmd/loop", runBuild); err != nil {
		t.Fatalf("the runner was not built: %v: %s", err, refusal)
	}
	if said := string(testcheck.Must(exec.Command(built).Output())(t)); said != "committed "+head {
		t.Fatalf("the runner says %q; want the committed runner told its commit, %q", said, "committed "+head)
	}
	stamped := &debug.BuildInfo{Settings: []debug.BuildSetting{{Key: "vcs.revision", Value: head}, {Key: "vcs.modified", Value: "true"}}}
	if named := describeBuild(head, stamped); named != head[:12] {
		t.Errorf("a runner built from %s names itself %q; want the commit alone, %q", head, named, head[:12])
	}
	if named := describeBuild("", stamped); named != head[:12]+" with its tree's changes" {
		t.Errorf("a build told no commit names itself %q; want go's stamp", named)
	}
}

// TestACommitWhoseSetupFailedRunsOnceRepaired holds a row-close run to running a commit whose lanes checkout could not
// be set up once the setup is repaired (SQA-496). The commit is due with the trunk's data roots missing, so the
// checkout cannot be given them: the runner fails and runs no lane. Once the roots are there, the next runner runs the
// commit, once. The commit had been recorded as begun before its setup, so the repaired runner found it run and ran
// nothing.
func TestACommitWhoseSetupFailedRunsOnceRepaired(t *testing.T) {
	t.Parallel()
	root := t.TempDir()
	gitFixture(t, root, "init", "-q", "-b", "master")
	testcheck.NoError(t, os.WriteFile(filepath.Join(root, "notes.md"), []byte("fixture\n"), 0o644))
	gitFixture(t, root, "add", "notes.md")
	gitFixture(t, root, "commit", "-q", "-m", "fixture")
	commit := gitFixture(t, root, "rev-parse", "HEAD")
	testcheck.NoError(t, processlock.EnsureStateDirectory(root))
	testcheck.NoError(t, jsonfile.Write(filepath.Join(root, processlock.StateDirectory, lanesDueFile), lanesDue{Commit: commit, Rows: []string{"fixture"}}, clioptions.PrivateFileMode))
	runs := 0
	run := func(string) error { runs++; return nil }
	none := func() error { return nil }
	if err := runLanesDue(root, io.Discard, run, none); err == nil || runs != 0 {
		t.Fatalf("a runner whose checkout lacked the data roots ended as %v after %d runs; want it to fail and run no lane", err, runs)
	}
	testcheck.NoError(t, os.WriteFile(filepath.Join(root, dataroot.ConfigFile), []byte(`{"models": "C:/fixture/models"}`), 0o600))
	testcheck.NoError(t, runLanesDue(root, io.Discard, run, none))
	testcheck.NoError(t, runLanesDue(root, io.Discard, run, none))
	if runs != 1 {
		t.Errorf("once the setup was repaired the lanes ran %d times at the commit due; want once", runs)
	}
}
