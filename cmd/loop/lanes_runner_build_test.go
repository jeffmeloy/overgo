package main

import (
	"bytes"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"overgo/internal/dataroot"
	"overgo/internal/processlock"
	"overgo/internal/testcheck"
)

// TestALanesRunNamesTheBuildThatSupervisedIt holds a run of the cycle's lanes to saying, on its first line, the build
// of the runner that supervised it, so a verdict is read as that build's (lanes-trunk): the runner built at 28390a169
// ran three later commits under the hand-off and bound they had replaced, and no line said so. A runner this test
// builds is unstamped, and says so.
func TestALanesRunNamesTheBuildThatSupervisedIt(t *testing.T) {
	t.Parallel()
	root := t.TempDir()
	gitFixture(t, root, "init", "-q", "-b", "master")
	testcheck.NoError(t, os.WriteFile(filepath.Join(root, "notes.md"), []byte("fixture\n"), 0o644))
	testcheck.NoError(t, os.WriteFile(filepath.Join(root, dataroot.ConfigFile), []byte(`{"models": "C:/fixture/models"}`), 0o600))
	gitFixture(t, root, "add", "notes.md")
	gitFixture(t, root, "commit", "-q", "-m", "fixture")
	commit := gitFixture(t, root, "rev-parse", "HEAD")
	testcheck.NoError(t, processlock.EnsureStateDirectory(root))
	checkout := filepath.Join(root, processlock.StateDirectory, lanesCheckout)
	var output bytes.Buffer
	testcheck.NoError(t, runLanesAt(root, checkout, lanesDue{Commit: commit, Rows: []string{"fixture"}}, "28390a169bcf", &output,
		func(string) error { return nil }))
	first, _, _ := strings.Cut(output.String(), "\n")
	if !strings.Contains(first, "the cycle's lanes run at "+commit[:12]) || !strings.HasSuffix(first, "under the runner built at 28390a169bcf") {
		t.Errorf("a run's first line reads %q; want its commit and the build of the runner that supervised it", first)
	}
	if build := lanesRunnerBuild(); build != "an unstamped build" {
		t.Errorf("a test binary's runner build reads %q; want an unstamped build", build)
	}
}
