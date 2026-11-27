package main

import (
	"bytes"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"overgo/internal/dataroot"
	"overgo/internal/processlock"
	"overgo/internal/testcheck"
)

// TestAYieldedRunSaysItsFailures holds the runner to saying how a run ended (SQA-437, lanes-record): a run that failed a
// test, before a yield or not, did not pass, its failure said; a run that only yielded, or passed, ended without a
// failure, each test it ran kept as it ended. The runner had said yielded alone whenever a yield record named its
// commit, the failed checkpoint left in the detailed log.
func TestAYieldedRunSaysItsFailures(t *testing.T) {
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
	ended := func(err error) string {
		var output bytes.Buffer
		testcheck.NoError(t, runLanesAt(root, checkout, lanesDue{Commit: commit, Rows: []string{"fixture"}}, "a fixture build", &output, func(string) error {
			return err
		}, nil))
		return output.String()
	}
	if said := ended(errors.New("cycle lane models: TestMOSSSamplingCheckpoint failed")); !strings.Contains(said, "did not pass: cycle lane models: TestMOSSSamplingCheckpoint failed") {
		t.Errorf("a run that failed a checkpoint was said as:\n%s\nwant its failure", said)
	}
	if said := ended(nil); strings.Contains(said, "did not pass") || !strings.Contains(said, "ended without a failure") {
		t.Errorf("a run that only yielded was said as:\n%s\nwant it ended without a failure", said)
	}
}
