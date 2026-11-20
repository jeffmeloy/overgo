package main

import (
	"bytes"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"overgo/internal/clioptions"
	"overgo/internal/dataroot"
	"overgo/internal/jsonfile"
	"overgo/internal/plan"
	"overgo/internal/processlock"
	"overgo/internal/testcheck"
)

// TestAYieldedRunSaysItsFailures holds the runner to saying how a run that yielded ended (SQA-437): a run whose yield
// record names lanes it failed before it yielded did not pass, its failures said beside the yield, and a run that only
// yielded is said to have yielded. The runner had said yielded alone whenever the record named its commit, the failed
// checkpoint left in the detailed log.
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
	record := filepath.Join(root, processlock.StateDirectory, plan.LanesYieldedFile)
	ended := func(failures ...string) string {
		var output bytes.Buffer
		testcheck.NoError(t, runLanesAt(root, checkout, lanesDue{Commit: commit, Rows: []string{"fixture"}}, &output, func(string) error {
			testcheck.NoError(t, jsonfile.Write(record, plan.LanesYielded{Commit: commit, To: "a-newer-close", Failures: failures}, clioptions.PrivateFileMode))
			return errors.New("exit status 1")
		}))
		return output.String()
	}
	if said := ended("cycle lane models: TestMOSSSamplingCheckpoint failed"); !strings.Contains(said, "did not pass: cycle lane models: TestMOSSSamplingCheckpoint failed") || !strings.Contains(said, "then yielded to a-newer-clos") {
		t.Errorf("a run that failed a checkpoint and then yielded was said as:\n%s\nwant its failure, then its yield", said)
	}
	if said := ended(); strings.Contains(said, "did not pass") || !strings.Contains(said, "yielded to a-newer-clos") {
		t.Errorf("a run that only yielded was said as:\n%s\nwant it yielded", said)
	}
}
