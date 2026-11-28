package main

import (
	"cmp"
	"io"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"overgo/internal/clioptions"
	"overgo/internal/testcheck"
)

// TestEveryValidateOfARunReadsItsRecord holds a measurement run to one hash of each unchanged model file (SQA-525):
// every validate it starts, the dry run that keys the cycle, each phase and the background fill, is given the run's
// record, which keeps the digests of the files its cells bind and which a later binding in the run reads. The dry run
// and the fill had been started without it, and each hashed every model file again beside the preflight, 311 paths and
// 233.8 GB by size at 3a7b3877d.
func TestEveryValidateOfARunReadsItsRecord(t *testing.T) {
	t.Parallel()
	repository := t.TempDir()
	gitFixture(t, repository, "init", "-q", "-b", "master")
	gitFixture(t, repository, "commit", "-q", "--allow-empty", "-m", "measured")
	checkout := filepath.Join(repository, "cycle")
	gitFixture(t, repository, "worktree", "add", "-q", "--detach", checkout, "HEAD")
	run := &runRecorder{root: repository}
	record := run.environment()[0]
	started := map[string][]string{}
	start := func(output io.Writer, _, _ string, environment []string, arguments ...string) error {
		name := cmp.Or(strings.Join(arguments, " "), "the dry run")
		started[name] = environment
		if len(arguments) > 0 {
			return nil
		}
		_, err := io.WriteString(output, `{"run":[],"background":[]}`)
		return err
	}
	testcheck.NoError(t, measureScorecard(checkout, t.TempDir(), nil, run, start))
	testcheck.NoError(t, fillCycle(checkout, t.TempDir(), run, start))
	for _, name := range []string{"the dry run", "-run -preflight", "-run -lanes -reached", "-run -cells -reached", "-run -background"} {
		environment, ran := started[name]
		if !ran || !slices.Contains(environment, record) {
			t.Errorf("validate %s was started (%t) with %q; want the run's record %q among it", name, ran, environment, record)
		}
	}
	if environment := started["-run -background"]; !slices.ContainsFunc(environment, func(setting string) bool {
		return strings.HasPrefix(setting, clioptions.CycleCreditEnvironment+"=")
	}) {
		t.Errorf("the fill was started with %q; want the cycle's credit beside the run's record", environment)
	}
}
