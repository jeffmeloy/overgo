package main

import (
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"overgo/internal/loop"
	"overgo/internal/testcheck"
)

// TestACampaignWorkerContinuesAndRecordsInItsOwnFolder runs the worker in a real contained subprocess that exits
// immediately after publishing its fixture completion, no in-process callback advancing the row, and holds the driver's
// record of each attempt to the folder the driver runs in (fail-early-runs/attempts-in-place): the test ran in its
// package's folder and wrote cmd/loop/.overgo-runtime/worker_attempts.json there, a file of the module.
func TestACampaignWorkerContinuesAndRecordsInItsOwnFolder(t *testing.T) {
	const helperRoot = "OVERGO_CAMPAIGN_WORKER_TEST_ROOT"
	if root := os.Getenv(helperRoot); root != "" {
		prompt := testcheck.Must(io.ReadAll(os.Stdin))(t)
		body, bound := strings.CutPrefix(string(prompt), "Test campaign policy\n\n")
		if !bound {
			t.Fatal("strategy prompt was not supplied to worker")
		}
		name := strings.TrimSpace(body)
		if name != "first" && name != "second" {
			t.Fatalf("unexpected worker request %q", name)
		}
		testcheck.NoError(t, os.WriteFile(filepath.Join(root, name), []byte("completed"), 0o600))
		return
	}
	packageFolder := testcheck.Must(os.Getwd())(t)
	// A record from an earlier run may stand there already (a gate's worktree keeps ignored files): what is held is that
	// this run writes none.
	recordInPlace := filepath.Join(packageFolder, ".overgo-runtime", "worker_attempts.json")
	before, beforeErr := os.Stat(recordInPlace)
	root := t.TempDir()
	// The driver records each worker attempt in the process state of the folder it runs in (loopWorktree): run here, the
	// record falls in the test's folder and not in cmd/loop/.overgo-runtime, a file of the module go test would read
	// as an input of the package's tests.
	t.Chdir(root)
	t.Setenv(helperRoot, root)
	world := &campaignWorkerWorld{root: root, runner: execWorld{promptPrefix: "Test campaign policy", config: config{Worker: []string{os.Args[0], "-test.run=^TestACampaignWorkerContinuesAndRecordsInItsOwnFolder$"}}}}
	result, err := loop.Run(world, loop.Config{MaxAttemptsPerStep: 2, MaxInvocations: 3})
	if err != nil || result.Reason != loop.ReasonPlanComplete || result.Invocations != 2 {
		t.Fatalf("real worker continuation: %+v %v", result, err)
	}
	if _, err := os.Stat(filepath.Join(root, "second")); err != nil {
		t.Fatal("second worker never executed", err)
	}
	if after, err := os.Stat(recordInPlace); err == nil && (beforeErr != nil || !after.ModTime().Equal(before.ModTime()) || after.Size() != before.Size()) {
		t.Error("the driver recorded its worker attempts in the package's folder, an input of its tests go test would read")
	}
}

type campaignWorkerWorld struct {
	root   string
	runner execWorld
}

func (w *campaignWorkerWorld) Current() (loop.Step, bool, error) {
	for _, name := range []string{"first", "second"} {
		if _, err := os.Stat(filepath.Join(w.root, name)); os.IsNotExist(err) {
			return loop.Step{Item: name, ID: "do"}, true, nil
		} else if err != nil {
			return loop.Step{}, false, err
		}
	}
	return loop.Step{}, false, nil
}
func (w *campaignWorkerWorld) Prompt(step loop.Step) (string, error) { return step.Item, nil }
func (w *campaignWorkerWorld) RunWorker(step loop.Step, prompt, feedback string) (string, error) {
	return w.runner.RunWorker(step, prompt, feedback)
}
func (w *campaignWorkerWorld) Verify(step loop.Step) (string, error) {
	return "worker did not complete " + step.Key(), nil
}
func (w *campaignWorkerWorld) Park(_ loop.Step, reason string) error {
	return fmt.Errorf("unexpected park: %s", reason)
}
func (w *campaignWorkerWorld) Paused() bool { return false }
