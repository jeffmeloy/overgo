package main

import (
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"overgo/internal/loop"
)

// The worker runs in a real contained subprocess and exits immediately after
// publishing its fixture completion. No in-process callback advances the row.
func TestCampaignWorkerExitContinues(t *testing.T) {
	const helperRoot = "OVERGO_CAMPAIGN_WORKER_TEST_ROOT"
	if root := os.Getenv(helperRoot); root != "" {
		prompt, err := io.ReadAll(os.Stdin)
		if err != nil {
			t.Fatal(err)
		}
		body, bound := strings.CutPrefix(string(prompt), "Test campaign policy\n\n")
		if !bound {
			t.Fatal("strategy prompt was not supplied to worker")
		}
		name := strings.TrimSpace(body)
		if name != "first" && name != "second" {
			t.Fatalf("unexpected worker request %q", name)
		}
		if err := os.WriteFile(filepath.Join(root, name), []byte("completed"), 0o600); err != nil {
			t.Fatal(err)
		}
		return
	}
	root := t.TempDir()
	t.Setenv(helperRoot, root)
	world := &campaignWorkerWorld{root: root, runner: execWorld{promptPrefix: "Test campaign policy", config: config{Worker: []string{os.Args[0], "-test.run=^TestCampaignWorkerExitContinues$"}, WorkerTimeoutMinutes: 1}}}
	result, err := loop.Run(world, loop.Config{MaxAttemptsPerStep: 2, MaxInvocations: 3})
	if err != nil || result.Reason != loop.ReasonPlanComplete || result.Invocations != 2 {
		t.Fatalf("real worker continuation: %+v %v", result, err)
	}
	if _, err := os.Stat(filepath.Join(root, "second")); err != nil {
		t.Fatal("second worker never executed", err)
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
