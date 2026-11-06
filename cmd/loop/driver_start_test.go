package main

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"overgo/internal/plan"
	"overgo/internal/processlock"
	"overgo/internal/testcheck"
)

// TestADriverStartsFromItsConfigAlone holds the unattended driver to starting from its config: in a directory that
// holds no store, no strategy and no plan, with no worker named in the environment, loop.json's worker, name, prompt
// file and two budgets start a run, which the pause marker ends at once, and the campaign it begins records the
// config's name. A config without its prompt file or its budgets, or one that still names a strategy, is refused.
func TestADriverStartsFromItsConfigAlone(t *testing.T) {
	// Serial: it changes the process's directory and environment.
	t.Chdir(t.TempDir())
	t.Setenv(plan.AutomationWorkerEnvironment, "")
	testcheck.NoError(t, processlock.EnsureStateDirectory("."))
	state := processlock.StateDirectory
	testcheck.NoError(t, os.WriteFile(filepath.Join(state, "loop_prompt.md"), []byte("Fixture policy"), 0o600))
	testcheck.NoError(t, os.WriteFile(filepath.Join(state, pauseMarker), nil, 0o600))
	absent := filepath.Join(t.TempDir(), "absent-store")
	start := func(config string) error {
		testcheck.NoError(t, os.WriteFile(filepath.Join(state, "loop.json"), []byte(config), 0o600))
		return run([]string{"-repo", absent})
	}
	testcheck.NoError(t, start(`{"worker": ["worker", "{prompt}"], "name": "fixture-worker", "prompt": "loop_prompt.md",
		"max_attempts_per_step": 1, "max_invocations": 1, "worker_timeout_minutes": 1}`))
	if locator := string(testcheck.Must(os.ReadFile(filepath.Join(state, "loop_supervisor.json")))(t)); !strings.Contains(locator, `"fixture-worker"`) {
		t.Fatalf("the campaign records %s, want the config's name", locator)
	}
	if worker := os.Getenv(plan.AutomationWorkerEnvironment); worker != "" {
		t.Fatalf("the run left the worker's name %q in the process's environment", worker)
	}
	for name, config := range map[string]string{
		"no prompt file": `{"worker": ["worker"], "name": "fixture-worker", "max_attempts_per_step": 1, "max_invocations": 1}`,
		"no budgets":     `{"worker": ["worker"], "name": "fixture-worker", "prompt": "loop_prompt.md"}`,
		"a strategy":     `{"worker": ["worker"], "name": "fixture-worker", "prompt": "loop_prompt.md", "strategy_id": "profile:sha256:00", "max_attempts_per_step": 1, "max_invocations": 1}`,
	} {
		if err := start(config); err == nil {
			t.Errorf("a config with %s started a run", name)
		}
	}
}
