package main

import (
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"overgo/internal/jsonfile"
	"overgo/internal/plan"
	"overgo/internal/processlock"
	"overgo/internal/testcheck"
)

// TestADriverStartsFromItsDefaultConfiguration holds what a driver start admits, in one fixture (SQA-338): in a
// directory that holds no store, no strategy and no plan, with no worker named in the environment, go run ./cmd/loop
// -init writes loop.json and its worker's prompt, naming Claude Code as the worker with the prompt on its standard
// input, its edits accepted and its permissions not bypassed, the lane's worker name and both budgets, and the prompt
// names skill.md and never stopping to ask; the driver starts from them, the pause marker ending the run at once, its
// campaign recording the name and the worker's name left out of the process's environment after; a second -init writes
// over nothing; and a config without its prompt file or its budgets, or one that still names a strategy, is refused.
func TestADriverStartsFromItsDefaultConfiguration(t *testing.T) {
	// Serial: it changes the process's directory and environment.
	t.Chdir(t.TempDir())
	t.Setenv(plan.AutomationWorkerEnvironment, "")
	testcheck.NoError(t, run([]string{"-init"}))
	state := processlock.StateDirectory
	var written config
	testcheck.NoError(t, jsonfile.DecodeStrict(filepath.Join(state, "loop.json"), &written))
	worker := strings.Join(written.Worker, " ")
	if written.Worker[0] != "claude" || !slices.Contains(written.Worker, "-p") || slices.Contains(written.Worker, "{prompt}") ||
		!strings.Contains(worker, "acceptEdits") || strings.Contains(worker, "bypass") || strings.Contains(worker, "dangerously") ||
		written.Name != defaultWorkerName || written.MaxAttemptsPerStep == 0 || written.MaxInvocations == 0 {
		t.Fatalf("-init wrote %+v; want Claude Code reading its prompt from standard input, edits accepted and nothing bypassed, the lane's name and both budgets", written)
	}
	if prompt := string(testcheck.Must(os.ReadFile(filepath.Join(state, written.Prompt)))(t)); !strings.Contains(prompt, "skill.md") || !strings.Contains(prompt, "never stop to ask") {
		t.Fatalf("the worker's prompt = %q; want it to name skill.md and never stopping to ask", prompt)
	}
	testcheck.NoError(t, os.WriteFile(filepath.Join(state, pauseMarker), nil, 0o600))
	absent := filepath.Join(t.TempDir(), "absent-store")
	testcheck.NoError(t, run([]string{"-repo", absent}))
	if locator := string(testcheck.Must(os.ReadFile(filepath.Join(state, "loop_supervisor.json")))(t)); !strings.Contains(locator, `"`+defaultWorkerName+`"`) {
		t.Fatalf("the campaign records %s, want the lane's worker name", locator)
	}
	if name := os.Getenv(plan.AutomationWorkerEnvironment); name != "" {
		t.Fatalf("the run left the worker's name %q in the process's environment", name)
	}
	if err := run([]string{"-init"}); err == nil || !strings.Contains(err.Error(), "writes over nothing") {
		t.Fatalf("a second -init = %v, want it refused", err)
	}
	for name, malformed := range map[string]string{
		"no prompt file": `{"worker": ["worker"], "name": "fixture-worker", "max_attempts_per_step": 1, "max_invocations": 1}`,
		"no budgets":     `{"worker": ["worker"], "name": "fixture-worker", "prompt": "loop_prompt.md"}`,
		"a strategy":     `{"worker": ["worker"], "name": "fixture-worker", "prompt": "loop_prompt.md", "strategy_id": "profile:sha256:00", "max_attempts_per_step": 1, "max_invocations": 1}`,
	} {
		testcheck.NoError(t, os.WriteFile(filepath.Join(state, "loop.json"), []byte(malformed), 0o600))
		if err := run([]string{"-repo", absent}); err == nil {
			t.Errorf("a config with %s started a run", name)
		}
	}
}
