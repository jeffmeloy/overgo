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

// TestTheDriverLandsAStepThroughItsWorker holds what the driver needs to land a step through its worker on this
// machine, since it runs every step through go run ./cmd/plan in the real module and no fixture repository exercises
// it end to end (the step's note); the driver working the next step is the end-to-end proof. Its configuration is the
// defaults the repository holds:
// go run ./cmd/loop -init writes loop.json and its worker's prompt where neither stands, naming Claude Code as the
// worker with the prompt on its standard input, its edits accepted and its permissions not bypassed, the lane's worker
// name and both budgets; the driver starts from them, the pause marker ending the run at once, its campaign recording
// the name; and a second -init writes over nothing.
func TestTheDriverLandsAStepThroughItsWorker(t *testing.T) {
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
	testcheck.NoError(t, run([]string{"-repo", filepath.Join(t.TempDir(), "absent-store")}))
	if locator := string(testcheck.Must(os.ReadFile(filepath.Join(state, "loop_supervisor.json")))(t)); !strings.Contains(locator, `"`+defaultWorkerName+`"`) {
		t.Fatalf("the campaign records %s, want the lane's worker name", locator)
	}
	if err := run([]string{"-init"}); err == nil || !strings.Contains(err.Error(), "writes over nothing") {
		t.Fatalf("a second -init = %v, want it refused", err)
	}
}
