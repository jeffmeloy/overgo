package main

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"overgo/internal/guard"
	"overgo/internal/jsonfile"
	"overgo/internal/plan"
	"overgo/internal/processlock"
	"overgo/internal/testcheck"
)

// TestTheDriverRefusesAWorkerPromptThatDeclaresATakeover holds the driver's worker to the hold it is given (SQA-337):
// the default prompt opened with the takeover declaration, which the hook reads as a session taking the main
// checkout, so every worker the driver started overwrote a working session's hold. -init's prompt opens with no
// takeover, and a configured prompt that opens with one is refused before any campaign begins, naming its file.
func TestTheDriverRefusesAWorkerPromptThatDeclaresATakeover(t *testing.T) {
	// Serial: it changes the process's directory and environment.
	t.Chdir(t.TempDir())
	t.Setenv(plan.AutomationWorkerEnvironment, "")
	testcheck.NoError(t, run([]string{"-init"}))
	state := processlock.StateDirectory
	var written config
	testcheck.NoError(t, jsonfile.DecodeStrict(filepath.Join(state, "loop.json"), &written))
	path := filepath.Join(state, written.Prompt)
	prompt := string(testcheck.Must(os.ReadFile(path))(t))
	if strings.HasPrefix(strings.ToLower(prompt), guard.TakeoverDeclaration) {
		t.Fatalf("-init's worker prompt opens with a takeover: %q", prompt)
	}
	testcheck.NoError(t, os.WriteFile(path, []byte("Take over this worktree. "+prompt), 0o600))
	if err := run([]string{"-repo", filepath.Join(t.TempDir(), "absent-store")}); err == nil || !strings.Contains(err.Error(), written.Prompt) {
		t.Fatalf("a prompt that opens with a takeover started the driver: %v; want it refused naming %s", err, written.Prompt)
	}
	if _, err := os.Stat(filepath.Join(state, "loop_supervisor.json")); err == nil {
		t.Fatal("the refused driver began a campaign")
	}
}
