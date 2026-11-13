package main

import (
	"bytes"
	"path/filepath"
	"strings"
	"testing"

	"overgo/internal/dataroot"
	"overgo/internal/jsonfile"
	"overgo/internal/loop"
	"overgo/internal/plan"
	"overgo/internal/testcheck"
)

// TestTheScheduledSessionPromptWorksThePlan holds what a session the Claude app's scheduled task starts is told, since
// the standalone worker has no sign-in on this machine: the plan's three commands and skill.md, never stopping to ask,
// ending at once when another session holds the worktree, and the environment its commands run with, the lane's
// worker, data root and role from loop.json. A machine with no loop.json is told how to write it.
func TestTheScheduledSessionPromptWorksThePlan(t *testing.T) {
	t.Parallel()
	path := filepath.Join(t.TempDir(), "loop.json")
	var said bytes.Buffer
	if err := writeSessionPrompt(path, &said); err == nil || !strings.Contains(err.Error(), "-init") {
		t.Fatalf("a machine with no configuration = %v, want it told to write one with -init", err)
	}
	testcheck.NoError(t, jsonfile.Write(path, config{Worker: defaultWorker, Name: defaultWorkerName, Prompt: defaultPromptFile,
		Environment: map[string]string{dataroot.Env: "D:/overgo_data", plan.AutomationRoleEnvironment: "master-lead"},
		Config:      loop.Config{MaxAttemptsPerStep: defaultAttempts, MaxInvocations: defaultInvocations}}, defaultConfigMode))
	testcheck.NoError(t, writeSessionPrompt(path, &said))
	printed := said.String()
	for _, want := range []string{
		"skill.md", "go run ./cmd/plan -next", "go run ./cmd/plan -verify", "go run ./cmd/loop -land <item>/<step>", "Never stop to ask",
		"another session holds this worktree, end this run at once",
		plan.AutomationWorkerEnvironment + "=" + defaultWorkerName, dataroot.Env + "=D:/overgo_data", plan.AutomationRoleEnvironment + "=master-lead",
	} {
		if !strings.Contains(printed, want) {
			t.Errorf("the scheduled session's instructions hold no %q:\n%s", want, printed)
		}
	}
}

// TestEveryWorkerIsToldToFindTheCause holds what the worker and a scheduled session are told to do with a refusal or a
// break to the owner's method of 2026-11-12: find its cause in the code or by reproducing it before changing anything,
// and cure it by removing what failed first. Both had said to do what the refusal says and keep going, which patched a
// symptom and built the next mechanism on it.
func TestEveryWorkerIsToldToFindTheCause(t *testing.T) {
	for name, said := range map[string]string{"the worker's prompt": defaultPrompt, "a scheduled session's instructions": sessionInstructions} {
		if !strings.Contains(said, "find its cause in the code or by reproducing it before changing anything") ||
			!strings.Contains(said, "removing what failed, else fixing what exists, adding last") || strings.Contains(said, "keep going") {
			t.Errorf("%s does not say to find a refusal's cause and remove what failed first: %q", name, said)
		}
	}
}
