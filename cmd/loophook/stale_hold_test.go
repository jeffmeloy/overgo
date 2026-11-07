package main

import (
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"testing"

	"overgo/internal/guard"
	"overgo/internal/processlock"
	"overgo/internal/testcheck"
)

// TestAHoldWhoseProcessEndedIsTaken holds the worktree's hold to the sessions that still run, as the owner put it on
// 2026-11-07, "'take over this worktree' is bullshit, you are already in control", after two of the driver's workers
// took the hold and ended. A hold naming a session whose process has exited is taken by the next implementing session,
// with no declaration typed, and records that session's process; a hold whose process runs sends the next session
// away; a hold written before the process was recorded is left as it was; and the session that holds the worktree
// keeps its process current across a restart.
func TestAHoldWhoseProcessEndedIsTaken(t *testing.T) {
	// Serial: the hook reads the checkout's process state from the process's directory and the session's process
	// from the environment.
	sessionCheckout(t)
	hold := filepath.Join(processlock.StateDirectory, guard.WorktreeHolder)
	ended := exec.CommandContext(t.Context(), "git", "--version")
	testcheck.NoError(t, ended.Run())
	running := os.Getpid()
	t.Setenv(claudeProcessVariable, strconv.Itoa(running))
	recordImplementer("lead")
	written := func(content string) {
		t.Helper()
		testcheck.NoError(t, os.WriteFile(hold, []byte(content), stateFileMode))
	}
	held := func() (string, int) {
		t.Helper()
		return guard.ReadHold(testcheck.Must(os.ReadFile(hold))(t))
	}

	written(guard.HoldText("ended-worker", ended.Process.Pid))
	if holder := holdWorktree("lead", false); holder != "" {
		t.Fatalf("a hold whose session's process exited sent the lead away to %q", holder)
	}
	if session, process := held(); session != "lead" || process != running {
		t.Fatalf("after taking a stale hold the hold names %q, process %d; want the lead and its process %d", session, process, running)
	}

	written(guard.HoldText("live-session", running))
	if holder := holdWorktree("lead", false); holder != "live-session" {
		t.Fatalf("a hold whose session runs was taken: the lead was told %q", holder)
	}

	written("older-session")
	if holder := holdWorktree("lead", false); holder != "older-session" {
		t.Fatalf("a hold that names no process was taken: the lead was told %q", holder)
	}

	written(guard.HoldText("lead", ended.Process.Pid))
	if holder := holdWorktree("lead", false); holder != "" {
		t.Fatalf("the lead was sent away from its own hold, to %q", holder)
	}
	if _, process := held(); process != running {
		t.Fatalf("the lead's hold names process %d after its restart; want its process now, %d", process, running)
	}
}
