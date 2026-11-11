package main

import (
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"

	"overgo/internal/guard"
	"overgo/internal/processlock"
)

// TestAnEndedSessionsScratchIsFinished holds the hook's half of session-leftovers/ended-session: the SessionEnd hook,
// run as the harness runs it, records in the checkout's process state that the session it is called for ended, beside
// that session's state, the one sign the leftovers' session line finishes its scratch folder by. The line's half is
// the test of this name in cmd/loop.
func TestAnEndedSessionsScratchIsFinished(t *testing.T) {
	t.Parallel()
	built := filepath.Join(t.TempDir(), "loophook.exe")
	if output, err := exec.Command("go", "build", "-o", built, ".").CombinedOutput(); err != nil {
		t.Fatalf("building the hook: %v: %s", err, output)
	}
	checkout := t.TempDir()
	hook := exec.Command(built, "session-end")
	hook.Dir, hook.Stdin = checkout, strings.NewReader(`{"session_id": "ended-one", "hook_event_name": "SessionEnd"}`)
	output, _ := hook.CombinedOutput()
	record, err := os.ReadFile(filepath.Join(checkout, processlock.StateDirectory, guard.SessionState("ended-one")+"-ended"))
	if err != nil || len(strings.TrimSpace(string(record))) == 0 {
		t.Fatalf("the session-end hook said %q and recorded %q (%v); want the session's end recorded beside its state", output, record, err)
	}
}
