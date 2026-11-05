package main

import (
	"io"
	"os"
	"strings"
	"testing"

	"overgo/internal/plan"
	"overgo/internal/testcheck"
)

// TestAStopThatPassedOnRetryIsReportedNextTurn keeps the retry valve, which lets a session stuck where its block found
// it end the turn so the gate never wedges it, and reports what it let through: the next prompt says the last turn
// ended with its block unresolved, naming the row, once. A prompt that asks to stop is kept as the request a requested
// stop needs, and a later prompt that does not clears it.
func TestAStopThatPassedOnRetryIsReportedNextTurn(t *testing.T) {
	// Serial: it changes the working directory and the hooks' environment (sessionCheckout).
	sessionCheckout(t)
	prompted := func(prompt string) string {
		t.Helper()
		read, written, err := os.Pipe()
		testcheck.NoError(t, err)
		saved := os.Stdout
		os.Stdout = written
		runPrompt(`{"session_id": "lead", "prompt": "` + prompt + `"}`)
		os.Stdout = saved
		_ = written.Close()
		printed, err := io.ReadAll(read)
		testcheck.NoError(t, err)
		return string(printed)
	}
	if got := stopHook("lead"); got != 2 {
		t.Fatalf("a stop with the plan's row open = %d, want the block", got)
	}
	if got := runStop(`{"session_id": "lead", "stop_hook_active": true}`); got != 0 {
		t.Fatalf("a retry with nothing changed since the block = %d, want the valve to let it through", got)
	}
	if printed := prompted("progress?"); !strings.Contains(printed, "STOP PASSED") || !strings.Contains(printed, stopContinue+": ") {
		t.Fatalf("the prompt after a stop the valve passed printed %q, want the block it passed", printed)
	}
	if printed := prompted("progress?"); strings.Contains(printed, "STOP PASSED") {
		t.Fatalf("the block a stop passed was reported a second time: %q", printed)
	}
	prompted("what is left? stop after this")
	if err := plan.RequireStopRequest("."); err != nil {
		t.Fatalf("a prompt that asked to stop left no request: %v", err)
	}
	prompted("progress?")
	if err := plan.RequireStopRequest("."); err == nil {
		t.Fatal("a prompt that asked for no stop left the earlier request standing")
	}
}
