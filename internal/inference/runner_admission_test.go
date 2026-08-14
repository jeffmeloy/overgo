package inference

import (
	"errors"
	"testing"
)

func TestRunnerLockOpenRejectsUnavailableRunner(t *testing.T) {
	var missing *Runner
	if err := missing.lockOpen(); !errors.Is(err, errRunnerNil) {
		t.Fatalf("missing runner error = %v", err)
	}

	closed := &Runner{runnerState: runnerState{closed: true}}
	if err := closed.lockOpen(); !errors.Is(err, errRunnerClosed) {
		t.Fatalf("closed runner error = %v", err)
	}
	if !closed.mu.TryLock() {
		t.Fatal("closed admission retained runner lock")
	}
	closed.mu.Unlock()
}
