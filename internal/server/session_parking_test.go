package server

import (
	"runtime"
	"testing"
)

func waitForServerSessionWaiter(t *testing.T, handler *Handler) {
	t.Helper()
	// The parked request shows in the director's snapshot once its
	// goroutine has yielded; the scheduler, not a clock, paces the look.
	for handler.sessions.Snapshot().Waiting == 0 {
		runtime.Gosched()
	}
}
