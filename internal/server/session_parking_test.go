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

// acquireSession leases a session without waiting, as a test holding a slot.
func (h *Handler) acquireSession(requested int) (*requestSession, bool) {
	lease, ok := h.sessions.TryLease(requested)
	if !ok {
		return nil, false
	}
	h.activateSession(lease)
	return lease, true
}
