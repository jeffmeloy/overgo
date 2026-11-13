package main

import (
	"fmt"
	"testing"

	"overgo/internal/guard"
	"overgo/internal/testcheck"
)

// TestAStartedSessionDropsItsEndRecord holds the session-start hook to a session's current incarnation (SQA-384): a
// session whose end was recorded and which starts again is not ended, though it holds nothing, as a review session
// holds nothing, so its new scratch is never cleared as finished.
func TestAStartedSessionDropsItsEndRecord(t *testing.T) {
	sessionCheckout(t)
	const session = "restarted-session"
	testcheck.NoError(t, guard.RecordSessionEnd(".", session))
	runDoctrine(fmt.Sprintf(`{"session_id": %q}`, session))
	if guard.SessionEnded(".", session) {
		t.Fatal("a session that started again is ended by its last incarnation's record")
	}
}
