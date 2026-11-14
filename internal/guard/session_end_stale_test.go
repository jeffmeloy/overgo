package guard

import (
	"os"
	"path/filepath"
	"testing"
	"time"

	"overgo/internal/processlock"
	"overgo/internal/testcheck"
)

// TestAnEndRecordOlderThanTheSessionsLastTurnIsNoEnd holds a recorded end to a sign of the session's end only while no
// turn of the session came after it (session-end/stale, SQA-384): a session that resumed writes its state at its next
// turn, so where its end record could not be removed, the state written after it outranks it and the session is live;
// with its state older than its end, the end stands.
func TestAnEndRecordOlderThanTheSessionsLastTurnIsNoEnd(t *testing.T) {
	t.Parallel()
	root := t.TempDir()
	const session = "resumed"
	testcheck.NoError(t, RecordSessionEnd(root, session))
	state := filepath.Join(root, processlock.StateDirectory)
	testcheck.NoError(t, os.WriteFile(filepath.Join(state, SessionState(session)), []byte("{}\n"), 0o600))
	ended := time.Now().Add(-time.Hour)
	testcheck.NoError(t, os.Chtimes(filepath.Join(state, sessionEnd(session)), ended, ended))
	turned := ended.Add(time.Minute)
	testcheck.NoError(t, os.Chtimes(filepath.Join(state, SessionState(session)), turned, turned))
	if SessionEnded(root, session) {
		t.Fatal("a session whose turn came after its recorded end reads as ended; want it live")
	}
	turned = ended.Add(-time.Minute)
	testcheck.NoError(t, os.Chtimes(filepath.Join(state, SessionState(session)), turned, turned))
	if !SessionEnded(root, session) {
		t.Fatal("a session whose last turn came before its recorded end reads as live; want it ended")
	}
}
