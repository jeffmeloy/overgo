package guard

import (
	"os"
	"path/filepath"
	"testing"

	"overgo/internal/processlock"
	"overgo/internal/testcheck"
)

// TestASessionThatResumesIsNotEnded holds a session's end to its current incarnation (SQA-384): a session whose end
// was recorded and which then holds the checkout with a running process, as a resumed session does, is not ended, so
// its new scratch is never cleared as finished. With no hold naming it, its recorded end stands; with the hold's process
// gone, it has ended.
func TestASessionThatResumesIsNotEnded(t *testing.T) {
	t.Parallel()
	root := t.TempDir()
	const session = "resumed-session"
	testcheck.NoError(t, RecordSessionEnd(root, session))
	if !SessionEnded(root, session) {
		t.Fatal("a session whose end is recorded and which holds nothing is not ended")
	}
	hold := filepath.Join(root, processlock.StateDirectory, WorktreeHolder)
	testcheck.NoError(t, os.WriteFile(hold, []byte(HoldText(session, os.Getpid())), 0o600))
	if SessionEnded(root, session) {
		t.Fatal("a session that holds the checkout with a running process is ended by its last incarnation's record")
	}
	testcheck.NoError(t, os.WriteFile(hold, []byte(session+"\n2147483646\n1"), 0o600))
	if !SessionEnded(root, session) {
		t.Fatal("a session whose hold names a process that has exited is not ended")
	}
}
