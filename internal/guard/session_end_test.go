package guard

import (
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"testing"

	"overgo/internal/processlock"
	"overgo/internal/testcheck"
)

// TestASessionsEndIsRecordedAndRead holds the end of a session, the one sign the leftovers' session line finishes its
// scratch folder by, to a positive sign alone: a recorded end, which the session-end hook and cmd/loop -end-session
// write, or the checkout's hold naming the session with a process that has exited. A session with neither has not
// ended, whatever else is so, a hold naming a running process among them; a session not named by one path element has
// no end recorded.
func TestASessionsEndIsRecordedAndRead(t *testing.T) {
	t.Parallel()
	root := t.TempDir()
	if SessionEnded(root, "quiet") {
		t.Fatal("a session with no record and no hold has ended; want it running")
	}
	testcheck.NoError(t, RecordSessionEnd(root, "recorded"))
	if !SessionEnded(root, "recorded") || SessionEnded(root, "quiet") {
		t.Fatal("a recorded end was not read, or was read for another session")
	}
	if err := RecordSessionEnd(root, filepath.Join("a", "b")); err == nil {
		t.Fatal("an end was recorded for a session named by two path elements")
	}
	hold := filepath.Join(root, processlock.StateDirectory, WorktreeHolder)
	testcheck.NoError(t, os.WriteFile(hold, []byte("held\n"+strconv.Itoa(os.Getpid())+"\n"), 0o600))
	if SessionEnded(root, "held") {
		t.Fatal("a session whose hold names a running process has ended")
	}
	exited := exec.Command("git", "--version")
	testcheck.NoError(t, exited.Run())
	testcheck.NoError(t, os.WriteFile(hold, []byte("held\n"+strconv.Itoa(exited.Process.Pid)+"\n"), 0o600))
	if !SessionEnded(root, "held") || SessionEnded(root, "quiet") {
		t.Fatal("a hold whose process exited did not end its session alone")
	}
}
