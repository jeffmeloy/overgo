package guard

import (
	"io/fs"
	"os"
	"path/filepath"
	"testing"

	"overgo/internal/processlock"
	"overgo/internal/testcheck"
)

// TestAnUnknownSessionEndIsNoEnd holds a session whose end the checkout's records cannot tell to no end
// (session-end/unknown, SQA-384's second criterion, unavailable ownership held or unknown): a recorded end beside a hold
// that cannot be read, which may name the session, leaves the session unended, where an absent hold lets the end
// stand.
func TestAnUnknownSessionEndIsNoEnd(t *testing.T) {
	t.Parallel()
	root := t.TempDir()
	const session = "unknown"
	testcheck.NoError(t, RecordSessionEnd(root, session))
	if !SessionEnded(root, session) {
		t.Fatal("a recorded end with no hold and no later turn reads as live; want it ended")
	}
	// A directory where the hold is kept cannot be read as a file, as an unreadable hold cannot.
	testcheck.NoError(t, os.Mkdir(filepath.Join(root, processlock.StateDirectory, WorktreeHolder), 0o700))
	if SessionEnded(root, session) {
		t.Fatal("a session beside a hold that cannot be read reads as ended; want its end unknown, so unended")
	}
}

// TestAnUnreadableHoldHoldsTheCheckout holds a checkout whose hold cannot be read to held (session-unknown/do, found
// where else SQA-384's fault occurs): heldBy read an unreadable hold as no hold, so another session's checkout read as
// free to land and remove in.
func TestAnUnreadableHoldHoldsTheCheckout(t *testing.T) {
	t.Parallel()
	root := t.TempDir()
	testcheck.NoError(t, os.MkdirAll(filepath.Join(root, processlock.StateDirectory, WorktreeHolder), 0o700))
	if holder := heldBy(root, "this"); holder == "" {
		t.Fatal("a checkout whose hold cannot be read reads as held by no session")
	}
}

// TestAnUnreadableSessionStateIsNoEnd holds a session whose state cannot be read to no end (SQA-384): its state may
// hold a turn after its recorded end, so the end is unknown. A state that is not there lets the end stand.
func TestAnUnreadableSessionStateIsNoEnd(t *testing.T) {
	root := t.TempDir()
	const session = "unreadable"
	testcheck.NoError(t, RecordSessionEnd(root, session))
	state := filepath.Join(root, processlock.StateDirectory, SessionState(session))
	read := statSession
	t.Cleanup(func() { statSession = read })
	statSession = func(path string) (os.FileInfo, error) {
		if path == state {
			return nil, &fs.PathError{Op: "stat", Path: path, Err: fs.ErrPermission}
		}
		return read(path)
	}
	if SessionEnded(root, session) {
		t.Fatal("a session whose state cannot be read reads as ended; want its end unknown, so unended")
	}
}
