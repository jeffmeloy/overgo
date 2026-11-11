package main

import (
	"os"
	"path/filepath"
	"slices"
	"testing"

	"overgo/internal/guard"
	"overgo/internal/processlock"
	"overgo/internal/testcheck"
)

// TestSessionScratchIsListedUnderEveryHarnessFolder holds the session line to every folder the harness keeps this
// checkout's sessions in (session-leftovers/every-harness-folder): the folder named for the checkout's path lists each
// session in it, as before, and every other folder of the temporary directory's claude folder lists the sessions this
// checkout recorded, by their loop_state-<session> files, and no other: another checkout's session beside them is not
// this checkout's to list.
func TestSessionScratchIsListedUnderEveryHarnessFolder(t *testing.T) {
	t.Parallel()
	root := t.TempDir()
	at := func(parts ...string) string { return filepath.Join(append([]string{root}, parts...)...) }
	for _, folder := range []string{
		at("temporary", "claude", "C--checkout", "named"),
		at("temporary", "claude", "a1b2c3", "recorded"),
		at("temporary", "claude", "a1b2c3", "another-checkouts"),
		at("checkout", processlock.StateDirectory),
	} {
		testcheck.NoError(t, os.MkdirAll(folder, 0o700))
	}
	testcheck.NoError(t, os.WriteFile(at("checkout", processlock.StateDirectory, guard.SessionState("recorded")), []byte("{}\n"), 0o600))
	line := leftoverPlaces[slices.IndexFunc(leftoverPlaces, func(line place) bool { return line.name == "session-scratch" })]
	f := &facts{hygiene: hygiene{checkout: at("checkout"), sessions: at("temporary", "claude", "C--checkout"), temporary: at("temporary")}}
	var listed []string
	for _, entry := range sessionFolders(f, line) {
		listed = append(listed, entry.path)
	}
	slices.Sort(listed)
	if want := []string{at("temporary", "claude", "C--checkout", "named"), at("temporary", "claude", "a1b2c3", "recorded")}; !slices.Equal(listed, slices.Sorted(slices.Values(want))) {
		t.Fatalf("the session line listed %v; want the path-named folder's session and the recorded session under the hash-named folder, and no other", listed)
	}
}
