package main

import (
	"path/filepath"
	"testing"

	"overgo/internal/processlock"
	"overgo/internal/testcheck"
)

// TestAReplayWorksInTheOneReplayCheckout holds every replay, a test's among them, to the one replay checkout under the
// repository's process state: a hold names it, a second replay is refused while the first holds it, and the next holds
// it once the first lets it go. A test's replay had worked in a checkout of its own and left 12 worktrees when go
// test's timeout ended it before its cleanup ran (replay-checkout/do).
func TestAReplayWorksInTheOneReplayCheckout(t *testing.T) {
	t.Parallel()
	root := t.TempDir()
	first := testcheck.Must(holdReplayCheckout(root))(t)
	if want := filepath.Join(root, processlock.StateDirectory, replayCheckout); first.checkout != want {
		t.Fatalf("the replay works in %s, want %s", first.checkout, want)
	}
	if second, err := holdReplayCheckout(root); err == nil {
		testcheck.NoError(t, second.lock.Close())
		t.Fatal("a second replay held the replay checkout while the first held it")
	}
	testcheck.NoError(t, first.lock.Close())
	next := testcheck.Must(holdReplayCheckout(root))(t)
	testcheck.NoError(t, next.lock.Close())
}
