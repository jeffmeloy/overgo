package main

import (
	"path/filepath"
	"strings"
	"testing"
	"time"

	"overgo/internal/scorecard"
)

// TestPushHeldPushesTheHeldCommitOnlyWhenAsked holds the owner's push to what it says: the
// commit the newest held scorecard names reaches the remote's main, not the checkout's HEAD or
// a later commit that was measured but did not hold; nothing is pushed without the explicit
// confirmation or from a worker's environment; a remote main the held commit does not contain
// is refused, not overwritten; and no held scorecard means nothing to push.
func TestPushHeldPushesTheHeldCommitOnlyWhenAsked(t *testing.T) {
	t.Parallel()
	repository, remote, records := t.TempDir(), t.TempDir(), filepath.Join(t.TempDir(), "scorecards")
	git := func(directory string, arguments ...string) string { return gitFixture(t, directory, arguments...) }
	git(remote, "init", "-q", "--bare")
	git(repository, "init", "-q", "-b", "master")
	git(repository, "commit", "-q", "--allow-empty", "-m", "first")
	first := git(repository, "rev-parse", "HEAD")
	git(repository, "remote", "add", "origin", remote)
	git(repository, "push", "-q", "origin", "master:main")
	git(repository, "commit", "-q", "--allow-empty", "-m", "held")
	held := git(repository, "rev-parse", "HEAD")
	git(repository, "commit", "-q", "--allow-empty", "-m", "measured, not held")
	measured := git(repository, "rev-parse", "HEAD")
	remoteMain := func() string { return git(remote, "rev-parse", "main") }

	if _, err := pushHeld(repository, records, "origin", true, false); err == nil || !strings.Contains(err.Error(), "no scorecard is recorded as held") {
		t.Fatalf("pushing with nothing held = %v, want it refused", err)
	}
	published := time.Date(2026, time.October, 5, 12, 0, 0, 0, time.UTC)
	for index, commit := range []string{first, held, measured} {
		if _, err := scorecard.Publish(records, published.Add(time.Duration(index)*time.Hour), scorecard.Record{Commit: commit}); err != nil {
			t.Fatal(err)
		}
	}
	scored, err := scorecard.Records(records)
	if err != nil {
		t.Fatal(err)
	}
	if err := scorecard.MarkHeld(records, scored[1]); err != nil {
		t.Fatal(err)
	}

	// Without the confirmation, or from a worker, nothing is pushed.
	for name, push := range map[string]func() (string, error){
		"without confirmation": func() (string, error) { return pushHeld(repository, records, "origin", false, false) },
		"from a worker":        func() (string, error) { return pushHeld(repository, records, "origin", true, true) },
	} {
		if _, err := push(); err == nil || remoteMain() != first {
			t.Fatalf("%s: err = %v, remote main %s; want a refusal and the remote at %s", name, err, remoteMain(), first)
		}
	}

	// Asked, the held commit is pushed: not HEAD, which is the later commit that did not hold.
	pushed, err := pushHeld(repository, records, "origin", true, false)
	if err != nil || pushed != held || remoteMain() != held {
		t.Fatalf("pushed %q (%v), remote main %s; want the held commit %s and not HEAD %s", pushed, err, remoteMain(), held, measured)
	}
	if again, err := pushHeld(repository, records, "origin", true, false); err != nil || again != held {
		t.Fatalf("pushing the commit the remote already holds = (%q, %v), want it a no-op", again, err)
	}

	// A remote main the held commit does not contain is refused, not overwritten.
	other := t.TempDir()
	git(other, "init", "-q", "-b", "elsewhere")
	git(other, "commit", "-q", "--allow-empty", "-m", "unrelated")
	unrelated := git(other, "rev-parse", "HEAD")
	git(other, "remote", "add", "origin", remote)
	git(other, "push", "-q", "--force", "origin", "elsewhere:main")
	git(repository, "fetch", "-q", "origin")
	if _, err := pushHeld(repository, records, "origin", true, false); err == nil || remoteMain() != unrelated {
		t.Fatalf("pushing over a main the held commit does not contain = %v, remote main %s; want a refusal and the remote left at %s", err, remoteMain(), unrelated)
	}
}
