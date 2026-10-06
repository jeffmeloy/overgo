package main

import (
	"strings"
	"testing"
)

// TestPushPushesTheNamedCommitOnlyWhenAsked holds the owner's push to what it says: the commit
// named reaches the remote's main, not the checkout's HEAD or a later commit; nothing is pushed
// without the explicit confirmation or from a worker's environment; a name that is no commit is
// refused; and a remote main the named commit does not contain is refused, not overwritten.
func TestPushPushesTheNamedCommitOnlyWhenAsked(t *testing.T) {
	t.Parallel()
	repository, remote := t.TempDir(), t.TempDir()
	git := func(directory string, arguments ...string) string { return gitFixture(t, directory, arguments...) }
	git(remote, "init", "-q", "--bare")
	git(repository, "init", "-q", "-b", "master")
	git(repository, "commit", "-q", "--allow-empty", "-m", "first")
	first := git(repository, "rev-parse", "HEAD")
	git(repository, "remote", "add", "origin", remote)
	git(repository, "push", "-q", "origin", "master:main")
	git(repository, "commit", "-q", "--allow-empty", "-m", "named")
	named := git(repository, "rev-parse", "HEAD")
	git(repository, "commit", "-q", "--allow-empty", "-m", "later")
	later := git(repository, "rev-parse", "HEAD")
	remoteMain := func() string { return git(remote, "rev-parse", "main") }

	// Without the confirmation, from a worker, or with a name that is no commit, nothing is pushed.
	for name, push := range map[string]func() (string, error){
		"without confirmation": func() (string, error) { return pushNamed(repository, named, "origin", false, false) },
		"from a worker":        func() (string, error) { return pushNamed(repository, named, "origin", true, true) },
		"a name that is no commit": func() (string, error) {
			return pushNamed(repository, "no-such-commit", "origin", true, false)
		},
	} {
		if _, err := push(); err == nil || remoteMain() != first {
			t.Fatalf("%s: err = %v, remote main %s; want a refusal and the remote at %s", name, err, remoteMain(), first)
		}
	}

	// Asked, the named commit is pushed: not HEAD, which is the later commit.
	pushed, err := pushNamed(repository, named, "origin", true, false)
	if err != nil || pushed != named || remoteMain() != named {
		t.Fatalf("pushed %q (%v), remote main %s; want the named commit %s and not HEAD %s", pushed, err, remoteMain(), named, later)
	}
	if again, err := pushNamed(repository, named, "origin", true, false); err != nil || again != named {
		t.Fatalf("pushing the commit the remote already holds = (%q, %v), want it a no-op", again, err)
	}

	// A remote main the named commit does not contain is refused, not overwritten.
	other := t.TempDir()
	git(other, "init", "-q", "-b", "elsewhere")
	git(other, "commit", "-q", "--allow-empty", "-m", "unrelated")
	unrelated := git(other, "rev-parse", "HEAD")
	git(other, "remote", "add", "origin", remote)
	git(other, "push", "-q", "--force", "origin", "elsewhere:main")
	git(repository, "fetch", "-q", "origin")
	if _, err := pushNamed(repository, later, "origin", true, false); err == nil || !strings.Contains(err.Error(), "fast-forwards") || remoteMain() != unrelated {
		t.Fatalf("pushing over a main the named commit does not contain = %v, remote main %s; want a refusal and the remote left at %s", err, remoteMain(), unrelated)
	}
}
