package landing

import (
	"errors"
	"io"
	"os"
	"path/filepath"
	"runtime"
	"sync"
	"testing"
	"time"

	"overgo/internal/clioptions"
	"overgo/internal/jsonfile"
	"overgo/internal/processlock"
)

// TestCommitsOfTwoWorktreesTakeTurns holds the commits begun in a
// repository's worktrees to one turn.
//
// The trunk is the main worktree and the branch it has checked out, the same
// read from the main worktree and from a linked one. A commit that holds the
// turn keeps a commit begun in the other worktree waiting: it says so when it
// starts to wait, takes the turn when the first gives it back, and records
// how long it waited, where the first, which found the turn free, records no
// wait. And a whole commit takes the turn and gives it back: one that lands
// in the linked worktree has the turn when its first stage reports, records
// no wait, and after it the turn is free.
func TestCommitsOfTwoWorktreesTakeTurns(t *testing.T) {
	t.Parallel()
	ctx := t.Context()
	fixture := newRowFixture(t, map[string]string{"a.txt": "base\n", "internal/p/p.go": "// Package p is the fixture.\npackage p\n"})
	fixture.git("branch", "-M", "trunk")
	// The repository's files are the bytes git holds, as this repository's are: a checkout rewrites none.
	fixture.git("config", "core.autocrlf", "false")
	linked := &sliceFixture{t: t, repo: filepath.Join(t.TempDir(), "linked")}
	fixture.git("worktree", "add", "-q", "--detach", linked.repo, "HEAD")
	main, err := os.Stat(fixture.repo)
	if err != nil {
		t.Fatal(err)
	}
	for _, checkout := range []*sliceFixture{fixture, linked} {
		root, branch, err := trunkOf(ctx, checkout.repo)
		if err != nil || branch != "refs/heads/trunk" {
			t.Fatalf("read from %s the trunk is %q on %q (%v), want the main worktree on its branch", checkout.repo, root, branch, err)
		}
		if found, err := os.Stat(root); err != nil || !os.SameFile(found, main) {
			t.Fatalf("read from %s the trunk's checkout is %s (%v), want the main worktree %s", checkout.repo, root, err, fixture.repo)
		}
	}

	first := &landing{Options: Options{Repo: fixture.repo, Log: io.Discard}}
	held, err := first.holdTurn(ctx)
	if err != nil || first.spent.Queue != 0 {
		t.Fatalf("the first commit took the turn after waiting %s (%v), want it free", first.spent.Queue, err)
	}
	waiting := make(chan struct{})
	said := sync.OnceFunc(func() { close(waiting) })
	second := &landing{Options: Options{Repo: linked.repo, Log: firstWrite{said}}}
	var group sync.WaitGroup
	group.Go(func() {
		<-waiting
		// The first ends once the clock the second reads has moved: this host's
		// clock ticks, and a wait within one tick reads as none.
		for began := time.Now(); !time.Now().After(began); {
			runtime.Gosched()
		}
		if err := held(); err != nil {
			t.Error(err)
		}
	})
	release, err := second.holdTurn(ctx)
	// A second that never said it waits took the turn beside the first.
	select {
	case <-waiting:
	default:
		err = errors.Join(err, errors.New("it took the turn while the first held it"))
	}
	said()
	group.Wait()
	if release != nil {
		err = errors.Join(err, release())
	}
	if err != nil || second.spent.Queue <= 0 {
		t.Fatalf("a commit begun in the other worktree while the first held the turn waited %s (%v), want it to wait for the first and record the wait", second.spent.Queue, err)
	}

	linked.write("a.txt", "changed in the linked worktree\n")
	message := []byte("Change the fixture\n\nCause: the fixture needs a change.\nPredicted effect: none.\n")
	// When the commit's first stage reports, the turn is taken: no one else has it.
	during := false
	reported := firstWrite{sync.OnceFunc(func() {
		taken, err := processlock.Acquire(filepath.Join(fixture.repo, processlock.StateDirectory, turnLock), clioptions.PrivateFileMode)
		if during = errors.Is(err, processlock.ErrBusy); err == nil {
			_ = taken.Close()
		}
	})}
	landed, err := Run(ctx, Options{Repo: linked.repo, Ref: "row/do", Message: message, Log: reported})
	if err != nil || landed == "" || linked.git("rev-parse", "HEAD") != landed || !during {
		t.Fatalf("a commit in the linked worktree = %q (%v), the turn held while its stages ran %t; want it landed there under the turn", landed, err, during)
	}
	var runs []spent
	if err := jsonfile.Decode(filepath.Join(linked.repo, processlock.StateDirectory, spentRecord), &runs); err != nil || len(runs) != 1 || runs[0].Outcome != outcomeLanded || runs[0].Queue != 0 {
		t.Fatalf("the commit's record is %+v (%v), want one landed run that waited for no one", runs, err)
	}
	after := &landing{Options: Options{Repo: fixture.repo, Log: io.Discard}}
	free, err := after.holdTurn(ctx)
	if err != nil || after.spent.Queue != 0 {
		t.Fatalf("after the commit the turn was taken after waiting %s (%v), want it given back", after.spent.Queue, err)
	}
	if err := free(); err != nil {
		t.Fatal(err)
	}
}
