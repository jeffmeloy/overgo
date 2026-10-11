package landing

import (
	"context"
	"errors"
	"fmt"
	"path/filepath"
	"strings"
	"time"

	"overgo/internal/clioptions"
	"overgo/internal/gitauthority"
	"overgo/internal/processlock"
)

// turnLock is the lock of a repository's one commit turn, in the process
// state of its trunk's checkout.
const turnLock = "landing_turn.lock"

// trunkOf reads a repository's trunk from git's list of its worktrees: the
// directory of the main worktree, which git lists first, and the branch it
// has checked out, none when it is on no branch. It is the same from whichever
// worktree of the repository it is read, which is what makes it the one place
// their commits take turns and the one branch they publish onto.
func trunkOf(ctx context.Context, repo string) (root, branch string, err error) {
	listed, err := gitauthority.Query(ctx, repo, "worktree", "list", "--porcelain")
	if err != nil {
		return "", "", err
	}
	for line := range strings.Lines(string(listed)) {
		key, value, _ := strings.Cut(strings.TrimSpace(line), " ")
		switch {
		case key == "" && root != "":
			return root, branch, nil
		case key == "worktree" && root == "":
			root = filepath.FromSlash(value)
		case key == "branch":
			branch = value
		}
	}
	if root == "" {
		return "", "", errors.New("land: git lists no worktree for this repository")
	}
	return root, branch, nil
}

// holdTurn takes the repository's one turn for a commit's whole run. Of the
// commits begun in its worktrees one runs and the others wait here, each with
// its candidate in its own worktree, so no two check trees that only one of
// them can publish onto the trunk as it stands. The turn is held until the
// run ends, and the wait for it is the run's to record. A repository with one
// worktree has no one to wait for.
func (l *landing) holdTurn(ctx context.Context) (release func() error, err error) {
	root, _, err := trunkOf(ctx, l.Repo)
	if err != nil {
		return nil, err
	}
	if err := processlock.EnsureStateDirectory(root); err != nil {
		return nil, err
	}
	path := filepath.Join(root, processlock.StateDirectory, turnLock)
	lock, err := processlock.Acquire(path, clioptions.PrivateFileMode)
	if errors.Is(err, processlock.ErrBusy) {
		fmt.Fprintln(l.Log, "land: waiting for the commit running in another worktree of this repository to finish")
		started := time.Now()
		if lock, err = processlock.AcquireContext(ctx, path, clioptions.PrivateFileMode); err == nil {
			l.spent.Queue = time.Since(started)
			fmt.Fprintf(l.Log, "land: waited %s for another worktree's commit\n", l.spent.Queue.Round(time.Millisecond))
			l.measured = append(l.measured, "queue wait "+l.spent.Queue.Round(time.Millisecond).String())
		}
	}
	if err != nil {
		return nil, fmt.Errorf("land: the repository's commit turn: %w", err)
	}
	return lock.Close, nil
}
