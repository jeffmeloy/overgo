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
	root, branch, err := trunkOf(ctx, l.Repo)
	if err != nil {
		return nil, err
	}
	l.trunk, l.branch = root, branch
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

// integrate brings the trunk's tip into a worktree that is not the trunk's
// own checkout, before anything of the commit is read. Such a worktree's
// change was written on an earlier commit of the trunk; committed onto the
// tip as it stands in the working tree, it would undo what landed between.
// So the working tree becomes the three-way merge of the tip, the commit the
// change was written on and the change, and the worktree's HEAD the tip: the
// commit then runs on the tree it will publish, reads its changed paths, its
// count and its check against the tip, and publishes onto the trunk (commit).
// A file the trunk and the change both changed otherwise refuses the commit by
// name, with the working tree as it was; so does a worktree whose commit the
// trunk does not descend from, since nothing says what its change is against.
// The trunk's own checkout, and a repository whose main worktree is on no
// branch, commit as they did.
func (l *landing) integrate(ctx context.Context) error {
	read := func(arguments ...string) (string, error) {
		output, err := gitauthority.Query(ctx, l.Repo, arguments...)
		return strings.TrimSpace(string(output)), err
	}
	checked, err := read("rev-parse", "--symbolic-full-name", "HEAD")
	if err != nil || l.branch == "" || checked == l.branch {
		return err
	}
	l.onto = l.branch
	head, err := read("rev-parse", "HEAD")
	if err != nil {
		return err
	}
	tip, err := read("rev-parse", l.onto)
	if err != nil || tip == head {
		return err
	}
	if _, err := read("merge-base", "--is-ancestor", head, tip); err != nil {
		return fmt.Errorf("land: this worktree is on %.12s, which the trunk %s at %.12s does not descend from; a worktree commits a change written on a commit of the trunk", head, l.onto, tip)
	}
	if _, err := l.staged(ctx, "read-tree", head); err != nil {
		return err
	}
	candidate, err := l.capture(ctx)
	if err != nil {
		return err
	}
	held, err := l.git(ctx, "commit-tree", candidate, "-p", head, "-m", "candidate of "+l.Ref)
	if err != nil {
		return err
	}
	// merge-tree writes the merged tree and names it first; when it cannot merge it names the files after it.
	merged, err := l.run(ctx, gitauthority.RepositoryEnvironment(), "git", "merge-tree", "--write-tree", "--name-only", tip, strings.TrimSpace(held))
	tree, contested, _ := strings.Cut(strings.TrimSpace(merged), "\n")
	if err != nil && contested == "" {
		return err
	}
	if err != nil {
		contested, _, _ = strings.Cut(contested, "\n\n")
		return fmt.Errorf("land: the trunk %s changed since %.12s, which this change was written on, files this change also changes: %s; bring the trunk's %.12s into this worktree and settle them, then commit again",
			l.onto, head, strings.ReplaceAll(contested, "\n", ", "), tip)
	}
	if _, err := l.staged(ctx, "read-tree", "-m", "-u", candidate, tree); err != nil {
		return err
	}
	if _, err := l.git(ctx, "update-ref", "-m", "integrate "+l.Ref, "HEAD", tip, head); err != nil {
		return err
	}
	fmt.Fprintf(l.Log, "land: brought the trunk %s, at %.12s, into this worktree, which was on %.12s\n", l.onto, tip, head)
	_, err = l.git(ctx, "reset", "-q")
	return err
}
