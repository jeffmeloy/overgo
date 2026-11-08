package landing

import (
	"bytes"
	"context"
	"errors"
	"io/fs"
	"os"
	"path/filepath"
	"slices"
	"strconv"

	"overgo/internal/clioptions"
	"overgo/internal/dataroot"
	"overgo/internal/gitauthority"
	"overgo/internal/processlock"
)

// candidateCheckout names, in a checkout's process state, the worktrees the stages read candidates in, each numbered.
const candidateCheckout = "landing-candidate-"

// folderStages are the stages that read the checkout itself: the free space it has, and the format and fix stages,
// which rewrite its files. Every later stage reads the candidate in a worktree of its own (checkOut).
var folderStages = []string{spaceClass, "format", "fix"}

// checkOut checks the candidate out in a worktree of its own, where every stage after format and fix reads it: on
// 2026-11-08 another session's document in the lead's checkout refused every landing, since the stages read the
// folder, which holds more than the commit. The worktree is kept in the checkout's process state from one landing to
// the next, as the cycle's checkout is (cmd/loop), since go build and go test key what they cache by a package's
// directory: a worktree under a new name would build and test everything again. A landing takes the first one no other
// landing or preflight holds, and gives it back as it ends (Run). A candidate taken again, once the generators derived
// from it (derive), moves the worktree to it.
func (l *landing) checkOut(ctx context.Context) error {
	if l.tree == "" {
		if err := l.takeTree(); err != nil {
			return err
		}
	}
	if l.treeHolds == l.candidate {
		return nil
	}
	if err := l.moveTree(ctx, l.tree); err != nil {
		return err
	}
	l.treeHolds = l.candidate
	return nil
}

// takeTree takes the first kept worktree no other landing or preflight holds.
func (l *landing) takeTree() error {
	if err := processlock.EnsureStateDirectory(l.Repo); err != nil {
		return err
	}
	for slot := 0; ; slot++ {
		tree := filepath.Join(l.Repo, processlock.StateDirectory, candidateCheckout+strconv.Itoa(slot))
		lock, err := processlock.Acquire(tree+".lock", clioptions.PrivateFileMode)
		if errors.Is(err, processlock.ErrBusy) {
			continue
		}
		if err != nil {
			return err
		}
		l.tree, l.treeHeld = tree, lock.Close
		return nil
	}
}

// moveTree makes tree the candidate: its HEAD the commit the landing builds on, its files the candidate's bytes, which
// no line-ending setting of the machine rewrites, and a file an earlier run left among them that git does not ignore
// removed. What git ignores stays, as it does in the cycle's checkout and the main one. A worktree whose folder is gone
// is pruned from git's list and added again.
func (l *landing) moveTree(ctx context.Context, tree string) error {
	if _, err := os.Lstat(filepath.Join(tree, ".git")); errors.Is(err, fs.ErrNotExist) {
		if _, err := l.git(ctx, "worktree", "prune"); err != nil {
			return err
		}
		if _, err := l.git(ctx, "worktree", "add", "--detach", "--no-checkout", tree, l.head); err != nil {
			return err
		}
	}
	for _, arguments := range [][]string{{"update-ref", "--no-deref", "HEAD", l.head}, {"-c", "core.autocrlf=false", "read-tree", "-u", "--reset", l.candidate}} {
		if _, err := l.runIn(ctx, tree, gitauthority.RepositoryEnvironment(), "git", gitauthority.WriterArguments(arguments...)...); err != nil {
			return err
		}
	}
	left, err := gitauthority.Query(ctx, tree, "ls-files", "--others", "--exclude-standard", "-z")
	if err != nil {
		return err
	}
	for name := range bytes.SplitSeq(left, []byte{0}) {
		if len(name) == 0 {
			continue
		}
		if err := os.Remove(filepath.Join(tree, filepath.FromSlash(string(name)))); err != nil && !errors.Is(err, fs.ErrNotExist) {
			return err
		}
	}
	// The machine's data roots are the checkout's, in a file git ignores, so a check that reads a model resolves it in
	// the worktree as it does there.
	roots, err := os.ReadFile(filepath.Join(l.Repo, dataroot.ConfigFile))
	if errors.Is(err, fs.ErrNotExist) {
		return nil
	}
	if err != nil {
		return err
	}
	return os.WriteFile(filepath.Join(tree, dataroot.ConfigFile), roots, clioptions.PrivateFileMode)
}

// derive runs the generators over the candidate, in the worktree the stages read it in, and writes what they derive
// into the checkout, which the next capture takes: on 2026-11-08 the API manifest, regenerated from the folder, carried
// the identity of a file the commit left out, and the commit's own test refused it.
func (l *landing) derive(ctx context.Context, generators []generator) error {
	if err := l.shipSet(ctx); err != nil {
		return err
	}
	if err := l.checkOut(ctx); err != nil {
		return err
	}
	for _, generator := range generators {
		if _, err := l.runIn(ctx, l.tree, generator.environment, "go", generator.arguments...); err != nil {
			return err
		}
		for _, name := range generator.writes {
			derived, err := os.ReadFile(filepath.Join(l.tree, filepath.FromSlash(name)))
			if errors.Is(err, fs.ErrNotExist) {
				continue
			}
			if err != nil {
				return err
			}
			if held, _ := os.ReadFile(filepath.Join(l.Repo, filepath.FromSlash(name))); bytes.Equal(held, derived) {
				continue
			}
			if err := os.WriteFile(filepath.Join(l.Repo, filepath.FromSlash(name)), derived, clioptions.OutputFileMode); err != nil {
				return err
			}
		}
	}
	return nil
}

// readsCandidate says whether the stage named reads the candidate rather than the checkout.
func readsCandidate(stage string) bool { return !slices.Contains(folderStages, stage) }
