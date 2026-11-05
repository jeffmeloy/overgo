package main

import (
	"path/filepath"
	"testing"

	"overgo/internal/landing"
	"overgo/internal/testcheck"
	"overgo/internal/testutil"
)

// TestOneListAnswersWhetherEachWorktreeIsOnTheTrunk holds the leftovers listing to one git command for every
// worktree's question of the trunk: a copy whose commit the trunk holds is on it and one with a commit of its own is
// not, and both answers stand once the repository can no longer be asked, so neither was asked alone. A list git
// cannot read, one naming a commit it does not hold, falls back to asking each commit alone, which still answers the
// commits it holds.
func TestOneListAnswersWhetherEachWorktreeIsOnTheTrunk(t *testing.T) {
	t.Parallel()
	repo := t.TempDir()
	git := func(directory string, arguments ...string) string { return gitFixture(t, directory, arguments...) }
	git(repo, "init", "-q", "-b", "master")
	testutil.WriteTextFile(t, repo, "tracked.txt", "tracked\n")
	git(repo, "add", "-A")
	git(repo, "commit", "-q", "-m", "base")
	behind, ahead := filepath.Join(t.TempDir(), "behind"), filepath.Join(t.TempDir(), "ahead")
	git(repo, "worktree", "add", "-q", "--detach", behind, "master")
	git(repo, "worktree", "add", "-q", "--detach", ahead, "master")
	testutil.WriteTextFile(t, ahead, "own.txt", "its own\n")
	git(ahead, "add", "-A")
	git(ahead, "commit", "-q", "-m", "a commit of its own")
	trees, err := landing.Worktrees(t.Context(), repo)
	testcheck.NoError(t, err)
	head := func(root string) string {
		t.Helper()
		for _, tree := range trees {
			if same, _ := filepath.Rel(tree.Root, root); same == "." {
				return tree.Head
			}
		}
		t.Fatalf("git lists no worktree at %s", root)
		return ""
	}
	trunk := trees[0].Head
	f := &facts{hygiene: hygiene{checkout: repo}, trees: trees}
	if off, err := f.offTheTrunk(head(ahead), trunk); err != nil || !off {
		t.Fatalf("a copy with a commit of its own is off the trunk = %t, %v", off, err)
	}
	f.checkout = t.TempDir()
	if off, err := f.offTheTrunk(head(behind), trunk); err != nil || off {
		t.Fatalf("a copy at the trunk's commit, asked after the repository went, is off the trunk = %t, %v", off, err)
	}

	gone := landing.Worktree{Root: t.TempDir(), Head: "0123456789abcdef0123456789abcdef01234567"}
	alone := &facts{hygiene: hygiene{checkout: repo}, trees: append(trees, gone)}
	if off, err := alone.offTheTrunk(head(ahead), trunk); err != nil || !off {
		t.Fatalf("beside a commit git does not hold, a copy with a commit of its own is off the trunk = %t, %v", off, err)
	}
	if off, err := alone.offTheTrunk(head(behind), trunk); err != nil || off {
		t.Fatalf("beside a commit git does not hold, a copy at the trunk's commit is off the trunk = %t, %v", off, err)
	}
	if _, err := alone.offTheTrunk(gone.Head, trunk); err == nil {
		t.Fatal("a commit git does not hold was answered")
	}
}
