package fsatomic

import (
	"errors"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"testing"

	"overgo/internal/testcheck"
)

// treeFixture writes under root a tree of two folders and three files, and
// returns the tree's path with its files'.
func treeFixture(t *testing.T, root string) (string, []string) {
	t.Helper()
	tree := filepath.Join(root, "tree")
	files := []string{filepath.Join(tree, "document"), filepath.Join(tree, "inner", "weights"), filepath.Join(tree, "inner", "deeper", "state")}
	for _, file := range files {
		testcheck.NoError(t, os.MkdirAll(filepath.Dir(file), DirectoryMode))
		testcheck.NoError(t, os.WriteFile(file, []byte("content of "+filepath.Base(file)), 0o600))
	}
	return tree, files
}

// TestRemoveTreeLeavesWhatALinkNames holds the one remover of a whole
// directory tree to removing only what is inside the tree.
//
// A tree that holds a link to a folder outside it is removed, and the folder
// the link names keeps its file: the link is deleted and never followed. On
// Windows the link is a symbolic link where the host lets one be made, and a
// junction, which needs no privilege, where it does not. Nothing is left
// beside where the tree stood, so the name it is renamed to on the way is
// gone too. A path that is not there is no error.
//
// What only one system does is held by removeTreeHere.
func TestRemoveTreeLeavesWhatALinkNames(t *testing.T) {
	t.Parallel()
	root, outside := t.TempDir(), t.TempDir()
	kept := filepath.Join(outside, "kept")
	testcheck.NoError(t, os.WriteFile(kept, []byte("outside the tree"), 0o600))
	tree, _ := treeFixture(t, root)
	link := filepath.Join(tree, "inner", "link")
	kind, err := "symbolic link", os.Symlink(outside, link)
	if err != nil && runtime.GOOS == "windows" {
		kind, err = "junction", exec.Command("cmd", "/c", "mklink", "/J", link, outside).Run()
	}
	if err != nil {
		t.Logf("this host makes no link (%v): the tree is removed without one", err)
	} else if _, err := os.Stat(filepath.Join(link, "kept")); err != nil {
		t.Fatalf("the %s does not reach the folder it names: %v", kind, err)
	} else {
		t.Logf("the tree holds a %s to the folder outside it", kind)
	}
	testcheck.NoError(t, RemoveTree(tree))
	left, err := os.ReadDir(root)
	if err != nil || len(left) != 0 {
		t.Fatalf("after the tree was removed its parent holds %v (%v), want nothing", left, err)
	}
	if content, err := os.ReadFile(kept); err != nil || string(content) != "outside the tree" {
		t.Fatalf("the file a link in the tree named reads %q (%v), want it as it was", content, err)
	}
	if err := RemoveTree(tree); err != nil {
		t.Fatalf("removing a tree that is not there = %v, want no error", err)
	}
	if err := RemoveTree(filepath.Join(root, "never", "made")); err != nil && !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("removing a path under a folder that is not there = %v", err)
	}
	removeTreeHere(t)
}
