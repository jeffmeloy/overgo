//go:build windows

package fsatomic

import (
	"os"
	"testing"
)

// removeTreeHere holds what RemoveTree does on Windows alone.
//
// A tree with a file in it held open is in use: the removal fails, and every
// file is still at its own path, because the tree is renamed before anything
// in it is deleted and Windows refuses to rename a folder that holds an open
// file. Once the file is closed the same call removes the tree.
//
// A removal that stopped after its rename left the tree under the name it was
// renamed to. The next call clears that name first, whether or not a tree
// stands at the path again, so the names do not pile up.
func removeTreeHere(t *testing.T) {
	t.Helper()
	root := t.TempDir()
	tree, files := treeFixture(t, root)
	held, err := os.Open(files[1])
	if err != nil {
		t.Fatal(err)
	}
	err = RemoveTree(tree)
	for _, file := range files {
		if _, statErr := os.Stat(file); statErr != nil {
			t.Errorf("with %s held open, removing the tree left %s as %v, want every file at its own path", files[1], file, statErr)
		}
	}
	if closeErr := held.Close(); err == nil || closeErr != nil {
		t.Fatalf("removing a tree with a file held open = %v (close: %v), want it refused", err, closeErr)
	}
	if err := RemoveTree(tree); err != nil {
		t.Fatalf("removing the tree once its file was closed = %v", err)
	}

	// A name an interrupted removal left, with and without a tree at the path.
	for _, standing := range []bool{false, true} {
		tree, _ := treeFixture(t, root)
		if err := os.Rename(tree, removalTombstone(tree)); err != nil {
			t.Fatal(err)
		}
		if standing {
			treeFixture(t, root)
		}
		if err := RemoveTree(tree); err != nil {
			t.Fatalf("removing over the name an earlier removal left (a tree at the path: %t) = %v", standing, err)
		}
		if left, err := os.ReadDir(root); err != nil || len(left) != 0 {
			t.Fatalf("after a removal over the name an earlier one left (a tree at the path: %t) the parent holds %v (%v), want nothing", standing, left, err)
		}
	}
}
