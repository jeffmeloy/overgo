package main

import (
	"os"
	"path/filepath"
	"testing"
)

// requireGone fails for each repository path that still exists.
func requireGone(t *testing.T, paths ...string) {
	t.Helper()
	for _, path := range paths {
		if _, err := os.Stat(filepath.Join("..", "..", filepath.FromSlash(path))); !os.IsNotExist(err) {
			t.Errorf("%s still exists (%v)", path, err)
		}
	}
}

// TestOldGateGone holds the landing path to the small gate: the old gate's
// package and its command modes are deleted.
func TestOldGateGone(t *testing.T) {
	t.Parallel()
	requireGone(t, "internal/gate")
}
