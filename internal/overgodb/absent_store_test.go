package overgodb

import (
	"errors"
	"os"
	"path/filepath"
	"testing"

	"overgo/internal/testcheck"
)

// TestAWritableOpenOfAnAbsentStoreRefuses holds a mistyped path to an error: Open
// leaves nothing behind, Create starts the store, and a store once started opens
// either way.
func TestAWritableOpenOfAnAbsentStoreRefuses(t *testing.T) {
	t.Parallel()
	base := t.TempDir()
	root := filepath.Join(base, "mistyped", "store")
	for _, path := range []string{root, filepath.Join(base, "store")} {
		if _, err := Open(path); !errors.Is(err, errStoreAbsent) {
			t.Fatalf("open of %s = %v", path, err)
		}
		if _, err := os.Stat(path); !errors.Is(err, os.ErrNotExist) {
			t.Fatalf("a refused open left %s: %v", path, err)
		}
	}
	started, err := Create(root)
	testcheck.NoError(t, err)
	testcheck.NoError(t, started.Close())
	for _, reopen := range []func(string) (*Store, error){Open, Create} {
		store, err := reopen(root)
		testcheck.NoError(t, err)
		testcheck.NoError(t, store.Close())
	}
}
