package main

import (
	"os"
	"path/filepath"
	"testing"

	"overgo/internal/testcheck"
)

// TestACommitClearsOnlyALandedRowsFolder holds SQA-479: of the scratch folder's entries a commit may clear, a folder
// named for a row it landed is that row's, in any letter case, and a loose file of the same name, a folder of another
// row's name and a path that is gone are not, so the file stays for the deleting command.
func TestACommitClearsOnlyALandedRowsFolder(t *testing.T) {
	t.Parallel()
	scratch := t.TempDir()
	testcheck.NoError(t, os.Mkdir(filepath.Join(scratch, "Row-A"), 0o755))
	testcheck.NoError(t, os.Mkdir(filepath.Join(scratch, "row-c"), 0o755))
	testcheck.NoError(t, os.WriteFile(filepath.Join(scratch, "row-b"), []byte("the only copy of a note\n"), 0o600))
	landed := []string{"row-a", "row-b"}
	for name, want := range map[string]bool{"Row-A": true, "row-b": false, "row-c": false, "row-gone": false} {
		if got := landedRowFolder(filepath.Join(scratch, name), landed); got != want {
			t.Errorf("tmp/%s is a landed row's folder: %v, want %v", name, got, want)
		}
	}
}
