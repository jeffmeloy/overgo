package main

import (
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// TestAnImportRefusesWhatItCannotRead holds the fixture import to its inputs: on 2026-11-09 the first mutation run
// (loop-proof/mutants) inverted its checks of them and no test failed, the flattening test never importing. An import
// missing the node definitions or the saved sections, either alone, is refused naming both; one whose workflow cannot
// be read fails with that file's absence, not with whatever reading nothing goes on to.
func TestAnImportRefusesWhatItCannotRead(t *testing.T) {
	t.Parallel()
	folder := t.TempDir()
	absent := filepath.Join(folder, "absent.json")
	for _, given := range [][2]string{{"", absent}, {absent, ""}} {
		if err := importFixture(folder, absent, given[0], given[1]); err == nil || !strings.Contains(err.Error(), "needs -import-object-info and -import-saved") {
			t.Errorf("an import given object info %q and saved sections %q ended as %v, want it refused naming both", given[0], given[1], err)
		}
	}
	if err := importFixture(folder, absent, absent, absent); !errors.Is(err, os.ErrNotExist) {
		t.Errorf("an import of a workflow that is not there ended as %v, want the file's absence", err)
	}
}
