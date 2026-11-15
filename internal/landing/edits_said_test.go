package landing

import (
	"strings"
	"testing"

	"overgo/internal/testcheck"
)

// TestEveryFailingCheckSaysItsEdit holds a refused landing's recorded cause to leading with the edit its refusal says
// (single-source/edits-said): a failed test's last assertion, which says what it found and where, and a rule's edit,
// however long the finding before it. Of the 25 refused landings the status listed on 2026-11-14 none said its edit:
// a failed test's line named the tests alone, and a rule's edit came after a finding the record's cut ended. Both
// count on the worklist as refusals that said their edit, and the failed test is still listed by its name.
func TestEveryFailingCheckSaysItsEdit(t *testing.T) {
	t.Parallel()
	failed := Classify("row/do", "gate: land: test: failed tests/packages: overgo/internal/x: TestCount, overgo/internal/x\n"+
		"overgo/internal/x: TestCount:\n=== RUN   TestCount\n    count_test.go:9: the walk counts 3 files\n"+
		"    count_test.go:12: the ceiling is 2: remove a file or raise the ceiling with the row that adds it\n--- FAIL: TestCount (0.01s)\n")
	if want := "edit: count_test.go:12: the ceiling is 2: remove a file or raise the ceiling with the row that adds it; test: "; !strings.HasPrefix(failed.Cause, want) {
		t.Errorf("a failed test's refusal is recorded as %q; want it to lead with %q", failed.Cause, want)
	}
	long := strings.Repeat("internal/x/pinned_file.go, ", 12)
	pinned := Classify("row/do", "gate: land: rules: changes bytes a record pins: "+long+"(pinned by docs/record.json; edit: run go run ./cmd/pins -update)\n")
	if want := "edit: run go run ./cmd/pins -update; rules: changes bytes a record pins: "; !strings.HasPrefix(pinned.Cause, want) {
		t.Errorf("a rule's refusal with its edit past %d bytes is recorded as %q; want it to lead with %q", causeBytes, pinned.Cause, want)
	}
	root := t.TempDir()
	for _, refusal := range []Refusal{failed, pinned} {
		testcheck.Must(RecordRefusal(root, refusal))(t)
	}
	for _, item := range testcheck.Must(Worklist(root))(t) {
		if item.Edits != item.Refusals {
			t.Errorf("the worklist counts %d of %s's %d refusals as saying their edit; want every one", item.Edits, item.By, item.Refusals)
		}
		if item.By == "overgo/internal/x" {
			t.Errorf("the worklist lists the failed test's refusal by its package alone: %+v", item)
		}
	}
}
