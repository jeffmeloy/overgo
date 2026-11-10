package main

import (
	"os"
	"os/exec"
	"path/filepath"
	"slices"
	"testing"

	"overgo/internal/testcheck"
)

// TestTheWorkerDeletesAFileItsStepMayChange gives the worker a step whose change is a deletion, as
// test-reach/fuzz-or-delete's check wants a script gone (loop-proof/delete-tool): delete_file removes a file among the
// step's files, refuses one outside them, and, in a replay, refuses to delete the step's held test as it refuses to edit
// one. In run 5 of loop-proof/local-demonstrated the worker had no way to remove a file and spent its 40 replies three
// times.
func TestTheWorkerDeletesAFileItsStepMayChange(t *testing.T) {
	t.Parallel()
	root := t.TempDir()
	testcheck.NoError(t, exec.Command("git", "-C", root, "init", "-q").Run())
	for _, name := range []string{"scripts/stall_check.sh", "internal/y/y.go", "scripts/held_test.go"} {
		path := filepath.Join(root, filepath.FromSlash(name))
		testcheck.NoError(t, os.MkdirAll(filepath.Dir(path), 0o755))
		testcheck.NoError(t, os.WriteFile(path, []byte("x\n"), 0o644))
	}
	task := replayTask{Files: []string{"scripts"}, Tests: []string{"scripts/held_test.go"}}
	tools := task.tools(t.Context(), root)
	at := slices.IndexFunc(tools, func(tool workerTool) bool { return tool.name == "delete_file" })
	if at < 0 {
		t.Fatal("the worker has no delete_file tool")
	}
	remove := tools[at].run
	if _, err := remove(map[string]string{"path": "scripts/stall_check.sh"}); err != nil {
		t.Fatalf("deleting a file the step may change: %v", err)
	}
	if _, err := os.Stat(filepath.Join(root, "scripts", "stall_check.sh")); !os.IsNotExist(err) {
		t.Fatalf("scripts/stall_check.sh stands after its delete (%v)", err)
	}
	for _, refused := range []string{"internal/y/y.go", "scripts/held_test.go"} {
		if _, err := remove(map[string]string{"path": refused}); err == nil {
			t.Errorf("deleting %s was not refused", refused)
		}
		if _, err := os.Stat(filepath.Join(root, filepath.FromSlash(refused))); err != nil {
			t.Errorf("%s is gone after a refused delete: %v", refused, err)
		}
	}
}
