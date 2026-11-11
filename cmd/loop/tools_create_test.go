package main

import (
	"os"
	"os/exec"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"overgo/internal/testcheck"
)

// TestTheWorkerIsToldAnEditCreatesAFile holds edit_file to telling the worker what it does with an empty old: it
// writes the whole file, creating the file and its folders where they do not exist. In runs 6 and 7 of
// loop-proof/local-demonstrated the worker searched for .github/workflows/fuzz.yml more than ten times, as if it had
// to be there already, and never wrote it, though the check failed only for want of it (loop-proof/worker-messages).
func TestTheWorkerIsToldAnEditCreatesAFile(t *testing.T) {
	t.Parallel()
	root := t.TempDir()
	testcheck.NoError(t, exec.Command("git", "-C", root, "init", "-q").Run())
	tools := workerTools(root, workerStep{files: func() ([]string, error) { return nil, nil }})
	edit := tools[slices.IndexFunc(tools, func(tool workerTool) bool { return tool.name == "edit_file" })]
	if !strings.Contains(edit.description, "creating the file and its folders") {
		t.Errorf("edit_file says %q; want it to say an empty old creates the file and its folders", edit.description)
	}
	testcheck.Must(edit.run(map[string]string{"path": ".github/workflows/fuzz.yml", "old": "", "new": "on: schedule\n"}))(t)
	if data, err := os.ReadFile(filepath.Join(root, ".github", "workflows", "fuzz.yml")); err != nil || string(data) != "on: schedule\n" {
		t.Errorf("an edit with an empty old in folders that do not exist left %q (%v)", data, err)
	}
}
