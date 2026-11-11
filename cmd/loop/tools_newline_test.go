package main

import (
	"os/exec"
	"slices"
	"strings"
	"testing"

	"overgo/internal/testcheck"
)

// TestAnEditSaysWhenAFileEndsWithoutANewline holds edit_file to saying when the file it leaves ends without a final
// newline, which a printed file does not show (loop-proof/edit-newline): in run 7 of loop-proof/local-demonstrated the
// worker wrote CLAUDE.md as "@AGENTS.md" with none, its check wanting one, and searched for a line-ending setting it
// could not see. A file ending in a newline is answered as before.
func TestAnEditSaysWhenAFileEndsWithoutANewline(t *testing.T) {
	t.Parallel()
	root := t.TempDir()
	testcheck.NoError(t, exec.Command("git", "-C", root, "init", "-q").Run())
	tools := workerTools(root, workerStep{files: func() ([]string, error) { return nil, nil }})
	edit := tools[slices.IndexFunc(tools, func(tool workerTool) bool { return tool.name == "edit_file" })].run
	without := testcheck.Must(edit(map[string]string{"path": "CLAUDE.md", "old": "", "new": "@AGENTS.md"}))(t)
	if !strings.Contains(without, "ends without a final newline") {
		t.Errorf("an edit leaving CLAUDE.md without its final newline answered %q; want it to say so", without)
	}
	if with := testcheck.Must(edit(map[string]string{"path": "CLAUDE.md", "old": "", "new": "@AGENTS.md\n"}))(t); with != "edited CLAUDE.md" {
		t.Errorf("an edit leaving CLAUDE.md with its final newline answered %q; want %q", with, "edited CLAUDE.md")
	}
}
