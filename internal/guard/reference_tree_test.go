package guard

import (
	"strings"
	"testing"
)

// TestNoSessionWritesIntoTheReferenceTree refuses each way a session writes into the reference tree, an editing
// tool's file or notebook, output sent there, and a program that deletes, moves, creates, edits in place or copies onto
// a path there, by either separator. It allows reading the tree, copying out of it, and writing a file whose name
// only begins with the tree's.
func TestNoSessionWritesIntoTheReferenceTree(t *testing.T) {
	t.Parallel()
	tree := "C:/Users/someone/" + referenceTree
	for _, check := range []struct {
		tool, path, notebook, command string
		refused                       bool
	}{
		{tool: "Edit", path: tree + "/internal/model/model.go", refused: true},
		{tool: "Write", path: `C:\Users\someone\` + referenceTree + `\notes.md`, refused: true},
		{tool: "NotebookEdit", notebook: tree + "/study.ipynb", refused: true},
		{tool: "Write", path: "docs/" + referenceTree + "_parity_report.md"},
		{tool: "Bash", command: "echo note > " + tree + "/notes.txt", refused: true},
		{tool: "Bash", command: "rm " + tree + "/x.go", refused: true},
		{tool: "Bash", command: "mv " + tree + "/x.go tmp/x.go", refused: true},
		{tool: "Bash", command: "cp internal/model/model.go " + tree + "/model.go", refused: true},
		{tool: "Bash", command: "sed -i s/a/b/ " + tree + "/x.go", refused: true},
		{tool: "Bash", command: "touch " + tree + "/new.txt", refused: true},
		{tool: "Bash", command: "cat " + tree + "/x.go"},
		{tool: "Bash", command: "grep -rn Forward " + tree},
		{tool: "Bash", command: "cp " + tree + "/x.go tmp/x.go"},
		{tool: "Bash", command: "sed -n 1,20p " + tree + "/x.go"},
	} {
		call := ToolCall{ToolName: check.tool}
		call.ToolInput.Path, call.ToolInput.Notebook, call.ToolInput.Command = check.path, check.notebook, check.command
		message, _ := Verdict(call, t.TempDir())
		if refused := strings.Contains(message, referenceTree); refused != check.refused || !refused && message != "" {
			t.Errorf("%s %s%s%s: the guard said %q, want refused=%v for the reference tree", check.tool, check.path, check.notebook, check.command, message, check.refused)
		}
	}
}
