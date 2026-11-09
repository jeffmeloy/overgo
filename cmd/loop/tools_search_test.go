package main

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"overgo/internal/clioptions"
	"overgo/internal/testcheck"
)

// TestTheWorkerSearchesAndReadsOnlyTheLinesItNeeds: the worker finds a text in the checkout's tracked files with the
// file and line it is on, reads a span of lines with their numbers, and a call missing an argument a tool names, or
// giving one as other than a string, is refused before the tool runs, the file as it was (SQA-347: an edit with no old
// wrote new over the whole file).
func TestTheWorkerSearchesAndReadsOnlyTheLinesItNeeds(t *testing.T) {
	t.Parallel()
	root := t.TempDir()
	source := "package a\n\n// needle is what the search finds.\nconst needle = 1\n"
	path := filepath.Join(root, "a", "a.go")
	testcheck.NoError(t, os.MkdirAll(filepath.Dir(path), clioptions.OutputDirectoryMode))
	testcheck.NoError(t, os.WriteFile(path, []byte(source), clioptions.OutputFileMode))
	testcheck.Must(runTool("git", "-C", root, "init", "-q"))(t)
	testcheck.Must(runTool("git", "-C", root, "add", "a/a.go"))(t)
	tools := workerTools(root, workerStep{files: func() ([]string, error) { return []string{"a"}, nil }})

	found, err := callTool(tools, "search_files", map[string]any{"text": "const needle"})
	if err != nil || !strings.Contains(found, "a/a.go:4:const needle = 1") {
		t.Fatalf("searching for a constant found %q (%v), want its file and line", found, err)
	}
	read, err := callTool(tools, "read_lines", map[string]any{"path": "a/a.go", "from": "3", "to": "4"})
	if want := "3\t// needle is what the search finds.\n4\tconst needle = 1\n"; err != nil || read != want {
		t.Fatalf("reading lines 3 to 4 gave %q (%v), want %q", read, err, want)
	}
	for _, malformed := range []map[string]any{
		{"path": "a/a.go", "new": "lost\n"},
		{"path": "a/a.go", "old": "needle = 1", "new": true},
	} {
		answer, err := callTool(tools, "edit_file", malformed)
		if data, readErr := os.ReadFile(path); err == nil || readErr != nil || string(data) != source {
			t.Fatalf("an edit given %v answered %q (%v) and left %q (%v); want it refused and the file as it was", malformed, answer, err, data, readErr)
		}
	}
	if _, err := callTool(tools, "read_lines", map[string]any{"path": "a/a.go", "from": []string{"3"}, "to": "4"}); err == nil {
		t.Fatal("reading lines from a list, not a string, was not refused")
	}
}
