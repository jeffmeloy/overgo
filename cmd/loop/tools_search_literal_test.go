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

// TestTheWorkerIsToldASearchMatchesTextAsWritten holds search_files to saying how it matches (loop-proof/search-literal):
// it finds its text as written within one line, and across runs 3 to 7 of loop-proof/local-demonstrated the worker
// searched for patterns ("func Test.*Store") and across lines, each answered that no file holds it. The description says
// the text is found as written, not as a pattern, and a search that finds nothing says the same.
func TestTheWorkerIsToldASearchMatchesTextAsWritten(t *testing.T) {
	t.Parallel()
	root := t.TempDir()
	testcheck.NoError(t, exec.Command("git", "-C", root, "init", "-q").Run())
	path := filepath.Join(root, "store_test.go")
	testcheck.NoError(t, os.WriteFile(path, []byte("package a\n\nfunc TestStore() {}\n"), 0o644))
	tools := workerTools(root, workerStep{files: func() ([]string, error) { return nil, nil }})
	search := tools[slices.IndexFunc(tools, func(tool workerTool) bool { return tool.name == "search_files" })]
	if !strings.Contains(search.description, "as written") || !strings.Contains(search.description, "not as a pattern") {
		t.Errorf("search_files is described as %q; want it to say the text is found as written, not as a pattern", search.description)
	}
	if answer := testcheck.Must(search.run(map[string]string{"text": "func Test.*Store"}))(t); !strings.Contains(answer, "as written") {
		t.Errorf("a pattern searched for answered %q; want it to say nothing holds the text as written", answer)
	}
}
