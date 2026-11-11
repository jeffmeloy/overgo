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

// TestTheWorkerSearchFindsAFileGitDoesNotTrackYet searches a checkout whose test file git does not track yet, as a
// replay puts a landing's added test at its parent (loop-proof/search-untracked): in run 6 of
// loop-proof/local-demonstrated every failed step had its check in such a file, and the worker was told no tracked file
// holds the test it searched for. search_files finds a text held by a file not added to git, with its file and line,
// and does not find one held only by a file git ignores.
func TestTheWorkerSearchFindsAFileGitDoesNotTrackYet(t *testing.T) {
	t.Parallel()
	root := t.TempDir()
	testcheck.NoError(t, exec.Command("git", "-C", root, "init", "-q").Run())
	for name, content := range map[string]string{
		".gitignore":                 "scratch/\n",
		"internal/a/pointer_test.go": "package a\n\nfunc TestThePointerFileNamesSkillAndTheCommands() {}\n",
		"scratch/notes.txt":          "TestOnlyInScratch\n",
	} {
		path := filepath.Join(root, filepath.FromSlash(name))
		testcheck.NoError(t, os.MkdirAll(filepath.Dir(path), 0o755))
		testcheck.NoError(t, os.WriteFile(path, []byte(content), 0o644))
	}
	tools := workerTools(root, workerStep{files: func() ([]string, error) { return []string{"internal/a"}, nil }})
	search := tools[slices.IndexFunc(tools, func(tool workerTool) bool { return tool.name == "search_files" })].run
	found := testcheck.Must(search(map[string]string{"text": "TestThePointerFileNamesSkillAndTheCommands"}))(t)
	if !strings.Contains(found, "internal/a/pointer_test.go:3:") {
		t.Fatalf("searching for a test in a file git does not track yet answered %q; want its file and line", found)
	}
	if ignored := testcheck.Must(search(map[string]string{"text": "TestOnlyInScratch"}))(t); strings.Contains(ignored, "scratch/notes.txt") {
		t.Fatalf("searching for a text held only by an ignored file answered %q; want it not found", ignored)
	}
}
