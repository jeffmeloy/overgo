package processlock

import (
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"sync"
	"testing"
)

// TestStateDirectoryIgnoresItself holds the state directory to hiding what it
// keeps from git by its own rule: a repository whose ignore file has never
// heard of the directory -- a checkout of an older commit, a fixture -- sees
// no untracked change when a lock is taken in it. Writers racing to the first
// use all succeed and leave one rule and nothing staged beside it.
func TestStateDirectoryIgnoresItself(t *testing.T) {
	t.Parallel()
	root := t.TempDir()
	if output, err := exec.Command("git", "-C", root, "init", "--quiet").CombinedOutput(); err != nil {
		t.Fatalf("git init: %v: %s", err, output)
	}
	var writers sync.WaitGroup
	failures := make([]error, 8)
	for index := range failures {
		writers.Go(func() { failures[index] = EnsureStateDirectory(root) })
	}
	writers.Wait()
	for _, err := range failures {
		if err != nil {
			t.Fatal(err)
		}
	}
	lock, err := Acquire(filepath.Join(root, StateDirectory, "holder.lock"), 0o600)
	if err != nil {
		t.Fatal(err)
	}
	defer lock.Close()
	status, err := exec.Command("git", "-C", root, "status", "--porcelain", "--untracked-files=all").CombinedOutput()
	if err != nil {
		t.Fatalf("git status: %v: %s", err, status)
	}
	if strings.TrimSpace(string(status)) != "" {
		t.Fatalf("state is visible to git:\n%s", status)
	}
	entries, err := os.ReadDir(filepath.Join(root, StateDirectory))
	if err != nil {
		t.Fatal(err)
	}
	names := make([]string, 0, len(entries))
	for _, entry := range entries {
		names = append(names, entry.Name())
	}
	if got, want := strings.Join(names, " "), stateIgnoreFile+" holder.lock"; got != want {
		t.Fatalf("state directory holds %q, want %q", got, want)
	}
}
