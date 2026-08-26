package main

import (
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

func TestCandidateDriftRefused(t *testing.T) {
	repo := t.TempDir()
	runGitFixture(t, repo, "init")
	runGitFixture(t, repo, "config", "user.email", "gate@example.invalid")
	runGitFixture(t, repo, "config", "user.name", "Gate Test")
	path := filepath.Join(repo, "candidate.go")
	if err := os.WriteFile(path, []byte("package candidate\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	runGitFixture(t, repo, "add", "candidate.go")
	runGitFixture(t, repo, "commit", "-m", "base")
	gate := gateContext{repo: repo, paths: []string{"candidate.go"}}
	planned, err := gate.treeStateKey()
	if err != nil {
		t.Fatal(err)
	}
	if err := gate.requireCandidateTree(planned); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, []byte("package candidate\n\nconst changed = true\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := gate.requireCandidateTree(planned); err == nil || !strings.Contains(err.Error(), "candidate drifted after manifest planning") {
		t.Fatalf("drift result = %v", err)
	}
}

func runGitFixture(t *testing.T, repo string, arguments ...string) {
	t.Helper()
	command := exec.Command("git", arguments...)
	command.Dir = repo
	if output, err := command.CombinedOutput(); err != nil {
		t.Fatalf("git %v: %v: %s", arguments, err, output)
	}
}
