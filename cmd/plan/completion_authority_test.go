package main

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"overgo/internal/overgodb"
	"overgo/internal/plan"
)

// mustTestRepositoryBinding seeds a repository holding document and returns
// its worktree and HEAD.
func mustTestRepositoryBinding(t *testing.T, document plan.Plan) (string, string) {
	t.Helper()
	root := initializePlanTestRepository(t, document)
	headRaw, err := commandOutput(root, "git", "rev-parse", "HEAD")
	if err != nil {
		t.Fatal(err)
	}
	return root, strings.TrimSpace(string(headRaw))
}

func initializePlanTestRepository(t *testing.T, document plan.Plan) string {
	t.Helper()
	root := t.TempDir()
	if err := os.MkdirAll(filepath.Join(root, "docs"), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := plan.Save(filepath.Join(root, filepath.FromSlash(plan.Path)), document); err != nil {
		t.Fatal(err)
	}
	for _, arguments := range [][]string{
		{"init"},
		{"config", "user.email", "overgo@example.invalid"},
		{"config", "user.name", "Overgo Test"},
		{"add", plan.Path},
		{"commit", "-m", "seed plan"},
	} {
		if _, err := commandOutput(root, "git", arguments...); err != nil {
			t.Fatal(err)
		}
	}
	store, err := overgodb.Open(filepath.Join(root, "overgodb-store"))
	if err != nil {
		t.Fatal(err)
	}
	if err := store.Close(); err != nil {
		t.Fatal(err)
	}
	return root
}
