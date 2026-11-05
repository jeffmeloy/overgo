package main

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"overgo/internal/overgodb"
	"overgo/internal/plan"
	"overgo/internal/testcheck"
)

// mustTestRepositoryBinding seeds a repository holding document and returns
// its worktree and HEAD.
func mustTestRepositoryBinding(t *testing.T, document plan.Plan) (string, string) {
	t.Helper()
	root := initializePlanTestRepository(t, document)
	headRaw, err := commandInput(root, nil, "git", "rev-parse", "HEAD")
	testcheck.NoError(t, err)
	return root, strings.TrimSpace(string(headRaw))
}

func initializePlanTestRepository(t *testing.T, document plan.Plan) string {
	t.Helper()
	root := t.TempDir()
	testcheck.NoError(t, os.MkdirAll(filepath.Join(root, "docs"), 0o755))
	testcheck.NoError(t, plan.Save(filepath.Join(root, filepath.FromSlash(plan.Path)), document))
	for _, arguments := range [][]string{
		{"init"},
		{"config", "user.email", "overgo@example.invalid"},
		{"config", "user.name", "Overgo Test"},
		{"add", plan.Path},
		{"commit", "-m", "seed plan"},
	} {
		if _, err := commandInput(root, nil, "git", arguments...); err != nil {
			t.Fatal(err)
		}
	}
	store, err := overgodb.Create(filepath.Join(root, "overgodb-store"))
	testcheck.NoError(t, err)
	testcheck.NoError(t, store.Close())
	return root
}
