package main

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"overgo/internal/overgodb"
	"overgo/internal/plan"
)

func mustTestCompletionAuthority(t *testing.T, document plan.Plan) plan.CompletionAuthority {
	t.Helper()
	authority, err := testCompletionAuthority(t, document)
	if err != nil {
		t.Fatal(err)
	}
	return authority
}

func testCompletionAuthority(t *testing.T, document plan.Plan) (plan.CompletionAuthority, error) {
	t.Helper()
	root := initializePlanTestRepository(t, document)
	store, err := overgodb.OpenReadOnly(filepath.Join(root, "overgodb-store"))
	if err != nil {
		return plan.CompletionAuthority{}, err
	}
	defer store.Close()
	return plan.ResolveCompletionAuthority(context.Background(), root, "HEAD", document, store)
}

func mustTestCompletionAuthorityBinding(
	t *testing.T,
	document plan.Plan,
) (plan.CompletionAuthority, string, string) {
	t.Helper()
	root := initializePlanTestRepository(t, document)
	headRaw, err := commandOutput(root, "git", "rev-parse", "HEAD")
	if err != nil {
		t.Fatal(err)
	}
	head := strings.TrimSpace(string(headRaw))
	store, err := overgodb.OpenReadOnly(filepath.Join(root, "overgodb-store"))
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	authority, err := plan.ResolveCompletionAuthority(context.Background(), root, head, document, store)
	if err != nil {
		t.Fatal(err)
	}
	return authority, root, head
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
