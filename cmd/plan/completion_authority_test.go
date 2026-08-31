package main

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"overgo/internal/artifact"
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
	return plan.ResolveCompletionAuthority(t.Context(), root, "HEAD", document, store)
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
	authority, err := plan.ResolveCompletionAuthority(t.Context(), root, head, document, store)
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

func TestNonDispatchCommandsDoNotRequireCompletionAuthority(t *testing.T) {
	census, err := artifact.IdentifyBytes(artifact.KindEvidence, []byte("census"))
	if err != nil {
		t.Fatal(err)
	}
	document := plan.Plan{Census: &census, Items: []plan.Item{{
		ID: "dependent", Status: plan.StatusOpen, Steps: []plan.Step{{
			ID: "do", Status: plan.StatusOpen, Verify: "go test ./...",
			DependsOn: []string{"missing/do"},
		}},
	}}}
	root := initializePlanTestRepository(t, document)
	previous, err := os.Getwd()
	if err != nil {
		t.Fatal(err)
	}
	if err := os.Chdir(root); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := os.Chdir(previous); err != nil {
			t.Errorf("restore working directory: %v", err)
		}
	})

	for name, invocation := range map[string]func() error{
		"status":  func() error { return run(cli{status: true}, nil) },
		"history": func() error { return run(cli{history: "all"}, nil) },
		"stop":    func() error { return run(cli{stop: true}, []string{"user-stop: fixture"}) },
		"control": func() error {
			return run(cli{contain: "device-instability", lane: "fixture"}, []string{"fixture"})
		},
	} {
		t.Run(name, func(t *testing.T) {
			if err := invocation(); err != nil {
				t.Fatalf("non-dispatch command required broken completion evidence: %v", err)
			}
		})
	}
	if _, err := testCompletionAuthority(t, document); err == nil || !strings.Contains(err.Error(), "lacks gated ancestor completion evidence") {
		t.Fatalf("fixture completion evidence is not broken: %v", err)
	}
}
