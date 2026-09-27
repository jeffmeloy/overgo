package main

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"overgo/internal/authoritylock"
	"overgo/internal/plan"
)

func TestPlanMutationRefusesSharedGateLock(t *testing.T) {
	t.Parallel()
	root := t.TempDir()
	document := mutationPlan(t, "before")
	path := saveMutationPlan(t, root, document)
	before, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	lock, err := authoritylock.Acquire(root)
	if err != nil {
		t.Fatal(err)
	}
	defer lock.Close()
	called := false
	err = withPlanMutation(root, func(plan.Plan) error {
		called = true
		return nil
	})
	if err == nil || !strings.Contains(err.Error(), "mutation is active") || called {
		t.Fatalf("locked plan mutation = (called=%t, err=%v)", called, err)
	}
	after, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if string(after) != string(before) {
		t.Fatal("refused plan mutation changed plan bytes")
	}
}

func TestPlanMutationReloadsAfterAcquiringAuthority(t *testing.T) {
	t.Parallel()
	root := t.TempDir()
	stale := mutationPlan(t, "stale")
	current := mutationPlan(t, "current")
	saveMutationPlan(t, root, current)
	seen := ""
	if err := withPlanMutation(root, func(document plan.Plan) error {
		seen = document.Items[0].Title
		return nil
	}); err != nil {
		t.Fatal(err)
	}
	if seen != "current" || stale.Items[0].Title != "stale" {
		t.Fatalf("mutation saw %q from stale snapshot %+v", seen, stale.Items[0])
	}
	if err := withPlanMutation(root, nil); err == nil {
		t.Fatal("nil mutation callback was accepted")
	}
}

func saveMutationPlan(t *testing.T, root string, document plan.Plan) string {
	t.Helper()
	path := filepath.Join(root, filepath.FromSlash(plan.Path))
	if err := os.MkdirAll(filepath.Dir(path), 0o700); err != nil {
		t.Fatal(err)
	}
	if err := plan.Save(path, document); err != nil {
		t.Fatal(err)
	}
	return path
}

func mutationPlan(t *testing.T, title string) plan.Plan {
	t.Helper()
	return plan.Plan{Items: []plan.Item{{
		ID: "row", Title: title, Status: plan.StatusOpen,
		Steps: []plan.Step{{ID: "do", Status: plan.StatusOpen, Verify: "go test ./cmd/plan"}},
	}}}
}
