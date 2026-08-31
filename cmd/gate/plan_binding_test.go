package main

import (
	"os"
	"path/filepath"
	"testing"

	"overgo/internal/overgodb"
	"overgo/internal/plan"
)

// TestEnforcePlanBindingRefusesOffPlan pins the commit-gate's plan binding: a
// commit may only serve the plan's current open step. This is the enforcement
// that makes off-plan work impossible to commit (the failure that motivated it).
func TestEnforcePlanBindingRefusesOffPlan(t *testing.T) {
	dir := t.TempDir()
	if err := os.MkdirAll(filepath.Join(dir, "docs"), 0o755); err != nil {
		t.Fatal(err)
	}
	// Current open step is r0/first (r0/second is open too but comes later).
	js := `{"campaign":"t","doctrine":"d","items":[` +
		`{"id":"r0","status":"open","steps":[` +
		`{"id":"first","status":"open","verify":"go test ./..."},` +
		`{"id":"second","status":"open","verify":"go test ./..."}]}]}`
	if err := os.WriteFile(filepath.Join(dir, "docs", "plan.json"), []byte(js), 0o644); err != nil {
		t.Fatal(err)
	}
	initializePlanBindingRepo(t, dir)

	if err := checkPlanBinding(dir, "r0/first"); err != nil {
		t.Fatalf("the current open step must pass: %v", err)
	}
	for _, bad := range []string{"r0/second", "other/x", "", "noslash", "r0/"} {
		if err := checkPlanBinding(dir, bad); err == nil {
			t.Fatalf("-plan %q must be REFUSED (not the current open step / malformed)", bad)
		}
	}
}

func TestPlanBindingRole(t *testing.T) {
	dir := t.TempDir()
	if err := os.MkdirAll(filepath.Join(dir, "docs"), 0o755); err != nil {
		t.Fatal(err)
	}
	js := `{"campaign":"t","doctrine":"d","items":[` +
		`{"id":"developer","owner":"developer","status":"open","steps":[{"id":"do","status":"open","verify":"go test ./..."}]},` +
		`{"id":"sqa","owner":"sqa","status":"open","steps":[{"id":"do","status":"open","verify":"go test ./..."}]}]}`
	if err := os.WriteFile(filepath.Join(dir, "docs", "plan.json"), []byte(js), 0o644); err != nil {
		t.Fatal(err)
	}
	initializePlanBindingRepo(t, dir)
	t.Setenv(plan.AutomationRoleEnvironment, "sqa")
	if err := checkPlanBinding(dir, "sqa/do"); err != nil {
		t.Fatalf("owned plan binding: %v", err)
	}
	if err := checkPlanBinding(dir, "developer/do"); err == nil {
		t.Fatal("foreign role plan binding passed")
	}
}

func initializePlanBindingRepo(t *testing.T, repository string) {
	t.Helper()
	runGitFixture(t, repository, "init", "-q")
	runGitFixture(t, repository, "config", "user.email", "binding@example.invalid")
	runGitFixture(t, repository, "config", "user.name", "Binding Test")
	runGitFixture(t, repository, "add", "--", plan.Path)
	runGitFixture(t, repository, "commit", "-q", "-m", "baseline")
	store, err := overgodb.Open(filepath.Join(repository, gateStorePath))
	if err != nil {
		t.Fatal(err)
	}
	if err := store.Close(); err != nil {
		t.Fatal(err)
	}
}
