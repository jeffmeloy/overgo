package main

import (
	"os"
	"path/filepath"
	"testing"
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

	if err := checkPlanBinding(dir, "r0/first"); err != nil {
		t.Fatalf("the current open step must pass: %v", err)
	}
	for _, bad := range []string{"r0/second", "other/x", "", "noslash", "r0/"} {
		if err := checkPlanBinding(dir, bad); err == nil {
			t.Fatalf("-plan %q must be REFUSED (not the current open step / malformed)", bad)
		}
	}
}
