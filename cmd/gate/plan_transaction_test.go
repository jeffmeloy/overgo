package main

import (
	"os"
	"path/filepath"
	"testing"

	"overgo/internal/plan"
)

func TestGateCommitAdvancesPlanAtomically(t *testing.T) {
	repo := t.TempDir()
	if err := os.Mkdir(filepath.Join(repo, "docs"), 0o755); err != nil {
		t.Fatal(err)
	}
	document := plan.Plan{Campaign: "test", Doctrine: "test", Items: []plan.Item{{
		ID: "automation", Title: "automation", Status: "open", Steps: []plan.Step{
			{ID: "first", Title: "first", Status: "open", Verify: "go test ./..."},
			{ID: "second", Title: "second", Status: "open", Verify: "go test ./..."},
		},
	}}}
	path := filepath.Join(repo, filepath.FromSlash(plan.Path))
	if err := plan.Save(path, document); err != nil {
		t.Fatal(err)
	}
	original, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	rollback, err := advancePlanFile(repo, "automation/first")
	if err != nil {
		t.Fatal(err)
	}
	advanced, err := plan.Load(path)
	if err != nil {
		t.Fatal(err)
	}
	if item, step, ok := plan.Current(advanced, plan.UnassignedRole); !ok || item.ID != "automation" || step.ID != "second" {
		t.Fatalf("advanced current = %s/%s, open=%v", item.ID, step.ID, ok)
	}
	if err := rollback(); err != nil {
		t.Fatal(err)
	}
	restored, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if string(restored) != string(original) {
		t.Fatal("failed commit did not restore the original plan bytes")
	}
}
