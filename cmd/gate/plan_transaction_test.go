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
	preAdvance, rollback, err := advancePlanFile(repo, "automation/first")
	if err != nil {
		t.Fatal(err)
	}
	if preAdvance.Items[0].Steps[0].ID != "first" {
		t.Fatalf("pre-advance plan = %+v", preAdvance.Items)
	}
	advanced, err := plan.Load(path)
	if err != nil {
		t.Fatal(err)
	}
	if len(advanced.Items) != 1 || len(advanced.Items[0].Steps) != 1 || advanced.Items[0].Steps[0].ID != "second" {
		t.Fatalf("advanced plan = %+v", advanced.Items)
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
