package main

import (
	"os"
	"path/filepath"
	"testing"
)

func TestDocumentationFreshnessRejectsStaleRankings(t *testing.T) {
	root := t.TempDir()
	docs := filepath.Join(root, "docs")
	if err := os.Mkdir(docs, 0o755); err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(docs, "assessment.md")
	stale := "# Assessment\n\n## Execution Program\n\n1. Repair a defect that is already fixed.\n"
	if err := os.WriteFile(path, []byte(stale), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := documentationFreshness(root); err == nil {
		t.Fatal("unmarked work ranking accepted as current evidence")
	}
	historical := historicalRankingMarker + "\n" + currentWorkMarker + "\n\n> Historical only; verify current work in docs/plan.json.\n\n" + stale
	if err := os.WriteFile(path, []byte(historical), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := documentationFreshness(root); err != nil {
		t.Fatalf("marked historical ranking rejected: %v", err)
	}
}
