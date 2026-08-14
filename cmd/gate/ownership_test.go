package main

import (
	"os"
	"path/filepath"
	"testing"

	"overgo/internal/repoanalysis"
)

func TestGateScopeSnapshot(t *testing.T) {
	dirty := []repoanalysis.DirtyPath{{Path: "new.go", OriginalPath: "old.go", IndexStatus: "R"}, {Path: "other.go", WorktreeStatus: "M"}}
	visible, unplanned := scopeDirty([]string{"new.go"}, dirty)
	if !visible["new.go"] || !visible["old.go"] || !visible["other.go"] {
		t.Fatalf("visible = %v", visible)
	}
	if len(unplanned) != 2 || unplanned[0] != "old.go" || unplanned[1] != "other.go" {
		t.Fatalf("unplanned = %v", unplanned)
	}
}

func TestNonGoOwnershipGateScope(t *testing.T) {
	repo, err := filepath.Abs(filepath.Join("..", ".."))
	if err != nil {
		t.Fatal(err)
	}
	g := gateContext{repo: repo, paths: []string{"internal/model/architecture_profiles.json"}}
	owners, err := g.directChangedPackages()
	if err != nil {
		t.Fatal(err)
	}
	if !owners["overgo/internal/model"] {
		t.Fatalf("embedded architecture catalog owner missing: %v", owners)
	}
	if _, err := os.Stat(filepath.Join(repo, "internal", "model", "architecture_profiles.json")); err != nil {
		t.Fatal(err)
	}
}

func TestEnvironmentBoundRetry(t *testing.T) {
	cache := retryCache{TreeKey: "tree", Environment: "env-a", Steps: map[string]string{"test": "succeeded"}}
	if !retryReusable(cache, "tree", "env-a") {
		t.Fatal("matching environment did not reuse cache")
	}
	if retryReusable(cache, "tree", "env-b") {
		t.Fatal("retry cache crossed environment identity")
	}
}
