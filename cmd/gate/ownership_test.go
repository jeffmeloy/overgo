package main

import (
	"os"
	"path/filepath"
	"testing"

	"overgo/internal/plan"
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
	cache := retryCache{Environment: "env-a", Steps: map[string]phaseCache{"test": {Input: "source", Outcome: "succeeded"}}}
	if !retryReusable(cache, "env-a") {
		t.Fatal("matching environment did not reuse cache")
	}
	if retryReusable(cache, "env-b") {
		t.Fatal("retry cache crossed environment identity")
	}
}

func TestPhaseCacheIgnoresUnownedPlanChanges(t *testing.T) {
	root := t.TempDir()
	files := map[string]string{
		"internal/feature.go":       "package internal\nfunc Feature() {}\n",
		"compatibility.json":        `{}`,
		"docs/COMPATIBILITY.md":     "current claims\n",
		plan.Path:                   "{\"campaign\":\"before\"}\n",
		"cmd/compatibility/main.go": "package main\n",
	}
	for path, content := range files {
		resolved := filepath.Join(root, filepath.FromSlash(path))
		if err := os.MkdirAll(filepath.Dir(resolved), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(resolved, []byte(content), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	paths := make([]string, 0, len(files))
	for path := range files {
		paths = append(paths, path)
	}
	buildBefore, err := fingerprintPhaseInputs(root, "build", paths, nil)
	if err != nil {
		t.Fatal(err)
	}
	claimsBefore, err := fingerprintPhaseInputs(root, "claims", paths, nil)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(root, "docs", "COMPATIBILITY.md"), []byte("refreshed claims\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	buildAfter, _ := fingerprintPhaseInputs(root, "build", paths, nil)
	claimsAfter, _ := fingerprintPhaseInputs(root, "claims", paths, nil)
	if buildBefore != buildAfter {
		t.Fatal("documentation invalidated build inputs")
	}
	if claimsBefore == claimsAfter {
		t.Fatal("compatibility documentation did not invalidate claim inputs")
	}
	testBefore, _ := fingerprintPhaseInputs(root, "test", paths, nil)
	if err := os.WriteFile(filepath.Join(root, filepath.FromSlash(plan.Path)), []byte("{\"campaign\":\"after\"}\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	testAfter, _ := fingerprintPhaseInputs(root, "test", paths, nil)
	if testBefore != testAfter {
		t.Fatal("plan-only acceptance change invalidated package tests")
	}
}
