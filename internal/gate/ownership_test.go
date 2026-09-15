package gate

import (
	"encoding/json"
	"maps"
	"os"
	"path/filepath"
	"slices"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/automationcheck"
	"overgo/internal/dataroot"
	"overgo/internal/plan"
	"overgo/internal/repoanalysis"
)

func TestGateScopeSnapshot(t *testing.T) {
	t.Parallel()
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
	t.Parallel()
	g := liveRepositoryFixture(t).context("internal/model/architecture_profiles.json")
	scope, err := g.deriveTestScope()
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Contains(scope.direct, "overgo/internal/model") {
		t.Fatalf("embedded architecture catalog owner missing: %v", scope.direct)
	}
	if _, err := os.Stat(filepath.Join(g.repo, "internal", "model", "architecture_profiles.json")); err != nil {
		t.Fatal(err)
	}
}

func TestEnvironmentBoundRetry(t *testing.T) {
	t.Parallel()
	environment, _ := artifact.IdentifyBytes(artifact.KindProfile, []byte("env-a"))
	other, _ := artifact.IdentifyBytes(artifact.KindProfile, []byte("env-b"))
	cache := automationcheck.NewEvidenceCache(environment)
	if !cache.Reusable(environment) {
		t.Fatal("matching environment did not reuse cache")
	}
	if cache.Reusable(other) {
		t.Fatal("retry cache crossed environment identity")
	}
}

func TestGateDataRootCacheIdentity(t *testing.T) {
	if isolatedProcess(t) {
		return
	}
	repo := t.TempDir()
	t.Setenv(dataroot.Env, "")
	t.Setenv("OVERGO_AUDIO_REFERENCE_STORE", "")
	identity := func(t *testing.T) artifact.ID {
		t.Helper()
		environment, err := discoverEnvironment(repo)
		if err != nil {
			t.Fatal(err)
		}
		return environment.ID
	}
	baseline := identity(t)
	cache := automationcheck.NewEvidenceCache(baseline)
	if !cache.Reusable(identity(t)) {
		t.Fatal("unchanged data roots invalidated evidence")
	}
	t.Setenv("OVERGO_AUDIO_REFERENCE_STORE", t.TempDir())
	if cache.Reusable(identity(t)) {
		t.Fatal("changed audio reference store reused prior evidence")
	}
	referenceIdentity := identity(t)
	t.Setenv("OVERGO_AUDIO_REFERENCE_STORE", t.TempDir())
	if identity(t) == referenceIdentity {
		t.Fatal("distinct audio reference stores shared an environment identity")
	}
	t.Setenv("OVERGO_AUDIO_REFERENCE_STORE", "")
	if identity(t) != baseline {
		t.Fatal("restored data roots did not restore environment identity")
	}
	for _, field := range []string{"store", "models", "datasets", "checkpoints"} {
		t.Run(field, func(t *testing.T) {
			data, err := json.Marshal(map[string]string{field: t.TempDir()})
			if err != nil {
				t.Fatal(err)
			}
			if err := os.WriteFile(filepath.Join(repo, dataroot.ConfigFile), data, 0o600); err != nil {
				t.Fatal(err)
			}
			if cache.Reusable(identity(t)) {
				t.Fatal("changed configured data root reused prior evidence")
			}
		})
	}
	first, second := t.TempDir(), t.TempDir()
	t.Setenv(dataroot.Env, first)
	firstIdentity := identity(t)
	t.Setenv(dataroot.Env, second)
	if identity(t) == firstIdentity {
		t.Fatal("changed explicit data root reused prior environment")
	}
	t.Setenv(dataroot.Env, first)
	if identity(t) != firstIdentity {
		t.Fatal("identical effective data roots produced unstable identity")
	}
	// The shared owner gives the explicit root precedence over local config.
	if err := os.WriteFile(filepath.Join(repo, dataroot.ConfigFile), []byte("invalid"), 0o600); err != nil {
		t.Fatal(err)
	}
	if identity(t) != firstIdentity {
		t.Fatal("inactive local configuration changed explicit-root identity")
	}
	t.Setenv(dataroot.Env, filepath.Join(first, "absent"))
	if _, err := discoverEnvironment(repo); err == nil {
		t.Fatal("invalid explicit data root accepted")
	}
	t.Setenv(dataroot.Env, "")
	if _, err := discoverEnvironment(repo); err == nil {
		t.Fatal("invalid active data-root configuration accepted")
	}
}

func TestPhaseCacheIgnoresUnownedPlanChanges(t *testing.T) {
	t.Parallel()
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
	paths := slices.Collect(maps.Keys(files))
	buildBefore, err := fingerprintPhaseInputs(root, "build", paths)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(root, "docs", "COMPATIBILITY.md"), []byte("refreshed claims\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	buildAfter, _ := fingerprintPhaseInputs(root, "build", paths)
	if buildBefore != buildAfter {
		t.Fatal("documentation invalidated build inputs")
	}
	testBefore, _ := fingerprintPhaseInputs(root, "test", paths)
	if err := os.WriteFile(filepath.Join(root, filepath.FromSlash(plan.Path)), []byte("{\"campaign\":\"after\"}\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	testAfter, _ := fingerprintPhaseInputs(root, "test", paths)
	if testBefore != testAfter {
		t.Fatal("plan-only acceptance change invalidated package tests")
	}
}
