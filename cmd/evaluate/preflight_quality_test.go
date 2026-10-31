package main

import (
	"errors"
	"os"
	"path/filepath"
	"testing"

	"overgo/internal/clioptions"
	"overgo/internal/evaluation"
	"overgo/internal/inference"
	"overgo/internal/modelrecipe"
	"overgo/internal/overgodb"
	"overgo/internal/recipe"
	"overgo/internal/servingtest"
	"overgo/internal/testutil"
)

// TestAQualityCellsPreflightReachesItsSetup holds a quality cell's preflight
// to the setup its session does before the model loads: the model's recipe
// resolved, the session's campaign and its authorities, and the family's
// derived suites. A family the store holds passes; one it does not is refused
// there, as the evaluation would refuse it, where the preflight had stopped at
// the verifying commit. The campaign is the session's own and loads no weights:
// a case put to it is refused.
func TestAQualityCellsPreflightReachesItsSetup(t *testing.T) {
	root := testutil.RepoRoot(t)
	repository := t.TempDir()
	store, err := overgodb.Open(repository)
	if err != nil {
		t.Fatal(err)
	}
	model := testutil.HermeticLlamaGGUF(t, 64)
	if err := servingtest.PublishActiveGGUFWithPolicy(t.Context(), store, model, recipe.PlacementHost, modelrecipe.DecodeSessionCapacity, recipe.ResidencyHostReference); err != nil {
		t.Fatal(err)
	}
	fixture, err := os.ReadFile(filepath.Join(root, "internal", "dataset", "testdata", "mmlu-dev.arrow"))
	if err != nil {
		t.Fatal(err)
	}
	cache := t.TempDir()
	target := filepath.Join(cache, "cais___mmlu", "abstract_algebra", "0.0.0", "aa11", "mmlu-test.arrow")
	if err := os.MkdirAll(filepath.Dir(target), 0o700); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(target, fixture, clioptions.OutputFileMode); err != nil {
		t.Fatal(err)
	}
	if _, _, err := evaluation.CatalogHFCacheBenchmarks(t.Context(), store, cache); err != nil {
		t.Fatal(err)
	}
	if err := store.Close(); err != nil {
		t.Fatal(err)
	}

	shared := manifest{Repository: repository, CodeCommit: hostedTestCommit}
	if err := derivedReady(t.Context(), shared, model, "mmlu"); err != nil {
		t.Fatalf("a family the store holds was refused before the model loads: %v", err)
	}
	if err := derivedReady(t.Context(), shared, model, "absent-family"); err == nil {
		t.Fatal("a family the store does not hold passed the preflight")
	}
	if _, _, err := (unloadedRuntime{}).Generate(t.Context(), "", inference.GenerateOptions{}); !errors.Is(err, errUnloaded) {
		t.Fatalf("the preflight's runtime answered a case: %v", err)
	}
}
