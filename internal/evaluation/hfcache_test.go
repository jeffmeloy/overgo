package evaluation

import (
	"context"
	"os"
	"path/filepath"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/dataset"
	"overgo/internal/overgodb"
)

// TestHFCacheBenchmarkCatalog pins the cache-scan import: a
// HuggingFace-layout cache derives its benchmark seeds from the
// directory structure alone, unrecognized directories and scratch
// files are skipped, the derived seeds import through the arrow
// reader, and the active benchmark catalog publishes with one entry
// per split.
func TestHFCacheBenchmarkCatalog(t *testing.T) {
	fixture, err := os.ReadFile(filepath.Join("..", "dataset", "testdata", "mmlu-dev.arrow"))
	if err != nil {
		t.Fatal(err)
	}
	root := t.TempDir()
	place := func(parts ...string) {
		t.Helper()
		path := filepath.Join(append([]string{root}, parts...)...)
		if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(path, fixture, 0o644); err != nil {
			t.Fatal(err)
		}
	}
	place("cais___mmlu", "abstract_algebra", "0.0.0", "aa11", "mmlu-test.arrow")
	place("cais___mmlu", "abstract_algebra", "0.0.0", "aa11", "cache-deadbeef.arrow")
	place("TAUR-Lab___mu_sr", "default", "0.0.0", "bb22", "mu_sr-murder_mysteries.arrow")
	place("unrelated___scratch", "default", "0.0.0", "cc33", "scratch-test.arrow")

	seeds, err := DeriveHFCacheBenchmarks(root)
	if err != nil {
		t.Fatal(err)
	}
	if len(seeds) != 2 {
		t.Fatalf("seeds = %+v, want mmlu test and musr murder_mysteries only", seeds)
	}
	if seeds[0].Name != "mmlu/abstract_algebra/test" || seeds[0].Spec.Split != "test" ||
		seeds[0].Spec.Format != dataset.BenchmarkFormatArrow ||
		seeds[1].Name != "musr/default/murder_mysteries" {
		t.Fatalf("seeds = %+v", seeds)
	}

	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	// The musr seed binds fields the mmlu fixture bytes do not carry;
	// importing it must refuse rather than fabricate, so the catalog
	// publishes from the mmlu seed alone.
	ctx := context.Background()
	if _, _, err := CatalogHFCacheBenchmarks(ctx, store, root); err == nil {
		t.Fatal("mismatched fields imported")
	}
	if err := os.RemoveAll(filepath.Join(root, "TAUR-Lab___mu_sr")); err != nil {
		t.Fatal(err)
	}
	catalog, count, err := CatalogHFCacheBenchmarks(ctx, store, root)
	if err != nil {
		t.Fatal(err)
	}
	if count != 1 || !catalog.Valid() {
		t.Fatalf("catalog = (%s, %d)", catalog, count)
	}
	active, bound, err := artifact.ResolveAlias(ctx, store, "evaluation/catalogs/active")
	if err != nil || !bound || active != catalog {
		t.Fatalf("active alias = (%s, %t, %v)", active, bound, err)
	}
}
