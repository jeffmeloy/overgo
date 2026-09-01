// Package overgodb_test verifies external OvergoDB contracts.
package overgodb_test

import (
	"path/filepath"
	"testing"

	"overgo/internal/modelrecipe"
	"overgo/internal/overgodb"
)

func TestProfileCatalogCompactionRetainsArchitectureAuthority(t *testing.T) {
	ctx := t.Context()
	root := t.TempDir()
	source, err := overgodb.Open(filepath.Join(root, "source"))
	if err != nil {
		t.Fatal(err)
	}
	publication, err := modelrecipe.PublishArchitectureProfileCatalog(ctx, source)
	if err != nil {
		t.Fatal(err)
	}
	destination := filepath.Join(root, "compact")
	if _, err := overgodb.Compact(ctx, source, destination, nil); err != nil {
		t.Fatal(err)
	}
	if err := source.Close(); err != nil {
		t.Fatal(err)
	}

	compacted, err := overgodb.OpenReadOnly(destination)
	if err != nil {
		t.Fatal(err)
	}
	defer compacted.Close()
	coverage, err := modelrecipe.InspectArchitectureProfileCatalog(ctx, compacted)
	if err != nil {
		t.Fatal(err)
	}
	if !coverage.Complete || coverage.Published != publication.Coverage.Registered {
		t.Fatalf("compacted coverage = %+v", coverage)
	}
}
