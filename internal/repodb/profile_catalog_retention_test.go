// Package repodb_test verifies external RepoDB contracts.
package repodb_test

import (
	"context"
	"path/filepath"
	"testing"

	"overgo/internal/modelrecipe"
	"overgo/internal/repodb"
)

func TestProfileCatalogCompactionRetainsArchitectureAuthority(t *testing.T) {
	ctx := context.Background()
	root := t.TempDir()
	source, err := repodb.Open(filepath.Join(root, "source"))
	if err != nil {
		t.Fatal(err)
	}
	publication, err := modelrecipe.PublishArchitectureProfileCatalog(ctx, source)
	if err != nil {
		t.Fatal(err)
	}
	destination := filepath.Join(root, "compact")
	if _, err := repodb.Compact(ctx, source, destination); err != nil {
		t.Fatal(err)
	}
	if err := source.Close(); err != nil {
		t.Fatal(err)
	}

	compacted, err := repodb.OpenReadOnly(destination)
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
