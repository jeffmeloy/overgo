// Package overgodb_test verifies external OvergoDB contracts.
package overgodb_test

import (
	"path/filepath"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/modelrecipe"
	"overgo/internal/overgodb"
)

// releaseEverything admits every class: a consumer that still works after it
// needs nothing outside the live set.
func releaseEverything(artifact.Descriptor) bool { return true }

func TestProfileCatalogReleaseRetainsArchitectureAuthority(t *testing.T) {
	ctx := t.Context()
	source, err := overgodb.Open(filepath.Join(t.TempDir(), "source"))
	if err != nil {
		t.Fatal(err)
	}
	defer source.Close()
	publication, err := modelrecipe.PublishArchitectureProfileCatalog(ctx, source)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := overgodb.Release(ctx, source, nil, overgodb.RetentionPolicy{}, releaseEverything); err != nil {
		t.Fatal(err)
	}
	coverage, err := modelrecipe.InspectArchitectureProfileCatalog(ctx, source)
	if err != nil {
		t.Fatal(err)
	}
	if !coverage.Complete || coverage.Published != publication.Coverage.Registered {
		t.Fatalf("released coverage = %+v", coverage)
	}
}
