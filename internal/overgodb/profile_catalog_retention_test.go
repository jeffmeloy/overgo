// Package overgodb_test verifies external OvergoDB contracts.
package overgodb_test

import (
	"path/filepath"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/modelrecipe"
	"overgo/internal/overgodb"
	"overgo/internal/testcheck"
)

// releaseEverything admits every class: a consumer that still works after it
// needs nothing outside the live set.
func releaseEverything(artifact.Descriptor) bool { return true }

func TestProfileCatalogReleaseRetainsArchitectureAuthority(t *testing.T) {
	t.Parallel()
	ctx := t.Context()
	source := testcheck.Must(overgodb.Create(filepath.Join(t.TempDir(), "source")))(t)
	defer source.Close()
	publication := testcheck.Must(modelrecipe.PublishArchitectureProfileCatalog(ctx, source))(t)
	if _, err := overgodb.Release(ctx, source, nil, overgodb.RetentionPolicy{}, releaseEverything); err != nil {
		t.Fatal(err)
	}
	coverage := testcheck.Must(modelrecipe.InspectArchitectureProfileCatalog(ctx, source))(t)
	if !coverage.Complete || coverage.Published != publication.Coverage.Registered {
		t.Fatalf("released coverage = %+v", coverage)
	}
}
