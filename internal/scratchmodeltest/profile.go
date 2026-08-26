// Package scratchmodeltest supplies published construction fixtures.
package scratchmodeltest

import (
	"context"
	"testing"

	"overgo/internal/overgodb"
	"overgo/internal/scratchmodel"
)

// Profile publishes and returns the bootstrap profile.
func Profile(t testing.TB) scratchmodel.DerivationProfile {
	t.Helper()
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	profile, err := scratchmodel.PublishDerivationProfileCatalog(context.Background(), store)
	if err != nil {
		t.Fatal(err)
	}
	return profile
}
