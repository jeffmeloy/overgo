// Package scratchmodeltest supplies published construction fixtures.
package scratchmodeltest

import (
	"context"
	"testing"

	"overgo/internal/overgodb"
	"overgo/internal/scratchmodel"
	"overgo/internal/testcheck"
)

// Profile publishes and returns the bootstrap profile.
func Profile(t testing.TB) scratchmodel.DerivationProfile {
	t.Helper()
	store, err := overgodb.Open(t.TempDir())
	testcheck.NoError(t, err)
	defer store.Close()
	profile, err := scratchmodel.PublishDerivationProfileCatalog(context.Background(), store)
	testcheck.NoError(t, err)
	return profile
}
