//go:build modeltest

package discovery

import (
	"testing"

	"overgo/internal/dataroot"
	"overgo/internal/overgodbtest"
	"overgo/internal/testcheck"
	"overgo/internal/testutil"
)

// TestTheCatalogBoundHoldsTheStore reads the live store's whole servable catalog under CatalogBound, the bound the
// long-form guard, the evaluator and the smoke check read it with: a read the bound truncates is refused, so every one
// of them would refuse the store (258 models on 2026-10-31 refused a bound of 256).
func TestTheCatalogBoundHoldsTheStore(t *testing.T) {
	roots := testcheck.Must(dataroot.Resolve(testutil.RepoRoot(t)))(t)
	store := overgodbtest.OpenReadOnly(t, roots.Store)
	entries := testcheck.Must(ServableWithMemo(t.Context(), store, CatalogBound, LoadMemo(t.Context(), store)))(t)
	t.Logf("%d servable entries under a bound of %d", len(entries), CatalogBound)
}
