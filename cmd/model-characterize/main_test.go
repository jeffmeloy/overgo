package main

import (
	"os"
	"path/filepath"
	"testing"

	"overgo/internal/modelartifact"
	"overgo/internal/overgodb"
	"overgo/internal/testcheck"
)

// TestCharacterizeCatalogEmptyStore exercises the catalog wiring end to end over
// an empty store: no servable models, no error, zero characterized. A populated
// servable model requires the full registration chain; the per-model resolution
// composes independently tested functions (discovery.Servable,
// ResolveModelDefinition, MeasureAtLocation).
func TestCharacterizeCatalogEmptyStore(t *testing.T) {
	dir := filepath.Join(t.TempDir(), "store")
	testcheck.NoError(t, os.MkdirAll(dir, 0o755))
	store, err := overgodb.Create(dir)
	testcheck.NoError(t, err)
	defer store.Close()

	count, err := characterizeCatalog(
		t.Context(), store, 4096,
		modelartifact.MeasurementPolicy{MaxSamplesPerTensor: 256, MaxReadBytes: 1 << 20},
	)
	testcheck.NoError(t, err)
	if count != 0 {
		t.Errorf("empty store characterized %d models, want 0", count)
	}
}
