package main

import (
	"os"
	"path/filepath"
	"testing"

	"overgo/internal/modelartifact"
	"overgo/internal/overgodbtest"
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
	store := overgodbtest.Create(t, dir)

	count := testcheck.Must(characterizeCatalog(
		t.Context(), store, 4096,
		modelartifact.MeasurementPolicy{MaxSamplesPerTensor: 256, MaxReadBytes: 1 << 20},
	))(t)
	if count != 0 {
		t.Errorf("empty store characterized %d models, want 0", count)
	}
}
