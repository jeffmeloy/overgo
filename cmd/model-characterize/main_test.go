package main

import (
	"context"
	"os"
	"path/filepath"
	"testing"

	"overgo/internal/modelartifact"
	"overgo/internal/overgodb"
)

// TestCharacterizeCatalogEmptyStore exercises the catalog wiring end to end over
// an empty store: no servable models, no error, zero characterized. A populated
// servable model requires the full registration chain; the per-model resolution
// composes independently tested functions (discovery.Servable,
// ResolveModelDefinition, MeasureAtLocation).
func TestCharacterizeCatalogEmptyStore(t *testing.T) {
	dir := filepath.Join(t.TempDir(), "store")
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatal(err)
	}
	store, err := overgodb.Open(dir)
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()

	count, err := characterizeCatalog(
		context.Background(), store, 4096,
		modelartifact.MeasurementPolicy{MaxSamplesPerTensor: 256, MaxReadBytes: 1 << 20},
	)
	if err != nil {
		t.Fatal(err)
	}
	if count != 0 {
		t.Errorf("empty store characterized %d models, want 0", count)
	}
}
