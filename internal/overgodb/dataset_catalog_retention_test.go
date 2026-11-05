package overgodb_test

import (
	"os"
	"path/filepath"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/dataset"
	"overgo/internal/overgodb"
	"overgo/internal/testcheck"
)

func TestDatasetCatalogCompactionRetainsAuthorityAndLocation(t *testing.T) {
	t.Parallel()
	ctx, root := t.Context(), t.TempDir()
	path := filepath.Join(root, "fixture.txt")
	testcheck.NoError(t, os.WriteFile(path, []byte("x"), 0o600))
	inventory, err := dataset.NewInventory([]dataset.InventoryFile{{
		Path: filepath.Base(path), OriginalName: filepath.Base(path), Modality: "text", Format: "txt", Bytes: 1,
	}})
	testcheck.NoError(t, err)
	version, err := dataset.NewVersion([]dataset.Asset{{Name: "inventory", Artifact: inventory.ID, Records: 1}})
	testcheck.NoError(t, err)
	catalog, err := dataset.NewCatalog([]dataset.CatalogEntry{{
		Name: "fixture", Dataset: version.ID, Inventory: inventory.ID, StorageKind: "file", Source: "test",
		Modality: "text", Formats: []string{"txt"}, Files: 1, Bytes: 1,
	}})
	testcheck.NoError(t, err)
	location, err := artifact.CanonicalLocalLocation(version.ID, artifact.LocationFile, path)
	testcheck.NoError(t, err)
	source, err := overgodb.Create(filepath.Join(root, "source"))
	testcheck.NoError(t, err)
	if _, err := dataset.PublishCatalog(ctx, source, dataset.CompiledCatalog{
		Catalog: catalog, Datasets: []dataset.Document{version}, Inventories: []dataset.Inventory{inventory},
		Locations: []artifact.Location{location},
	}); err != nil {
		t.Fatal(err)
	}
	defer source.Close()
	if _, err := overgodb.Release(ctx, source, nil, overgodb.RetentionPolicy{}, releaseEverything); err != nil {
		t.Fatal(err)
	}
	coverage, err := dataset.InspectCatalog(ctx, source)
	testcheck.NoError(t, err)
	if !coverage.Complete || coverage.Available != coverage.Registered {
		t.Fatalf("coverage = %+v", coverage)
	}
}
