package overgodb_test

import (
	"os"
	"path/filepath"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/dataset"
	"overgo/internal/overgodb"
)

func TestDatasetCatalogCompactionRetainsAuthorityAndLocation(t *testing.T) {
	ctx, root := t.Context(), t.TempDir()
	path := filepath.Join(root, "fixture.txt")
	if err := os.WriteFile(path, []byte("x"), 0o600); err != nil {
		t.Fatal(err)
	}
	inventory, err := dataset.NewInventory([]dataset.InventoryFile{{
		Path: filepath.Base(path), OriginalName: filepath.Base(path), Modality: "text", Format: "txt", Bytes: 1,
	}})
	if err != nil {
		t.Fatal(err)
	}
	version, err := dataset.NewVersion([]dataset.Asset{{Name: "inventory", Artifact: inventory.ID, Records: 1}})
	if err != nil {
		t.Fatal(err)
	}
	catalog, err := dataset.NewCatalog([]dataset.CatalogEntry{{
		Name: "fixture", Dataset: version.ID, Inventory: inventory.ID, StorageKind: "file", Source: "test",
		Modality: "text", Formats: []string{"txt"}, Files: 1, Bytes: 1,
	}})
	if err != nil {
		t.Fatal(err)
	}
	location, err := artifact.CanonicalLocalLocation(version.ID, artifact.LocationFile, path)
	if err != nil {
		t.Fatal(err)
	}
	source, err := overgodb.Open(filepath.Join(root, "source"))
	if err != nil {
		t.Fatal(err)
	}
	if _, err := dataset.PublishCatalog(ctx, source, dataset.CompiledCatalog{
		Catalog: catalog, Datasets: []dataset.Document{version}, Inventories: []dataset.Inventory{inventory},
		Locations: []artifact.Location{location},
	}); err != nil {
		t.Fatal(err)
	}
	destination := filepath.Join(root, "compact")
	if _, err := overgodb.Compact(ctx, source, destination, nil); err != nil {
		t.Fatal(err)
	}
	if err := source.Close(); err != nil {
		t.Fatal(err)
	}
	compacted, err := overgodb.OpenReadOnly(destination)
	if err != nil {
		t.Fatal(err)
	}
	defer compacted.Close()
	coverage, err := dataset.InspectCatalog(ctx, compacted)
	if err != nil {
		t.Fatal(err)
	}
	if !coverage.Complete || coverage.Available != coverage.Registered {
		t.Fatalf("coverage = %+v", coverage)
	}
}
