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
	inventory := testcheck.Must(dataset.NewInventory([]dataset.InventoryFile{{
		Path: filepath.Base(path), OriginalName: filepath.Base(path), Modality: "text", Format: "txt", Bytes: 1,
	}}))(t)
	version := testcheck.Must(dataset.NewVersion([]dataset.Asset{{Name: "inventory", Artifact: inventory.ID, Records: 1}}))(t)
	catalog := testcheck.Must(dataset.NewCatalog([]dataset.CatalogEntry{{
		Name: "fixture", Dataset: version.ID, Inventory: inventory.ID, StorageKind: "file", Source: "test",
		Modality: "text", Formats: []string{"txt"}, Files: 1, Bytes: 1,
	}}))(t)
	location := testcheck.Must(artifact.CanonicalLocalLocation(version.ID, artifact.LocationFile, path))(t)
	source := testcheck.Must(overgodb.Create(filepath.Join(root, "source")))(t)
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
	coverage := testcheck.Must(dataset.InspectCatalog(ctx, source))(t)
	if !coverage.Complete || coverage.Available != coverage.Registered {
		t.Fatalf("coverage = %+v", coverage)
	}
}
