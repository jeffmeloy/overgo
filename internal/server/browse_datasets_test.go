package server

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/dataset"
	"overgo/internal/overgodb"
)

func TestBrowseDatasetsUsesCatalog(t *testing.T) {
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = store.Close() })
	publishDatasetFixture(t, store)
	handler := newTestHandlerForRepository(t, store, &fakeGenerator{})
	response := serveTestRequest(handler, http.MethodGet, "/datasets", "")
	if response.Code != http.StatusOK {
		t.Fatalf("status=%d body=%s", response.Code, response.Body.String())
	}
	var result browseDatasetsResponse
	if err := json.Unmarshal(response.Body.Bytes(), &result); err != nil {
		t.Fatal(err)
	}
	if result.Count != 1 || len(result.Datasets) != result.Count || result.Datasets[0].Name != "fixture" ||
		result.Datasets[0].Source != "test" || !result.Datasets[0].Available || result.Catalog == "" {
		t.Fatalf("result = %+v", result)
	}
}

func TestBrowseDatasetsUnconfigured(t *testing.T) {
	handler := newTestHandler(t, &fakeGenerator{})
	response := serveTestRequest(handler, http.MethodGet, "/datasets", "")
	if response.Code != http.StatusNotImplemented {
		t.Fatalf("status=%d, want 501", response.Code)
	}
}

func TestBrowseDatasetsRequiresActiveCatalog(t *testing.T) {
	handler := newTestHandlerWithRepository(t, &fakeGenerator{})
	response := serveTestRequest(handler, http.MethodGet, "/datasets", "")
	if response.Code != http.StatusNotFound {
		t.Fatalf("status=%d, want 404", response.Code)
	}
}

func TestBrowseDatasetsRejectsNonGet(t *testing.T) {
	handler := newTestHandlerWithRepository(t, &fakeGenerator{})
	response := httptest.NewRecorder()
	handler.ServeHTTP(response, httptest.NewRequest(http.MethodPost, "/datasets", nil))
	if response.Code != http.StatusMethodNotAllowed {
		t.Fatalf("status=%d, want 405", response.Code)
	}
}

func publishDatasetFixture(t *testing.T, store *overgodb.Store) {
	t.Helper()
	path := filepath.Join(t.TempDir(), "fixture.txt")
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
		Name: "fixture", Dataset: version.ID, Inventory: inventory.ID, StorageKind: "file",
		Source: "test", Modality: "text", Formats: []string{"txt"}, Files: 1, Bytes: 1,
	}})
	if err != nil {
		t.Fatal(err)
	}
	location, err := artifact.CanonicalLocalLocation(version.ID, artifact.LocationFile, path)
	if err != nil {
		t.Fatal(err)
	}
	_, err = dataset.PublishCatalog(t.Context(), store, dataset.CompiledCatalog{
		Catalog: catalog, Datasets: []dataset.Document{version}, Inventories: []dataset.Inventory{inventory},
		Locations: []artifact.Location{location},
	})
	if err != nil {
		t.Fatal(err)
	}
}
