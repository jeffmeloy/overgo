package main

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/dataset"
	"overgo/internal/model"
	"overgo/internal/modelrecipe"
	"overgo/internal/overgodb"
)

func TestProfileCatalogQueryReportsExactCoverage(t *testing.T) {
	repository := t.TempDir()
	store, err := overgodb.Open(repository)
	if err != nil {
		t.Fatal(err)
	}
	if err := store.Close(); err != nil {
		t.Fatal(err)
	}
	var emptyOutput bytes.Buffer
	if err := run([]string{"-repo", repository, "-profiles", "-json"}, &emptyOutput); err != nil {
		t.Fatal(err)
	}
	var empty modelrecipe.ProfileCatalogCoverage
	if err := json.Unmarshal(emptyOutput.Bytes(), &empty); err != nil {
		t.Fatal(err)
	}
	if empty.Complete || empty.Published != 0 || empty.Registered != len(model.SupportedArchitectures()) {
		t.Fatalf("empty coverage = %+v", empty)
	}
	store, err = overgodb.Open(repository)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := modelrecipe.PublishArchitectureProfileCatalog(t.Context(), store); err != nil {
		t.Fatal(err)
	}
	if err := store.Close(); err != nil {
		t.Fatal(err)
	}
	var output bytes.Buffer
	if err := run([]string{"-repo", repository, "-profiles", "-json"}, &output); err != nil {
		t.Fatal(err)
	}
	var coverage modelrecipe.ProfileCatalogCoverage
	if err := json.Unmarshal(output.Bytes(), &coverage); err != nil {
		t.Fatal(err)
	}
	if !coverage.Complete || coverage.Registered != len(model.SupportedArchitectures()) ||
		coverage.Published != coverage.Registered || len(coverage.Entries) != coverage.Registered {
		t.Fatalf("coverage = %+v", coverage)
	}
}

func TestDatasetCatalogQueryReportsExactCoverage(t *testing.T) {
	root, path := t.TempDir(), filepath.Join(t.TempDir(), "fixture.txt")
	if err := os.WriteFile(path, []byte("x"), 0o600); err != nil {
		t.Fatal(err)
	}
	store, err := overgodb.Open(root)
	if err != nil {
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
	if _, err := dataset.PublishCatalog(t.Context(), store, dataset.CompiledCatalog{
		Catalog: catalog, Datasets: []dataset.Document{version}, Inventories: []dataset.Inventory{inventory},
		Locations: []artifact.Location{location},
	}); err != nil {
		t.Fatal(err)
	}
	if err := store.Close(); err != nil {
		t.Fatal(err)
	}
	var output bytes.Buffer
	if err := run([]string{"-repo", root, "-datasets", "-json"}, &output); err != nil {
		t.Fatal(err)
	}
	var coverage dataset.CatalogCoverage
	if err := json.Unmarshal(output.Bytes(), &coverage); err != nil {
		t.Fatal(err)
	}
	if !coverage.Complete || coverage.Published != coverage.Registered || coverage.Available != coverage.Registered {
		t.Fatalf("coverage = %+v", coverage)
	}
}

func TestRunQueriesSeededStore(t *testing.T) {
	root := t.TempDir()
	store, err := overgodb.Open(root)
	if err != nil {
		t.Fatal(err)
	}
	id, _ := artifact.IdentifyBytes(artifact.KindDataset, []byte("dataset"))
	if _, err := store.Commit(t.Context(), artifact.Batch{
		Key:       "fixture/query-cli",
		Artifacts: []artifact.Descriptor{{ID: id, Size: 7}},
		Aliases:   []artifact.AliasBinding{{Name: "dataset/current", Target: id}},
	}); err != nil {
		t.Fatal(err)
	}
	if err := store.Close(); err != nil {
		t.Fatal(err)
	}
	var text bytes.Buffer
	if err := run([]string{"-repo", root, "-kind", "dataset"}, &text); err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(text.String(), id.String()) || !strings.Contains(text.String(), "dataset/current") {
		t.Fatalf("text output = %q", text.String())
	}
	var encoded bytes.Buffer
	if err := run([]string{"-repo", root, "-alias", "dataset/current", "-json"}, &encoded); err != nil {
		t.Fatal(err)
	}
	var result overgodb.QueryResult
	if err := json.Unmarshal(encoded.Bytes(), &result); err != nil {
		t.Fatal(err)
	}
	if len(result.Artifacts) != 1 || result.Artifacts[0].ID != id {
		t.Fatalf("JSON result = %+v", result)
	}
}

func TestRunRejectsUnboundedAndInvalidFilters(t *testing.T) {
	if err := run(nil, &bytes.Buffer{}); err == nil {
		t.Fatal("missing repository accepted")
	}
	if err := run([]string{"-repo", "x", "-limit", "0"}, &bytes.Buffer{}); err == nil {
		t.Fatal("zero result bound accepted")
	}
	if _, err := parseFollow("sideways"); err == nil {
		t.Fatal("invalid follow direction accepted")
	}
}
