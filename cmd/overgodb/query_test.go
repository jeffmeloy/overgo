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
	"overgo/internal/testcheck"
)

func TestProfileCatalogQueryReportsExactCoverage(t *testing.T) {
	repository := t.TempDir()
	store, err := overgodb.Create(repository)
	testcheck.NoError(t, err)
	testcheck.NoError(t, store.Close())
	var emptyOutput bytes.Buffer
	testcheck.NoError(t, runQuery([]string{"-repo", repository, "-profiles", "-json"}, &emptyOutput))
	var empty modelrecipe.ProfileCatalogCoverage
	testcheck.NoError(t, json.Unmarshal(emptyOutput.Bytes(), &empty))
	if empty.Complete || empty.Published != 0 || empty.Registered != len(model.SupportedArchitectures()) {
		t.Fatalf("empty coverage = %+v", empty)
	}
	store, err = overgodb.Create(repository)
	testcheck.NoError(t, err)
	if _, err := modelrecipe.PublishArchitectureProfileCatalog(t.Context(), store); err != nil {
		t.Fatal(err)
	}
	testcheck.NoError(t, store.Close())
	var output bytes.Buffer
	testcheck.NoError(t, runQuery([]string{"-repo", repository, "-profiles", "-json"}, &output))
	var coverage modelrecipe.ProfileCatalogCoverage
	testcheck.NoError(t, json.Unmarshal(output.Bytes(), &coverage))
	if !coverage.Complete || coverage.Registered != len(model.SupportedArchitectures()) ||
		coverage.Published != coverage.Registered || len(coverage.Entries) != coverage.Registered {
		t.Fatalf("coverage = %+v", coverage)
	}
}

func TestDatasetCatalogQueryReportsExactCoverage(t *testing.T) {
	root, path := t.TempDir(), filepath.Join(t.TempDir(), "fixture.txt")
	testcheck.NoError(t, os.WriteFile(path, []byte("x"), 0o600))
	store, err := overgodb.Create(root)
	testcheck.NoError(t, err)
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
	if _, err := dataset.PublishCatalog(t.Context(), store, dataset.CompiledCatalog{
		Catalog: catalog, Datasets: []dataset.Document{version}, Inventories: []dataset.Inventory{inventory},
		Locations: []artifact.Location{location},
	}); err != nil {
		t.Fatal(err)
	}
	testcheck.NoError(t, store.Close())
	var output bytes.Buffer
	testcheck.NoError(t, runQuery([]string{"-repo", root, "-datasets", "-json"}, &output))
	var coverage dataset.CatalogCoverage
	testcheck.NoError(t, json.Unmarshal(output.Bytes(), &coverage))
	if !coverage.Complete || coverage.Published != coverage.Registered || coverage.Available != coverage.Registered {
		t.Fatalf("coverage = %+v", coverage)
	}
}

func TestRunQueriesSeededStore(t *testing.T) {
	root := t.TempDir()
	store, err := overgodb.Create(root)
	testcheck.NoError(t, err)
	id, _ := artifact.IdentifyBytes(artifact.KindDataset, []byte("dataset"))
	if _, err := store.Commit(t.Context(), artifact.Batch{
		Key:       "fixture/query-cli",
		Artifacts: []artifact.Descriptor{{ID: id, Size: 7}},
		Aliases:   []artifact.AliasBinding{{Name: "dataset/current", Target: id}},
	}); err != nil {
		t.Fatal(err)
	}
	testcheck.NoError(t, store.Close())
	var text bytes.Buffer
	testcheck.NoError(t, runQuery([]string{"-repo", root, "-kind", "dataset"}, &text))
	if !strings.Contains(text.String(), id.String()) || !strings.Contains(text.String(), "dataset/current") {
		t.Fatalf("text output = %q", text.String())
	}
	var encoded bytes.Buffer
	testcheck.NoError(t, runQuery([]string{"-repo", root, "-alias", "dataset/current", "-json"}, &encoded))
	var result overgodb.QueryResult
	testcheck.NoError(t, json.Unmarshal(encoded.Bytes(), &result))
	if len(result.Artifacts) != 1 || result.Artifacts[0].ID != id {
		t.Fatalf("JSON result = %+v", result)
	}
}

func TestRunRejectsUnboundedAndInvalidFilters(t *testing.T) {
	if err := runQuery(nil, &bytes.Buffer{}); err == nil {
		t.Fatal("missing repository accepted")
	}
	if err := runQuery([]string{"-repo", "x", "-limit", "0"}, &bytes.Buffer{}); err == nil {
		t.Fatal("zero result bound accepted")
	}
	if _, err := parseFollow("sideways"); err == nil {
		t.Fatal("invalid follow direction accepted")
	}
}
