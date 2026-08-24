package dataset

import (
	"bytes"
	"context"
	"encoding/binary"
	"hash/crc32"
	"os"
	"path/filepath"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/overgodb"
)

const (
	fixtureDirectoryName = "records"
	fixtureFileName      = "examples.jsonl"
	fixtureDatasetName   = "examples"
	fixtureFileBody      = "{}\n"
)

func TestDatasetCatalogPublishReplayAndLocations(t *testing.T) {
	legacyRoot, contentRoot := legacyCatalogFixture(t)
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	stale, err := artifact.IdentifyBytes(artifact.KindDataset, []byte("stale"))
	if err != nil {
		t.Fatal(err)
	}
	if _, err := store.Commit(context.Background(), artifact.Batch{
		Key: "fixture/stale-dataset-alias", Artifacts: []artifact.Descriptor{{ID: stale}},
		Aliases: []artifact.AliasBinding{{Name: registeredAlias(fixtureDatasetName), Target: stale}},
	}); err != nil {
		t.Fatal(err)
	}
	publication, err := PublishLegacyCatalog(context.Background(), store, legacyRoot, contentRoot)
	if err != nil {
		t.Fatal(err)
	}
	if !publication.Changed || !publication.Coverage.Complete || publication.Coverage.Registered != len([]string{fixtureDatasetName, fixtureDirectoryName}) {
		t.Fatalf("publication = %+v", publication)
	}
	replay, err := PublishLegacyCatalog(context.Background(), store, legacyRoot, contentRoot)
	if err != nil {
		t.Fatal(err)
	}
	if replay.Changed || replay.Commit != publication.Commit {
		t.Fatalf("replay = %+v", replay)
	}
	if err := store.Close(); err != nil {
		t.Fatal(err)
	}
}

func TestDatasetCatalogIncludesRootOnlyContentWithoutSemanticInference(t *testing.T) {
	legacyRoot, contentRoot := legacyCatalogFixture(t)
	if err := os.WriteFile(filepath.Join(contentRoot, "uncataloged.txt"), []byte("x"), 0o600); err != nil {
		t.Fatal(err)
	}
	bundle, err := CompileLegacyCatalog(legacyRoot, contentRoot)
	if err != nil {
		t.Fatal(err)
	}
	entry := bundle.Catalog.Catalog[len(bundle.Catalog.Catalog)-1]
	if entry.Name != "uncataloged.txt" || entry.Source != "content-root" || entry.Modality != legacyUnknownFact ||
		len(entry.Formats) != 1 || entry.Formats[0] != legacyUnknownFact {
		t.Fatalf("root-only entry = %+v", entry)
	}
}

func legacyCatalogFixture(t *testing.T) (string, string) {
	t.Helper()
	legacyRoot, contentRoot := t.TempDir(), t.TempDir()
	if err := os.Mkdir(filepath.Join(contentRoot, fixtureDirectoryName), 0o700); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(contentRoot, fixtureDirectoryName, "part.txt"), []byte("part"), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(contentRoot, fixtureFileName), []byte(fixtureFileBody), 0o600); err != nil {
		t.Fatal(err)
	}
	var log bytes.Buffer
	writeLegacyRecord(t, &log, legacyDatasetRecord, encodeLegacyDatasetFixture(legacyDataset{
		ID: 1, Name: fixtureDirectoryName, Kind: "directory", Source: "loose", Modality: "text", FileCount: 1, ByteCount: 4,
	}))
	writeLegacyRecord(t, &log, legacyDatasetFileRecord, encodeLegacyFileFixture(legacyDatasetFile{
		ID: 1, DatasetID: 1, Path: "part.txt", OriginalName: "part.txt", Extension: ".txt",
		Modality: "text", Format: "txt", Bytes: 4,
	}))
	writeLegacyRecord(t, &log, legacyDatasetRecord, encodeLegacyDatasetFixture(legacyDataset{
		ID: 2, Name: fixtureDatasetName, Kind: "file", Source: "loose", Modality: "structured", FileCount: 1, ByteCount: int64(len(fixtureFileBody)),
	}))
	writeLegacyRecord(t, &log, legacyDatasetFileRecord, encodeLegacyFileFixture(legacyDatasetFile{
		ID: 2, DatasetID: 2, Path: fixtureFileName, OriginalName: fixtureFileName, Extension: ".jsonl",
		Modality: "structured", Format: "jsonl", Bytes: int64(len(fixtureFileBody)), Structured: true,
	}))
	if err := os.WriteFile(filepath.Join(legacyRoot, legacyRegistryName), log.Bytes(), 0o600); err != nil {
		t.Fatal(err)
	}
	return legacyRoot, contentRoot
}

func writeLegacyRecord(t *testing.T, output *bytes.Buffer, kind byte, payload []byte) {
	t.Helper()
	header := []byte{kind, 0, 0, 0, 0}
	binary.LittleEndian.PutUint32(header[1:], uint32(len(payload)))
	output.Write(header)
	output.Write(payload)
	checksum := crc32.Update(crc32.Update(0, legacyChecksumTable, header), legacyChecksumTable, payload)
	if err := binary.Write(output, binary.LittleEndian, checksum); err != nil {
		t.Fatal(err)
	}
}

type legacyFixtureEncoder struct{ bytes.Buffer }

func (encoder *legacyFixtureEncoder) uint32(value uint32) {
	_ = binary.Write(&encoder.Buffer, binary.LittleEndian, value)
}
func (encoder *legacyFixtureEncoder) int64(value int64) {
	_ = binary.Write(&encoder.Buffer, binary.LittleEndian, value)
}
func (encoder *legacyFixtureEncoder) text(value string) {
	encoder.uint32(uint32(len(value)))
	encoder.WriteString(value)
}

func encodeLegacyDatasetFixture(value legacyDataset) []byte {
	encoder := &legacyFixtureEncoder{}
	encoder.uint32(value.ID)
	for _, field := range []string{value.Name, value.Kind, value.Source, value.Modality} {
		encoder.text(field)
	}
	encoder.int64(value.FileCount)
	encoder.int64(value.ByteCount)
	return encoder.Bytes()
}

func encodeLegacyFileFixture(value legacyDatasetFile) []byte {
	encoder := &legacyFixtureEncoder{}
	encoder.uint32(value.ID)
	encoder.uint32(value.DatasetID)
	for _, field := range []string{value.Path, value.OriginalName, value.Extension, value.Modality, value.Format} {
		encoder.text(field)
	}
	encoder.int64(value.Bytes)
	encoder.int64(value.ModifiedUnix)
	if value.Structured {
		encoder.WriteByte(1)
	} else {
		encoder.WriteByte(0)
	}
	encoder.text(value.Attributes)
	return encoder.Bytes()
}

func testCompiledCatalog(t *testing.T, path string) CompiledCatalog {
	t.Helper()
	inventory, err := NewInventory([]InventoryFile{{
		Path: filepath.Base(path), OriginalName: filepath.Base(path), Modality: "text", Format: "txt", Bytes: 1,
	}})
	if err != nil {
		t.Fatal(err)
	}
	version, err := NewVersion([]Asset{{Name: "inventory", Artifact: inventory.ID, Records: 1}})
	if err != nil {
		t.Fatal(err)
	}
	catalog, err := NewCatalog([]CatalogEntry{{
		Name: "fixture", Dataset: version.ID, Inventory: inventory.ID, StorageKind: "file",
		Source: "fixture", Modality: "text", Formats: []string{"txt"}, Files: 1, Bytes: 1,
	}})
	if err != nil {
		t.Fatal(err)
	}
	location, err := artifact.CanonicalLocalLocation(version.ID, artifact.LocationFile, path)
	if err != nil {
		t.Fatal(err)
	}
	return CompiledCatalog{Catalog: catalog, Datasets: []Document{version}, Inventories: []Inventory{inventory}, Locations: []artifact.Location{location}}
}

// TestDatasetCatalogPublishRecordsLateArrivals closes the
// catalog-first-download-later finding: a dataset absent at first
// publication gains its location fact on republication after its
// bytes appear, and the converged catalog is then idempotent again.
func TestDatasetCatalogPublishRecordsLateArrivals(t *testing.T) {
	ctx := context.Background()
	legacyRoot, contentRoot := legacyCatalogFixture(t)
	moved := filepath.Join(t.TempDir(), fixtureFileName)
	if err := os.Rename(filepath.Join(contentRoot, fixtureFileName), moved); err != nil {
		t.Fatal(err)
	}
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	first, err := PublishLegacyCatalog(ctx, store, legacyRoot, contentRoot)
	if err != nil {
		t.Fatal(err)
	}
	if !first.Changed || !first.Coverage.Complete || first.Coverage.Available >= first.Coverage.Registered {
		t.Fatalf("first publication = %+v, want complete aliases with the moved dataset unavailable", first)
	}
	if err := os.Rename(moved, filepath.Join(contentRoot, fixtureFileName)); err != nil {
		t.Fatal(err)
	}
	second, err := PublishLegacyCatalog(ctx, store, legacyRoot, contentRoot)
	if err != nil {
		t.Fatalf("republication after arrival failed: %v", err)
	}
	if !second.Changed || second.Coverage.Available != first.Coverage.Available+1 {
		t.Fatalf("second publication = %+v, want the late arrival recorded", second)
	}
	third, err := PublishLegacyCatalog(ctx, store, legacyRoot, contentRoot)
	if err != nil {
		t.Fatal(err)
	}
	if third.Changed {
		t.Fatalf("converged catalog changed again: %+v", third)
	}
}
