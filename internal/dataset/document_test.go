package dataset

import (
	"encoding/json"
	"reflect"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/overgodbtest"
	"overgo/internal/testcheck"
	"overgo/internal/testutil"
)

func TestDatasetDocumentsRoundTripAndPublish(t *testing.T) {
	assetA := testutil.ArtifactID(t, artifact.KindDatasetShard, "asset-a")
	assetB := testutil.ArtifactID(t, artifact.KindFile, "asset-b")
	version := testcheck.Must(NewVersion([]Asset{
		{Name: "second", Artifact: assetB, Records: 3},
		{Name: "first", Artifact: assetA, Records: 2},
	}))(t)
	selector := testutil.ArtifactID(t, artifact.KindDatasetShard, "selector")
	viewA := testcheck.Must(NewView(version.ID, &selector, []string{"target", "prompt"}))(t)
	viewB := testcheck.Must(NewView(version.ID, nil, []string{"prompt"}))(t)
	split := testcheck.Must(NewSplit(version.ID, []Partition{
		{Name: "train", View: viewA.ID}, {Name: "test", View: viewB.ID},
	}))(t)
	mixture := testcheck.Must(NewMixture([]Member{
		{Dataset: split.ID, Weight: 6}, {Dataset: version.ID, Weight: 2},
	}))(t)
	weights := map[artifact.ID]uint64{}
	for _, member := range mixture.Members {
		weights[member.Dataset] = member.Weight
	}
	if weights[version.ID] != 1 || weights[split.ID] != 3 {
		t.Fatalf("normalized mixture = %+v", mixture.Members)
	}

	entries := []CatalogEntry{
		{Name: "second", Dataset: split.ID, Inventory: assetA, StorageKind: "file", Source: "fixture", Modality: "mixed", Formats: []string{"txt", "json", "txt"}, Modalities: []string{"text", "audio", "text"}},
		{Name: "first", Dataset: version.ID, Inventory: assetA, StorageKind: "file", Source: "fixture", Modality: "text"},
	}
	before := testcheck.Must(json.Marshal(entries))(t)
	catalog := testcheck.Must(NewCatalog(entries))(t)
	after, err := json.Marshal(entries)
	if err != nil || string(after) != string(before) {
		t.Fatalf("catalog mutated its input: %s, %v", after, err)
	}
	expected := []CatalogEntry{entries[1], entries[0]}
	expected[1].Formats = []string{"json", "txt"}
	expected[1].Modalities = []string{"audio", "text"}
	expectedJSON := testcheck.Must(json.Marshal(Document{Version: artifact.InitialDocumentVersion, Type: TypeCatalog, Catalog: expected}))(t)
	expectedID := testcheck.Must(artifact.IdentifyBytes(artifact.KindDataset, expectedJSON))(t)
	content, err := catalog.Content()
	if err != nil || string(content.Data) != string(expectedJSON) || catalog.ID != expectedID {
		t.Fatalf("catalog bytes or identity differ: %s, %v", content.Data, err)
	}
	for _, formats := range [][]string{nil, {}} {
		entry := entries[1]
		entry.Formats = formats
		value := testcheck.Must(NewCatalog([]CatalogEntry{entry}))(t)
		if (value.Catalog[0].Formats == nil) != (formats == nil) {
			t.Fatal("catalog changed nil/empty input state")
		}
	}
	duplicate := []CatalogEntry{entries[0], entries[0]}
	before = testcheck.Must(json.Marshal(duplicate))(t)
	if _, err := NewCatalog(duplicate); err == nil {
		t.Fatal("duplicate catalog entry accepted")
	}
	after, err = json.Marshal(duplicate)
	if err != nil || string(after) != string(before) {
		t.Fatalf("refused catalog mutated its input: %s, %v", after, err)
	}

	documents := []Document{version, viewA, viewB, split, mixture, catalog}
	for _, document := range documents {
		content, contentErr := document.Content()
		if contentErr != nil {
			t.Fatal(contentErr)
		}
		parsed, parseErr := documentCodec.Parse(content.Data)
		if parseErr != nil {
			t.Fatal(parseErr)
		}
		if !reflect.DeepEqual(parsed, document) {
			t.Fatalf("parsed document = %+v, want %+v", parsed, document)
		}
	}
	store := overgodbtest.Open(t, t.TempDir())
	descriptors := []artifact.Descriptor{
		{ID: assetA, Size: 7}, {ID: assetB, Size: 7}, {ID: selector, Size: 8},
	}
	batch := testcheck.Must(PublicationBatch("dataset-fixture", documents, []artifact.AliasBinding{{
		Name: "datasets/training", Target: mixture.ID,
	}}))(t)
	batch.Artifacts = descriptors
	if _, err := store.Commit(t.Context(), batch); err != nil {
		t.Fatal(err)
	}
	parents := testcheck.Must(store.Parents(t.Context(), mixture.ID))(t)
	if len(parents) != len(mixture.Members) {
		t.Fatalf("mixture parents = %+v", parents)
	}
	resolved, ok, err := Resolve(t.Context(), store, "datasets/training")
	if err != nil || !ok || !reflect.DeepEqual(resolved, mixture) {
		t.Fatalf("resolved mixture = %+v, %v, %v", resolved, ok, err)
	}
}

func TestDatasetDocumentsRejectInvalidAndNonCanonicalContent(t *testing.T) {
	source := testutil.ArtifactID(t, artifact.KindDataset, "source")
	if _, err := NewView(source, nil, nil); err == nil {
		t.Fatal("identity view accepted")
	}
	if _, err := NewMixture([]Member{{Dataset: source, Weight: 1}}); err == nil {
		t.Fatal("singleton mixture accepted")
	}
	version := testcheck.Must(NewVersion([]Asset{{
		Name: "asset", Artifact: testutil.ArtifactID(t, artifact.KindFile, "asset"), Records: 1,
	}}))(t)
	content := testcheck.Must(version.Content())(t)
	var body map[string]any
	testcheck.NoError(t, json.Unmarshal(content.Data, &body))
	body["unknown"] = true
	modified := testcheck.Must(json.Marshal(body))(t)
	if _, err := documentCodec.Parse(modified); err == nil {
		t.Fatal("unknown dataset field accepted")
	}
}
