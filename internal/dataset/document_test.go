package dataset

import (
	"context"
	"encoding/json"
	"reflect"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/repodb"
	"overgo/internal/testutil"
)

func TestDatasetDocumentsRoundTripAndPublish(t *testing.T) {
	assetA := testutil.ArtifactID(t, artifact.KindDatasetShard, "asset-a")
	assetB := testutil.ArtifactID(t, artifact.KindFile, "asset-b")
	version, err := NewVersion([]Asset{
		{Name: "second", Artifact: assetB, Records: 3},
		{Name: "first", Artifact: assetA, Records: 2},
	})
	if err != nil {
		t.Fatal(err)
	}
	selector := testutil.ArtifactID(t, artifact.KindDatasetShard, "selector")
	viewA, err := NewView(version.ID, &selector, []string{"target", "prompt"})
	if err != nil {
		t.Fatal(err)
	}
	viewB, err := NewView(version.ID, nil, []string{"prompt"})
	if err != nil {
		t.Fatal(err)
	}
	split, err := NewSplit(version.ID, []Partition{
		{Name: "train", View: viewA.ID}, {Name: "test", View: viewB.ID},
	})
	if err != nil {
		t.Fatal(err)
	}
	mixture, err := NewMixture([]Member{
		{Dataset: split.ID, Weight: 6}, {Dataset: version.ID, Weight: 2},
	})
	if err != nil {
		t.Fatal(err)
	}
	weights := map[artifact.ID]uint64{}
	for _, member := range mixture.Members {
		weights[member.Dataset] = member.Weight
	}
	if weights[version.ID] != 1 || weights[split.ID] != 3 {
		t.Fatalf("normalized mixture = %+v", mixture.Members)
	}

	documents := []Document{version, viewA, viewB, split, mixture}
	for _, document := range documents {
		content, contentErr := document.Content()
		if contentErr != nil {
			t.Fatal(contentErr)
		}
		parsed, parseErr := Parse(content.Data)
		if parseErr != nil {
			t.Fatal(parseErr)
		}
		if !reflect.DeepEqual(parsed, document) {
			t.Fatalf("parsed document = %+v, want %+v", parsed, document)
		}
	}
	store, err := repodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	descriptors := []artifact.Descriptor{
		{ID: assetA, Size: 7}, {ID: assetB, Size: 7}, {ID: selector, Size: 8},
	}
	batch, err := PublicationBatch("dataset-fixture", documents, []artifact.AliasBinding{{
		Name: "datasets/training", Target: mixture.ID,
	}})
	if err != nil {
		t.Fatal(err)
	}
	batch.Artifacts = descriptors
	if _, err := store.Commit(context.Background(), batch); err != nil {
		t.Fatal(err)
	}
	parents, err := store.Parents(context.Background(), mixture.ID)
	if err != nil {
		t.Fatal(err)
	}
	if len(parents) != len(mixture.Members) {
		t.Fatalf("mixture parents = %+v", parents)
	}
	resolved, ok, err := Resolve(context.Background(), store, "datasets/training")
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
	version, err := NewVersion([]Asset{{
		Name: "asset", Artifact: testutil.ArtifactID(t, artifact.KindFile, "asset"), Records: 1,
	}})
	if err != nil {
		t.Fatal(err)
	}
	content, err := version.ContentBytes()
	if err != nil {
		t.Fatal(err)
	}
	var body map[string]any
	if err := json.Unmarshal(content, &body); err != nil {
		t.Fatal(err)
	}
	body["unknown"] = true
	modified, err := json.Marshal(body)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := Parse(modified); err == nil {
		t.Fatal("unknown dataset field accepted")
	}
}
