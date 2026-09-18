package dataset

import (
	"testing"

	"overgo/internal/artifact"
)

// TestIsContentArtifact proves the shared predicate accepts exactly the dataset
// content kinds — a dataset, a shard, or a raw file — and nothing else, so
// dataset transforms and records over dataset content share one rule.
func TestIsContentArtifact(t *testing.T) {
	content := []artifact.Kind{artifact.KindDataset, artifact.KindDatasetShard, artifact.KindFile}
	for _, kind := range content {
		id, err := artifact.IdentifyBytes(kind, []byte("content"))
		if err != nil {
			t.Fatal(err)
		}
		if !IsContentArtifact(id) {
			t.Fatalf("content kind %s rejected", kind)
		}
	}
	for _, kind := range []artifact.Kind{artifact.KindModel, artifact.KindRecipe, artifact.KindEvidence, artifact.KindProfile} {
		id, err := artifact.IdentifyBytes(kind, []byte("other"))
		if err != nil {
			t.Fatal(err)
		}
		if IsContentArtifact(id) {
			t.Fatalf("non-content kind %s accepted", kind)
		}
	}
	if IsContentArtifact(artifact.ID{}) {
		t.Fatal("the zero identity is not content")
	}
}
