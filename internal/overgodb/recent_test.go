package overgodb

import (
	"strings"
	"testing"

	"overgo/internal/artifact"
)

// TestRecentArtifactsWalkNewestFirst pins the recency read: files come
// newest first with their introducing sequence and payload presence, a
// keep predicate filters them, the limit bounds them, and a kind with
// none answers empty.
func TestRecentArtifactsWalkNewestFirst(t *testing.T) {
	store, err := Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	commit := func(key, mediaType string, data []byte) artifact.ID {
		t.Helper()
		id, err := artifact.IdentifyBytes(artifact.KindFile, data)
		if err != nil {
			t.Fatal(err)
		}
		if _, err := artifact.CommitBatch(t.Context(), store, artifact.Batch{
			Key:      key,
			Contents: []artifact.Content{{Descriptor: artifact.Descriptor{ID: id, Size: uint64(len(data)), MediaType: mediaType}, Data: data}},
		}); err != nil {
			t.Fatal(err)
		}
		return id
	}
	older := commit("recent/1", "image/png", []byte("older image"))
	clip := commit("recent/2", "audio/wav", []byte("a clip"))
	newer := commit("recent/3", "image/jpeg", []byte("newer image"))
	recent, err := store.RecentArtifacts(t.Context(), artifact.KindFile, 5, nil)
	if err != nil {
		t.Fatal(err)
	}
	if len(recent) != 3 || recent[0].Descriptor.ID != newer || recent[1].Descriptor.ID != clip || recent[2].Descriptor.ID != older ||
		recent[0].Sequence <= recent[2].Sequence || !recent[0].Payload || len(recent[0].Producers) != 0 {
		t.Fatalf("recent = %+v", recent)
	}
	// A produced artifact lists the runs that produced it (a gallery names the record).
	producer, err := artifact.IdentifyBytes(artifact.KindRun, []byte("the producing run"))
	if err != nil {
		t.Fatal(err)
	}
	if _, err := artifact.CommitBatch(t.Context(), store, artifact.Batch{
		Key:       "recent/4",
		Artifacts: []artifact.Descriptor{{ID: producer, Size: 1}},
		Lineage:   []artifact.Lineage{{Child: newer, Parent: producer, Relation: artifact.RelationProducedBy}},
	}); err != nil {
		t.Fatal(err)
	}
	produced, err := store.RecentArtifacts(t.Context(), artifact.KindFile, 1, nil)
	if err != nil || len(produced) != 1 || len(produced[0].Producers) != 1 || produced[0].Producers[0] != producer {
		t.Fatalf("produced = %+v, %v", produced, err)
	}
	images, err := store.RecentArtifacts(t.Context(), artifact.KindFile, 1, func(descriptor artifact.Descriptor) bool {
		return strings.HasPrefix(descriptor.MediaType, "image/")
	})
	if err != nil || len(images) != 1 || images[0].Descriptor.ID != newer {
		t.Fatalf("newest image = %+v, %v", images, err)
	}
	if none, err := store.RecentArtifacts(t.Context(), artifact.KindModel, 5, nil); err != nil || len(none) != 0 {
		t.Fatalf("a kind with no artifact = %+v, %v", none, err)
	}
	if _, err := store.RecentArtifacts(t.Context(), artifact.KindFile, 0, nil); err == nil {
		t.Fatal("a zero limit was accepted")
	}
}
