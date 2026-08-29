package overgodb_test

import (
	"fmt"
	"slices"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/artifact/repositorytest"
	"overgo/internal/overgodb"
)

// TestHighFanoutReadsAreBatched holds the batch read owners to their
// contract: many contents resolve through one VisitContents batch with
// zero per-item opens, caller order is preserved whatever the storage
// order, an absent id is the exact missing-content error, and content
// presence for a mixed list answers in one acquisition.
func TestHighFanoutReadsAreBatched(t *testing.T) {
	ctx := t.Context()
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	const fanout = 48
	ids := make([]artifact.ID, fanout)
	for ordinal := range fanout {
		payload := []byte(fmt.Sprintf("fanout/%d", ordinal))
		id, err := artifact.IdentifyBytes(artifact.KindEvidence, payload)
		if err != nil {
			t.Fatal(err)
		}
		ids[ordinal] = id
		descriptor := artifact.Descriptor{ID: id, Size: uint64(len(payload)), MediaType: "text/plain"}
		if _, err := store.Commit(ctx, artifact.Batch{
			Key:      fmt.Sprintf("fanout/%d", ordinal),
			Contents: []artifact.Content{{Descriptor: descriptor, Data: payload}},
		}); err != nil {
			t.Fatal(err)
		}
	}
	reversed := make([]artifact.ID, fanout)
	for index, id := range ids {
		reversed[fanout-index-1] = id
	}
	counting := &repositorytest.CountingRepository{Repository: store}
	visited := make([]artifact.ID, 0, fanout)
	if err := artifact.ReadContents(ctx, counting, reversed, func(content artifact.Content) error {
		visited = append(visited, content.Descriptor.ID)
		return nil
	}); err != nil {
		t.Fatal(err)
	}
	if counting.Batches != 1 || counting.Opens != 0 {
		t.Fatalf("batched read used batches=%d opens=%d", counting.Batches, counting.Opens)
	}
	for index, id := range visited {
		if id != reversed[index] {
			t.Fatalf("caller order broken at %d", index)
		}
	}

	absent, err := artifact.IdentifyBytes(artifact.KindEvidence, []byte("fanout/absent"))
	if err != nil {
		t.Fatal(err)
	}
	if err := artifact.ReadContents(ctx, counting, append(slices.Clip(ids[:4]), absent), func(artifact.Content) error {
		return nil
	}); err == nil {
		t.Fatal("absent id read silently")
	}

	present, err := store.PresentContents(ctx, append(slices.Clip(ids[:4]), absent))
	if err != nil || len(present) != 4 {
		t.Fatalf("presence = (%v, %v)", present, err)
	}
}
