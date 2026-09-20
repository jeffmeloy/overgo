package overgodb

import (
	"errors"
	"path/filepath"
	"testing"

	"overgo/internal/artifact"
)

// TestWriteAdmissionBindsKindsToProducers holds the commit door to the
// producer table: a guarded kind is refused to a caller holding only the
// API, to a batch that names the producer itself and to another producer;
// its own producer commits it and the frame records whose it was across a
// reopen; a repeat of the committed batch replays through either door;
// naming an artifact the store already holds mints nothing, so an
// unproduced batch may depend on it; and Rebuild transplants the kind into a
// new chain.
func TestWriteAdmissionBindsKindsToProducers(t *testing.T) {
	ctx := t.Context()
	root := filepath.Join(t.TempDir(), "store")
	store, err := Open(root)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = store.Close() })
	kind := producerKinds[0]
	guarded := retentionContent(t, artifact.KindEvidence, map[string]any{"claim": "guarded"})
	guarded.Descriptor.Schema = kind.prefix + "v-next"
	batch := artifact.Batch{Key: "admission/guarded", Contents: []artifact.Content{guarded}, Producer: kind.producer}

	refused := map[string]func() (artifact.CommitID, error){
		"the API alone":    func() (artifact.CommitID, error) { return store.Commit(ctx, batch) },
		"another producer": func() (artifact.CommitID, error) { return store.CommitAs(ctx, NewProducer("another"), batch) },
	}
	for holder, commit := range refused {
		if _, err := commit(); !errors.Is(err, ErrProducerRefused) {
			t.Fatalf("%s committed a guarded kind: %v", holder, err)
		}
	}
	if _, sequence := store.Head(); sequence != 0 {
		t.Fatalf("a refused batch advanced the head to %d", sequence)
	}
	committed, err := store.CommitAs(ctx, NewProducer(kind.producer), batch)
	if err != nil {
		t.Fatal(err)
	}
	// Who asked is no part of what was asked: the repeat replays and mints nothing.
	if repeat, err := store.Commit(ctx, batch); err != nil || repeat != committed {
		t.Fatalf("a repeat through the plain door = %s, %v; want the committed %s", repeat, err, committed)
	}
	if _, sequence := store.Head(); sequence != 1 {
		t.Fatalf("a replayed batch moved the head to %d", sequence)
	}

	dependent := retentionContent(t, artifact.KindOutput, map[string]any{"reads": "guarded"})
	if _, err := store.Commit(ctx, artifact.Batch{
		Key: "admission/dependent", Producer: kind.producer,
		Artifacts: []artifact.Descriptor{guarded.Descriptor}, Contents: []artifact.Content{dependent},
		Lineage: []artifact.Lineage{{Child: dependent.Descriptor.ID, Parent: guarded.Descriptor.ID, Relation: artifact.RelationDependsOn}},
	}); err != nil {
		t.Fatalf("an unproduced batch that only names a held guarded artifact was refused: %v", err)
	}
	if err := store.Close(); err != nil {
		t.Fatal(err)
	}

	store, err = OpenReadOnly(root)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = store.Close() })
	for sequence, producer := range map[uint64]string{1: kind.producer, 2: ""} {
		frame, found, err := store.CommitDeltaAt(ctx, sequence)
		if err != nil || !found || frame.Delta.Producer != producer {
			t.Fatalf("frame %d records producer %q, want %q (found=%v err=%v)", sequence, frame.Delta.Producer, producer, found, err)
		}
	}

	destination := filepath.Join(t.TempDir(), "rebuilt")
	if _, err := Rebuild(ctx, store, destination, nil, func(string, artifact.ID) error { return nil }); err != nil {
		t.Fatalf("rebuild could not transplant a guarded kind: %v", err)
	}
	rebuilt, err := OpenReadOnly(destination)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = rebuilt.Close() })
	if held, err := rebuilt.HasContent(ctx, guarded.Descriptor.ID); err != nil || !held {
		t.Fatalf("rebuilt store holds the guarded content = %v, %v", held, err)
	}
}
