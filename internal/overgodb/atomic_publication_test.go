package overgodb

import (
	"os"
	"testing"

	"overgo/internal/artifact"
)

// TestProjectionPublicationIsAtomic holds publication to its contract:
// a completed commit exposes one head-consistent view -- the committed
// artifact, its alias, and its lineage together. A commit any projection refuses to accept becomes
// durable nowhere: the journal, the head, and every facet stay at the
// last complete view.
func TestProjectionPublicationIsAtomic(t *testing.T) {
	ctx := t.Context()
	store, err := Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	base := fixtureBatch(t)
	if _, err := store.Commit(ctx, base); err != nil {
		t.Fatal(err)
	}

	payload := []byte("atomic-publication payload")
	id, err := artifact.IdentifyBytes(artifact.KindRun, payload)
	if err != nil {
		t.Fatal(err)
	}
	descriptor := artifact.Descriptor{ID: id, Size: uint64(len(payload)), MediaType: "text/plain"}
	if _, err := store.Commit(ctx, artifact.Batch{
		Key:      "atomic/commit",
		Contents: []artifact.Content{{Descriptor: descriptor, Data: payload}},
		Aliases:  []artifact.AliasBinding{{Name: "atomic/alias", Target: id}},
		Lineage:  []artifact.Lineage{{Child: id, Parent: base.Artifacts[0].ID, Relation: artifact.RelationDerivedFrom}},
	}); err != nil {
		t.Fatal(err)
	}
	if _, found, err := store.Artifact(ctx, id); err != nil || !found {
		t.Fatalf("published view lacks the committed artifact: found=%v err=%v", found, err)
	}
	if target, found, err := store.ResolveAlias(ctx, "atomic/alias"); err != nil || !found || target != id {
		t.Fatalf("published view lacks the committed alias: found=%v err=%v", found, err)
	}
	if parents, err := store.Parents(ctx, id); err != nil || len(parents) != 1 {
		t.Fatalf("published view lacks the committed lineage: %v %v", parents, err)
	}

	// An idempotent replay publishes nothing new.
	_, sequenceBeforeReplay := store.Head()
	if _, err := store.Commit(ctx, base); err != nil {
		t.Fatal(err)
	}
	if _, sequence := store.Head(); sequence != sequenceBeforeReplay {
		t.Fatalf("replayed commit advanced from %d to %d", sequenceBeforeReplay, sequence)
	}

	// A commit a projection refuses becomes durable nowhere: drive the
	// coordinator with a content that has no bound locator.
	journalBefore, err := os.Stat(store.log.file.Name())
	if err != nil {
		t.Fatal(err)
	}
	digestBefore := catalogDigest(t, store.state)
	refusedPayload := []byte("refused content")
	refusedID, err := artifact.IdentifyBytes(artifact.KindRun, refusedPayload)
	if err != nil {
		t.Fatal(err)
	}
	refused := artifact.Batch{Key: "atomic/refused", Artifacts: []artifact.Descriptor{{
		ID: refusedID, Size: uint64(len(refusedPayload)), MediaType: "text/plain",
	}}}
	refused.Contents = []artifact.Content{{Descriptor: refused.Artifacts[0], Data: refusedPayload}}
	normalized, payloadHash, err := encodeBatch(refused)
	if err != nil {
		t.Fatal(err)
	}
	delta := store.state.delta(normalized)
	if err := store.state.acceptAll(delta, map[artifact.ID]contentLocator{}, store.sequence+1); err == nil {
		t.Fatal("projection accepted a content with no durable locator")
	}
	_ = payloadHash
	if journalAfter, err := os.Stat(store.log.file.Name()); err != nil || journalAfter.Size() != journalBefore.Size() {
		t.Fatalf("refused commit touched the journal: %v", err)
	}
	if digest := catalogDigest(t, store.state); digest != digestBefore {
		t.Fatal("refused commit mutated facet state")
	}
}
