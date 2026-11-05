package overgodb

import (
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/testcheck"
)

// TestCommitCoordinatorAtomicity holds the one transaction path to its
// guarantees: a failure before the durable append publishes nothing to
// any facet, and a successful append advances every required facet
// exactly once, with an exact key repeat replaying the same commit
// identity instead of moving state. An uncertainty during the append,
// which faults the store until reopen and leaves every facet as it was,
// is TestAppendFaultClosesEveryStoreSurfaceUntilReopen.
func TestCommitCoordinatorAtomicity(t *testing.T) {
	t.Parallel()
	ctx := t.Context()
	store, err := Open(t.TempDir())
	testcheck.NoError(t, err)
	defer store.Close()
	base := fixtureBatch(t)
	baseID, err := store.Commit(ctx, base)
	testcheck.NoError(t, err)
	digestBefore := catalogDigest(t, store.state)
	headBefore, sequenceBefore := store.Head()

	invalidTarget := fixtureDescriptor(t, artifact.KindOutput, "coordinator-unknown-target")
	if _, err := store.Commit(ctx, artifact.Batch{
		Key:     "coordinator/invalid-alias",
		Aliases: []artifact.AliasBinding{{Name: "coordinator/missing", Target: invalidTarget.ID}},
	}); err == nil {
		t.Fatal("alias to unknown artifact admitted")
	}
	if digest := catalogDigest(t, store.state); digest != digestBefore {
		t.Fatal("failed validation mutated facet state")
	}
	if head, sequence := store.Head(); head != headBefore || sequence != sequenceBefore {
		t.Fatal("failed validation moved the head")
	}

	repeatID, err := store.Commit(ctx, base)
	if err != nil || repeatID != baseID {
		t.Fatalf("idempotent repeat = (%s, %v), want %s", repeatID, err, baseID)
	}
	if digest := catalogDigest(t, store.state); digest != digestBefore {
		t.Fatal("idempotent repeat mutated facet state")
	}

	descriptor := fixtureDescriptor(t, artifact.KindOutput, "coordinator-advance")
	content := artifact.Content{Descriptor: descriptor, Data: []byte("coordinator-advance")}
	advanceID, err := store.Commit(ctx, artifact.Batch{
		Key:      "coordinator/advance",
		Contents: []artifact.Content{content},
		Aliases:  []artifact.AliasBinding{{Name: "coordinator/alias", Target: descriptor.ID}},
		Lineage:  []artifact.Lineage{{Child: descriptor.ID, Parent: base.Artifacts[0].ID, Relation: artifact.RelationDerivedFrom}},
	})
	testcheck.NoError(t, err)
	if head, sequence := store.Head(); head != advanceID || sequence != sequenceBefore+1 {
		t.Fatalf("advance head = (%s, %d)", head, sequence)
	}
	if !store.state.artifacts.has(descriptor.ID) || !store.state.contents.has(descriptor.ID) ||
		store.state.lineage.count() != 1+countLineage(base) {
		t.Fatal("successful append did not advance every facet exactly once")
	}
	if target, ok := store.state.aliases.resolve("coordinator/alias"); !ok || target != descriptor.ID {
		t.Fatal("alias facet did not advance")
	}
}

func countLineage(batch artifact.Batch) int { return len(batch.Lineage) }
