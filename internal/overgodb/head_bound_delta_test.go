package overgodb

import (
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/testutil"
)

// TestHeadBoundDeltaReconciliation pins the delta contract a read-only view
// reconciles against: a stateless consumer is sent to a snapshot, ordered
// windows walk the exact commit chain naming previous and resulting heads,
// alias churn coalesces to each name's final state while additions stay
// append-only, truncation bounds one call without losing continuity, and a
// diverged head or foreign projection contract forces resync instead of a
// delta with a hole in it.
func TestHeadBoundDeltaReconciliation(t *testing.T) {
	store, err := Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	ctx := t.Context()
	contract := ProjectionContractVersion()

	first := artifact.Descriptor{ID: testutil.ArtifactID(t, artifact.KindEvidence, "first"), Size: 1}
	second := artifact.Descriptor{ID: testutil.ArtifactID(t, artifact.KindEvidence, "second"), Size: 1}
	third := artifact.Descriptor{ID: testutil.ArtifactID(t, artifact.KindEvidence, "third"), Size: 1}
	if _, err := store.Commit(ctx, artifact.Batch{
		Key: "delta/one", Artifacts: []artifact.Descriptor{first},
		Aliases: []artifact.AliasBinding{{Name: "view/current", Target: first.ID}},
	}); err != nil {
		t.Fatal(err)
	}
	anchorHead, anchorSequence := store.Head()

	if _, resync, err := store.DeltasSince(ctx, artifact.CommitID{}, 0, contract, 8); err != nil || !resync {
		t.Fatalf("stateless consumer = (resync=%t, %v), want snapshot resync", resync, err)
	}
	if _, resync, err := store.DeltasSince(ctx, anchorHead, anchorSequence, "foreign", 8); err != nil || !resync {
		t.Fatalf("foreign projection contract = (resync=%t, %v), want resync", resync, err)
	}
	current, resync, err := store.DeltasSince(ctx, anchorHead, anchorSequence, contract, 8)
	if err != nil || resync || current.Head != anchorHead || len(current.Commits) != 0 || current.Truncated {
		t.Fatalf("up-to-date consumer = (%+v, resync=%t, %v)", current, resync, err)
	}

	previousTarget := first.ID
	if _, err := store.Commit(ctx, artifact.Batch{
		Key: "delta/two", Artifacts: []artifact.Descriptor{second},
		Aliases: []artifact.AliasBinding{{Name: "view/current", Target: second.ID, Previous: &previousTarget}},
		Lineage: []artifact.Lineage{{Child: second.ID, Parent: first.ID, Relation: artifact.RelationDerivedFrom}},
	}); err != nil {
		t.Fatal(err)
	}
	secondTarget := second.ID
	if _, err := store.Commit(ctx, artifact.Batch{
		Key: "delta/three", Artifacts: []artifact.Descriptor{third},
		Aliases: []artifact.AliasBinding{{Name: "view/current", Target: third.ID, Previous: &secondTarget}},
	}); err != nil {
		t.Fatal(err)
	}
	head, sequence := store.Head()

	delta, resync, err := store.DeltasSince(ctx, anchorHead, anchorSequence, contract, 8)
	if err != nil || resync {
		t.Fatalf("delta window = (resync=%t, %v)", resync, err)
	}
	if delta.PreviousHead != anchorHead || delta.PreviousSequence != anchorSequence ||
		delta.Head != head || delta.Sequence != sequence || delta.Truncated {
		t.Fatalf("delta heads = %+v, want %s@%d -> %s@%d", delta, anchorHead, anchorSequence, head, sequence)
	}
	if len(delta.Artifacts) != 2 || len(delta.Commits) != 2 || len(delta.Lineage) != 1 {
		t.Fatalf("delta additions = %+v", delta)
	}
	if len(delta.Aliases) != 1 || delta.Aliases[0].Name != "view/current" || delta.Aliases[0].Target != third.ID {
		t.Fatalf("alias churn did not coalesce to the final binding: %+v", delta.Aliases)
	}

	bounded, resync, err := store.DeltasSince(ctx, anchorHead, anchorSequence, contract, 1)
	if err != nil || resync || !bounded.Truncated || len(bounded.Commits) != 1 ||
		bounded.Aliases[0].Target != second.ID {
		t.Fatalf("bounded window = (%+v, resync=%t, %v)", bounded, resync, err)
	}
	resumed, resync, err := store.DeltasSince(ctx, bounded.Head, bounded.Sequence, contract, 8)
	if err != nil || resync || resumed.Truncated || len(resumed.Commits) != 1 ||
		resumed.Head != head || resumed.Aliases[0].Target != third.ID {
		t.Fatalf("resumed window = (%+v, resync=%t, %v)", resumed, resync, err)
	}

	diverged := artifact.CommitID{0xde, 0xad}
	if _, resync, err := store.DeltasSince(ctx, diverged, anchorSequence, contract, 8); err != nil || !resync {
		t.Fatalf("diverged consumer = (resync=%t, %v), want resync", resync, err)
	}
	if _, resync, err := store.DeltasSince(ctx, head, sequence+3, contract, 8); err != nil || !resync {
		t.Fatalf("consumer ahead of authority = (resync=%t, %v), want resync", resync, err)
	}
}
