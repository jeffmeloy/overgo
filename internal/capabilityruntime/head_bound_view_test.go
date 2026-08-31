package capabilityruntime

import (
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/overgodb"
	"overgo/internal/testutil"
)

// TestHeadBoundDeltaReconciliation pins the runtime-view contract the
// session director's authority freshness check rides: an unseeded view
// resyncs once and adopts the current coordinate, an unmoved store reports
// no change so authority re-resolution is skipped, committed movement hands
// over the coalesced delta and advances the view, and a diverged coordinate
// forces resync instead of a delta with a hole in it.
func TestHeadBoundDeltaReconciliation(t *testing.T) {
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	ctx := t.Context()
	first := artifact.Descriptor{ID: testutil.ArtifactID(t, artifact.KindEvidence, "view-first"), Size: 1}
	if _, err := store.Commit(ctx, artifact.Batch{
		Key: "view/base", Artifacts: []artifact.Descriptor{first},
		Aliases: []artifact.AliasBinding{{Name: "capability/active", Target: first.ID}},
	}); err != nil {
		t.Fatal(err)
	}

	view := &HeadBoundView{}
	changed, resync, _, err := view.Reconcile(ctx, store)
	if err != nil || !changed || !resync {
		t.Fatalf("unseeded view = (changed=%t, resync=%t, %v), want snapshot resync", changed, resync, err)
	}
	changed, resync, _, err = view.Reconcile(ctx, store)
	if err != nil || changed || resync {
		t.Fatalf("unmoved store = (changed=%t, resync=%t, %v), want skipped re-resolution", changed, resync, err)
	}

	second := artifact.Descriptor{ID: testutil.ArtifactID(t, artifact.KindEvidence, "view-second"), Size: 1}
	firstTarget := first.ID
	if _, err := store.Commit(ctx, artifact.Batch{
		Key: "view/move", Artifacts: []artifact.Descriptor{second},
		Aliases: []artifact.AliasBinding{{Name: "capability/active", Target: second.ID, Previous: &firstTarget}},
	}); err != nil {
		t.Fatal(err)
	}
	changed, resync, delta, err := view.Reconcile(ctx, store)
	if err != nil || !changed || resync {
		t.Fatalf("moved store = (changed=%t, resync=%t, %v), want delta", changed, resync, err)
	}
	if len(delta.Commits) != 1 || len(delta.Aliases) != 1 || delta.Aliases[0].Target != second.ID {
		t.Fatalf("authority delta = %+v", delta)
	}
	changed, resync, _, err = view.Reconcile(ctx, store)
	if err != nil || changed || resync {
		t.Fatalf("reconciled view = (changed=%t, resync=%t, %v), want no change", changed, resync, err)
	}

	diverged := &HeadBoundView{head: artifact.CommitID{0xde, 0xad}, sequence: 1, projection: overgodb.ProjectionContractVersion()}
	changed, resync, _, err = diverged.Reconcile(ctx, store)
	if err != nil || !changed || !resync {
		t.Fatalf("diverged view = (changed=%t, resync=%t, %v), want resync", changed, resync, err)
	}
	changed, resync, _, err = diverged.Reconcile(ctx, store)
	if err != nil || changed || resync {
		t.Fatalf("resynced view = (changed=%t, resync=%t, %v), want continuity", changed, resync, err)
	}
}
