package capabilityruntime

import (
	"sync"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/overgodb"
	"overgo/internal/testutil"
)

// TestCoordinationCoalescesWithoutLostWork pins the runtime's probe
// coalescing: concurrent authority freshness probes over an unmoved store
// all resolve without re-derivation and without lost signals, and a store
// movement during the flurry is never dropped — the next probe observes
// exactly the committed change.
func TestCoordinationCoalescesWithoutLostWork(t *testing.T) {
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	ctx := t.Context()
	seed := artifact.Descriptor{ID: testutil.ArtifactID(t, artifact.KindEvidence, "probe-seed"), Size: 1}
	if _, err := store.Commit(ctx, artifact.Batch{
		Key: "probe/seed", Artifacts: []artifact.Descriptor{seed},
		Aliases: []artifact.AliasBinding{{Name: "capability/active", Target: seed.ID}},
	}); err != nil {
		t.Fatal(err)
	}
	view := &HeadBoundView{}
	if changed, resync, _, err := view.Reconcile(ctx, store); err != nil || !changed || !resync {
		t.Fatalf("bootstrap probe = (changed=%t, resync=%t, %v)", changed, resync, err)
	}

	var group sync.WaitGroup
	rederivations := make(chan struct{}, 16)
	for range 8 {
		group.Go(func() {
			changed, resync, _, err := view.Reconcile(ctx, store)
			if err != nil {
				t.Error(err)
				return
			}
			if changed || resync {
				rederivations <- struct{}{}
			}
		})
	}
	group.Wait()
	close(rederivations)
	if extra := len(rederivations); extra != 0 {
		t.Fatalf("unmoved store forced %d re-derivations across concurrent probes", extra)
	}

	moved := artifact.Descriptor{ID: testutil.ArtifactID(t, artifact.KindEvidence, "probe-moved"), Size: 1}
	previous := seed.ID
	if _, err := store.Commit(ctx, artifact.Batch{
		Key: "probe/move", Artifacts: []artifact.Descriptor{moved},
		Aliases: []artifact.AliasBinding{{Name: "capability/active", Target: moved.ID, Previous: &previous}},
	}); err != nil {
		t.Fatal(err)
	}
	changed, resync, delta, err := view.Reconcile(ctx, store)
	if err != nil || !changed || resync {
		t.Fatalf("post-move probe = (changed=%t, resync=%t, %v)", changed, resync, err)
	}
	if len(delta.Aliases) != 1 || delta.Aliases[0].Target != moved.ID {
		t.Fatalf("movement during probing was lost: %+v", delta)
	}
	if changed, resync, _, err := view.Reconcile(ctx, store); err != nil || changed || resync {
		t.Fatalf("reconciled probe = (changed=%t, resync=%t, %v)", changed, resync, err)
	}
}
