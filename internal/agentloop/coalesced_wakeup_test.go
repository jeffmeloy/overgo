package agentloop

import (
	"context"
	"encoding/json"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/runrecord"
	"overgo/internal/testutil"
)

// TestCoordinationCoalescesWithoutLostWork pins the coordinator's wakeup
// coalescing: duplicate late-stimulus notifications for one consumed
// boundary converge on the same durable admission with exactly one winner,
// repeated flaps never re-admit, and after the flurry no boundary is left
// silently pending — every admitted wakeup was processed or is visible.
func TestCoordinationCoalescesWithoutLostWork(t *testing.T) {
	ctx := context.Background()
	coordinator, store := coordinatorFixture(t)
	session := &Session{ID: "coalesce-session"}
	if _, err := coordinator.Propose(ctx, session, "probe.read", json.RawMessage(`{"step":1}`), false); err != nil {
		t.Fatal(err)
	}
	interaction, found, err := runrecord.ResolveInteraction(ctx, store, "coalesce-session-step-1")
	if err != nil || !found {
		t.Fatalf("interaction = (%t, %v)", found, err)
	}
	late := testutil.ArtifactID(t, artifact.KindEvidence, "coalesced-late-stimulus")
	if _, err := store.Commit(ctx, artifact.Batch{
		Key: "agent/coalesce/late", Artifacts: []artifact.Descriptor{{ID: late}},
	}); err != nil {
		t.Fatal(err)
	}
	first, won, err := coordinator.ReconcileLateStimuli(ctx, interaction.Stimulus, []artifact.ID{late})
	if err != nil || !won {
		t.Fatalf("first wakeup = (%+v, %t, %v)", first, won, err)
	}
	winners := 0
	for range 5 {
		followup, fresh, err := coordinator.ReconcileLateStimuli(ctx, interaction.Stimulus, []artifact.ID{late})
		if err != nil {
			t.Fatal(err)
		}
		if fresh {
			winners++
		}
		if followup.ID != first.ID {
			t.Fatalf("flapped wakeup diverged from the durable admission: %s vs %s", followup.ID, first.ID)
		}
	}
	if winners != 0 {
		t.Fatalf("repeated wakeups re-admitted %d times", winners)
	}
	if pending := coordinator.wakeups.Pending(); len(pending) != 0 {
		t.Fatalf("wakeups left silently pending: %v", pending)
	}
}
