package agentloop

import (
	"context"
	"encoding/json"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/runrecord"
	"overgo/internal/testutil"
)

func TestLateStimulusReconciliation(t *testing.T) {
	ctx := context.Background()
	coordinator, store := coordinatorFixture(t)
	session := &Session{ID: "followup-session"}
	if _, err := coordinator.Propose(ctx, session, "probe.read", json.RawMessage(`{"step":1}`), false); err != nil {
		t.Fatal(err)
	}
	interaction, found, err := runrecord.ResolveInteraction(ctx, store, "followup-session-step-1")
	if err != nil || !found {
		t.Fatalf("interaction = (%+v, %t, %v)", interaction, found, err)
	}
	late := testutil.ArtifactID(t, artifact.KindEvidence, "agent-late-stimulus")
	if _, err := store.Commit(ctx, artifact.Batch{Key: "agent/followup/late", Artifacts: []artifact.Descriptor{{ID: late}}}); err != nil {
		t.Fatal(err)
	}
	first, won, err := coordinator.ReconcileLateStimuli(ctx, interaction.Stimulus, []artifact.ID{late, late})
	if err != nil || !won {
		t.Fatalf("first reconciliation = (%+v, %t, %v)", first, won, err)
	}
	second, won, err := coordinator.ReconcileLateStimuli(ctx, interaction.Stimulus, []artifact.ID{late})
	if err != nil || won || second.ID != first.ID {
		t.Fatalf("duplicate reconciliation = (%+v, %t, %v)", second, won, err)
	}
}
