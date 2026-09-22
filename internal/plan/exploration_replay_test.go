package plan

import (
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/testutil"
)

// TestExplorationGrantReplayKeepsCharges holds a grant replayed from its
// committed bytes to the same balance: a charge against the original still
// counts against the replay, and a continuation past the budget is refused.
// This round trip lived in the media resource acceptance; it is the plan's
// contract and binds nothing but the plan.
func TestExplorationGrantReplayKeepsCharges(t *testing.T) {
	t.Parallel()
	id := func(label string) artifact.ID {
		return testutil.ArtifactID(t, artifact.KindEvidence, "exploration replay "+label)
	}
	grant, err := NewExplorationGrant(id("proposer"), id("authority"), 2)
	if err != nil {
		t.Fatal(err)
	}
	charge, err := NewExplorationCharge(grant.ID, id("experiment"), 1)
	if err != nil {
		t.Fatal(err)
	}
	content, err := grant.Content()
	if err != nil {
		t.Fatal(err)
	}
	replayed, err := ParseExplorationGrant(content.Data)
	if err != nil {
		t.Fatal(err)
	}
	remaining, err := ExplorationBalance(replayed, []ExplorationCharge{charge})
	if err != nil || remaining != 1 {
		t.Fatalf("replay reset charged budget: remaining=%d err=%v", remaining, err)
	}
	decision, err := AdmitExploration(replayed, []ExplorationCharge{charge}, 2, false)
	if err != nil || decision.Admitted {
		t.Fatalf("over-budget continuation accepted: %+v %v", decision, err)
	}
}
