package agentloop

import (
	"context"
	"encoding/json"
	"testing"
)

// TestIncrementalContextPreservesEvidence pins the resumable-session
// contract: a session's first attempt receives the bounded full context, a
// continuous session whose next boundary cites content it already holds
// receives only the identity reference — zero transfer bytes — and a
// stateless session (no held boundary) deterministically rebuilds the full
// context instead of trusting a cursor it cannot prove.
func TestIncrementalContextPreservesEvidence(t *testing.T) {
	ctx := context.Background()
	coordinator, _ := coordinatorFixture(t)
	session := &Session{ID: "incremental-session"}
	arguments := json.RawMessage(`{"step":1}`)
	if _, err := coordinator.Propose(ctx, session, "probe.read", arguments, false); err != nil {
		t.Fatal(err)
	}
	if !session.handoff.full || len(session.handoff.fresh) != 1 || len(session.handoff.reused) != 0 {
		t.Fatalf("first attempt handoff = %+v, want bounded full context", session.handoff)
	}
	firstBoundary := session.handoff.boundary
	firstSource := session.handoff.fresh[0].Source

	if _, err := coordinator.Propose(ctx, session, "probe.read", arguments, false); err != nil {
		t.Fatal(err)
	}
	if session.handoff.full || session.handoff.boundary == firstBoundary {
		t.Fatalf("continuous attempt handoff = %+v, want a distinct cursor delta", session.handoff)
	}
	if len(session.handoff.reused) != 1 || session.handoff.reused[0] != firstSource || len(session.handoff.fresh) != 0 {
		t.Fatalf("repeated content was not reused by identity: %+v", session.handoff)
	}

	resumed := &Session{ID: "resumed-session"}
	if _, err := coordinator.Propose(ctx, resumed, "probe.read", arguments, false); err != nil {
		t.Fatal(err)
	}
	if !resumed.handoff.full || len(resumed.handoff.fresh) != 1 {
		t.Fatalf("stateless session handoff = %+v, want deterministic full rebuild", resumed.handoff)
	}

	varied := &Session{ID: "varied-session"}
	if _, err := coordinator.Propose(ctx, varied, "probe.read", arguments, false); err != nil {
		t.Fatal(err)
	}
	if _, err := coordinator.Propose(ctx, varied, "probe.read", json.RawMessage(`{"step":2}`), false); err != nil {
		t.Fatal(err)
	}
	if varied.handoff.full || len(varied.handoff.fresh) != 1 || len(varied.handoff.reused) != 0 {
		t.Fatalf("new content handoff = %+v, want one fresh arc", varied.handoff)
	}
}
