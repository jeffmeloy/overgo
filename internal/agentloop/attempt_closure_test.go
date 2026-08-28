package agentloop

import (
	"context"
	"testing"

	"overgo/internal/executionfailure"
	"overgo/internal/runrecord"
)

// TestTerminalAttemptReceiptClosure pins the coordinator closure path:
// a failed session attempt closes as one published receipt binding the
// typed failure evidence and the session-derived disposition, and the
// receipt resolves back through the coordinator's own store.
func TestTerminalAttemptReceiptClosure(t *testing.T) {
	coordinator, store := coordinatorFixture(t)
	ctx := context.Background()

	observation, err := runrecord.PublishFailureObservation(ctx, store, runrecord.FailureObservation{
		Source: "agent-session", Message: "request exceeds the maximum context length",
		ObservedUnixNS: 1_700_000_000_000_000_000,
	})
	if err != nil {
		t.Fatal(err)
	}
	normalization, err := runrecord.PublishFailureNormalization(ctx, store, runrecord.NormalizeFailureObservation(observation))
	if err != nil {
		t.Fatal(err)
	}
	session := &Session{ID: "closure-session", Steps: fixtureSessionSteps}
	disposition := coordinator.SessionDisposition(session, executionfailure.Classification{
		ClassifierVersion: executionfailure.ClassifierVersion,
		Cause:             normalization.Cause, Rule: normalization.Rule,
	}, false, 1, 3, true)
	if disposition.Decision != executionfailure.DecisionFreshSession {
		t.Fatalf("session disposition = %+v", disposition)
	}

	receipt, err := coordinator.CloseAttempt(ctx, runrecord.TerminalAttemptReceipt{
		Operation: observation.ID, Outcome: runrecord.OutcomeFailed,
		FailureObservation:   observation.ID,
		FailureNormalization: normalization.ID,
		Disposition:          &disposition,
		Gaps: []string{
			runrecord.GapProcessTermination, runrecord.GapResources,
			runrecord.GapToolOutput, runrecord.GapTranscript,
		},
		ObservedUnixNS: 1_700_000_000_000_000_001,
	})
	if err != nil {
		t.Fatal(err)
	}
	resolved, found, err := runrecord.ResolveTerminalAttemptReceipt(ctx, store, observation.ID, 0)
	if err != nil || !found || resolved.ID != receipt.ID {
		t.Fatalf("closure resolution = %+v found=%v err=%v", resolved, found, err)
	}
	if resolved.Disposition == nil || resolved.Disposition.Decision != executionfailure.DecisionFreshSession ||
		!resolved.Disposition.Situation.Health.ContextFull {
		t.Fatalf("published disposition differs: %+v", resolved.Disposition)
	}
}
