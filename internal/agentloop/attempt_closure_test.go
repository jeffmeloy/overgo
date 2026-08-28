package agentloop

import (
	"context"
	"runtime"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/executionfailure"
	"overgo/internal/runrecord"
	"overgo/internal/testutil"
)

// TestTerminalAttemptReceiptClosure pins the coordinator closure path:
// a failed session attempt closes as one published receipt binding the
// typed failure evidence and the session-derived disposition, and the
// receipt resolves back through the coordinator's own store.
func TestTerminalAttemptReceiptClosure(t *testing.T) {
	coordinator, store := coordinatorFixture(t)
	ctx := context.Background()
	implementation := testutil.ArtifactID(t, artifact.KindFile, "attempt-capability-implementation")
	schema := testutil.ArtifactID(t, artifact.KindProfile, "attempt-capability-schema")
	if _, err := store.Commit(ctx, artifact.Batch{Key: "attempt/capability/parents", Artifacts: []artifact.Descriptor{{ID: implementation}, {ID: schema}}}); err != nil {
		t.Fatal(err)
	}
	capability, err := (runrecord.CapabilityIdentity{
		Implementation: implementation, Release: "1.0.0",
		Transport: runrecord.CapabilityTransport{Kind: runrecord.CapabilityTransportBuiltin, Protocol: "agent-loop/1"},
		Schema:    schema, Platform: runrecord.CapabilityPlatform{OS: runtime.GOOS, Arch: runtime.GOARCH},
		Resources: runrecord.CapabilityResourceEnvelope{
			MaxInputBytes: 4096, MaxOutputBytes: 4096, MaxConcurrent: 1, CPUThreads: 1, HostBytes: 4096,
		},
	}).Identify()
	if err != nil {
		t.Fatal(err)
	}
	capabilityContent, err := capability.Content()
	if err != nil {
		t.Fatal(err)
	}
	capabilityBatch, err := artifact.NewDocumentBatch("attempt/capability", []artifact.Content{capabilityContent}, capability.Lineage(), nil)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := store.Commit(ctx, capabilityBatch); err != nil {
		t.Fatal(err)
	}

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
		Operation: observation.ID, Capability: capability.ID, Outcome: runrecord.OutcomeFailed,
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

func TestExactCapabilityPlacementAndReceipt(t *testing.T) {
	TestTerminalAttemptReceiptClosure(t)
}
