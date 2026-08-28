package runrecord

import (
	"context"
	"runtime"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/executionfailure"
	"overgo/internal/overgodb"
	"overgo/internal/processcontrol"
	"overgo/internal/testutil"
)

// TestTerminalAttemptReceiptClosure pins the receipt contract: a
// terminal attempt is self-contained -- outcome, typed failure
// evidence, disposition, process termination, resources, referenced
// blobs, and recovery lineage all resolve from the receipt alone --
// and every absent optional measurement is named as an explicit gap.
func TestTerminalAttemptReceiptClosure(t *testing.T) {
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	ctx := context.Background()
	operation := testutil.ArtifactID(t, artifact.KindEvidence, "terminal-operation")
	implementation := testutil.ArtifactID(t, artifact.KindFile, "terminal-capability-implementation")
	schema := testutil.ArtifactID(t, artifact.KindProfile, "terminal-capability-schema")
	transcript := testutil.ArtifactID(t, artifact.KindFile, "terminal-transcript")
	if _, err := store.Commit(ctx, artifact.Batch{Key: "terminal/fixture/authorities", Artifacts: []artifact.Descriptor{
		{ID: operation}, {ID: implementation}, {ID: schema}, {ID: transcript},
	}}); err != nil {
		t.Fatal(err)
	}
	capability, err := (CapabilityIdentity{
		Implementation: implementation, Release: "1.0.0",
		Transport: CapabilityTransport{Kind: CapabilityTransportBuiltin, Protocol: "terminal-attempt/1"},
		Schema:    schema, Platform: CapabilityPlatform{OS: runtime.GOOS, Arch: runtime.GOARCH},
		Resources: CapabilityResourceEnvelope{
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
	capabilityBatch, err := artifact.NewDocumentBatch("terminal/fixture/capability", []artifact.Content{capabilityContent}, capability.Lineage(), nil)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := store.Commit(ctx, capabilityBatch); err != nil {
		t.Fatal(err)
	}

	observation, err := PublishFailureObservation(ctx, store, FailureObservation{
		Source: "test-lane", Message: "dial tcp 127.0.0.1:9: connection refused",
		ExitCode: 1, ObservedUnixNS: failureFixtureObservedNS,
	})
	if err != nil {
		t.Fatal(err)
	}
	normalization, err := PublishFailureNormalization(ctx, store, NormalizeFailureObservation(observation))
	if err != nil {
		t.Fatal(err)
	}
	disposition := executionfailure.Decide(executionfailure.Situation{
		Cause: normalization.Cause, Attempts: 1, MaxAttempts: 3,
	})
	process := NewProcessTermination(processcontrol.Receipt{
		ExitCode: 1, TreeTerminated: true, StdoutBytes: 11, StderrBytes: 42, WallNS: 5_000_000,
	})

	first, err := PublishTerminalAttemptReceipt(ctx, store, TerminalAttemptReceipt{
		Operation: operation, Capability: capability.ID, Outcome: OutcomeFailed,
		FailureObservation:   observation.ID,
		FailureNormalization: normalization.ID,
		Disposition:          &disposition,
		Process:              &process,
		Resources:            ServingResources{PeakHostBytes: 64},
		Transcript:           transcript,
		Gaps:                 []string{GapToolOutput},
		ObservedUnixNS:       failureFixtureObservedNS,
	})
	if err != nil {
		t.Fatal(err)
	}

	// The retry closes as a second attempt chained to the first, with
	// only its outcome measured: every other facet is an explicit gap.
	second, err := PublishTerminalAttemptReceipt(ctx, store, TerminalAttemptReceipt{
		Operation: operation, Capability: capability.ID, Attempt: 1, Outcome: OutcomeSucceeded,
		Previous:       first.ID,
		Gaps:           []string{GapProcessTermination, GapResources, GapToolOutput, GapTranscript},
		ObservedUnixNS: failureFixtureObservedNS + 1,
	})
	if err != nil {
		t.Fatal(err)
	}

	// The receipt alone reaches everything evaluation and recovery read.
	resolved, found, err := ResolveTerminalAttemptReceipt(ctx, store, operation, 0)
	if err != nil || !found || resolved.ID != first.ID {
		t.Fatalf("attempt 0 resolution = %+v found=%v err=%v", resolved, found, err)
	}
	if resolved.Disposition == nil || resolved.Disposition.Decision != executionfailure.DecisionRetry ||
		resolved.Process == nil || resolved.Process.StderrBytes != 42 {
		t.Fatalf("receipt evidence differs: %+v", resolved)
	}
	if _, err := RequireFailureObservation(ctx, store, resolved.FailureObservation); err != nil {
		t.Fatal(err)
	}
	if _, err := RequireFailureNormalization(ctx, store, resolved.FailureNormalization); err != nil {
		t.Fatal(err)
	}
	chained, found, err := ResolveTerminalAttemptReceipt(ctx, store, operation, 1)
	if err != nil || !found || chained.ID != second.ID || chained.Previous != first.ID {
		t.Fatalf("attempt 1 resolution = %+v found=%v err=%v", chained, found, err)
	}

	refusals := []struct {
		name   string
		mutate func(*TerminalAttemptReceipt)
	}{
		{"failed outcome without failure evidence", func(value *TerminalAttemptReceipt) {
			value.FailureObservation, value.FailureNormalization = artifact.ID{}, artifact.ID{}
			value.Disposition = nil
		}},
		{"success carrying failure evidence", func(value *TerminalAttemptReceipt) {
			value.Outcome = OutcomeSucceeded
		}},
		{"absent measurement without its gap", func(value *TerminalAttemptReceipt) {
			value.Process = nil
		}},
		{"gap naming a present measurement", func(value *TerminalAttemptReceipt) {
			value.Gaps = []string{GapProcessTermination, GapToolOutput}
		}},
		{"foreign gap label", func(value *TerminalAttemptReceipt) {
			value.Gaps = []string{GapToolOutput, "wall-clock"}
		}},
		{"transcript blob of a foreign kind", func(value *TerminalAttemptReceipt) {
			value.Transcript = testutil.ArtifactID(t, artifact.KindEvidence, "not-a-blob")
		}},
		{"recovery lineage without an attempt ordinal", func(value *TerminalAttemptReceipt) {
			value.Previous = first.ID
		}},
		{"foreign disposition decision", func(value *TerminalAttemptReceipt) {
			mutated := *value.Disposition
			mutated.Decision = "abandon"
			value.Disposition = &mutated
		}},
	}
	for _, refusal := range refusals {
		value := TerminalAttemptReceipt{
			Operation: operation, Capability: capability.ID, Outcome: OutcomeFailed,
			FailureObservation:   observation.ID,
			FailureNormalization: normalization.ID,
			Disposition:          &disposition,
			Process:              &process,
			Resources:            ServingResources{PeakHostBytes: 64},
			Transcript:           transcript,
			Gaps:                 []string{GapToolOutput},
			ObservedUnixNS:       failureFixtureObservedNS,
		}
		refusal.mutate(&value)
		if _, err := PublishTerminalAttemptReceipt(ctx, store, value); err == nil {
			t.Fatalf("%s: accepted", refusal.name)
		}
	}
}
