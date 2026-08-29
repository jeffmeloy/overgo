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
	ctx := t.Context()
	operation := testutil.ArtifactID(t, artifact.KindEvidence, "terminal-operation")
	implementation := testutil.ArtifactID(t, artifact.KindFile, "terminal-capability-implementation")
	schema := testutil.ArtifactID(t, artifact.KindProfile, "terminal-capability-schema")
	transcript := testutil.ArtifactID(t, artifact.KindFile, "terminal-transcript")
	if _, err := store.Commit(ctx, artifact.Batch{Key: "terminal/fixture/authorities", Artifacts: []artifact.Descriptor{
		{ID: operation}, {ID: implementation}, {ID: schema}, {ID: transcript},
	}}); err != nil {
		t.Fatal(err)
	}
	capability := publishTerminalTestCapability(
		t, ctx, store, implementation, schema, "terminal-attempt/1", "terminal/fixture/capability",
	)

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

// TestTerminalAttemptReceiptReferenceClosure rejects individually canonical
// documents that disagree when they are assembled into one terminal fact.
// Historical classifier output remains readable: reference closure verifies
// the stored tuple and decision policy, not today's normalization rules.
func TestTerminalAttemptReceiptReferenceClosure(t *testing.T) {
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	ctx := t.Context()

	implementation := testutil.ArtifactID(t, artifact.KindFile, "terminal-reference-implementation")
	schema := testutil.ArtifactID(t, artifact.KindProfile, "terminal-reference-schema")
	foreignCapability := testutil.ArtifactID(t, artifact.KindProfile, "terminal-reference-foreign-capability")
	foreignRun := testutil.ArtifactID(t, artifact.KindRun, "terminal-reference-foreign-run")
	missingOperation := testutil.ArtifactID(t, artifact.KindEvidence, "terminal-reference-missing-operation")
	operationNames := []string{
		"historical", "foreign-normalization", "foreign-cause", "forged-policy",
		"foreign-interruption", "foreign-attempt", "zero-budget", "spent-budget",
		"foreign-exit", "foreign-process-interruption", "stored-foreign-normalization",
		"stored-foreign-capability", "run-bound-observation", "stored-run-bound-observation",
	}
	operations := make(map[string]artifact.ID, len(operationNames))
	descriptors := []artifact.Descriptor{{ID: implementation}, {ID: schema}, {ID: foreignCapability}, {ID: foreignRun}}
	for _, name := range operationNames {
		operation := testutil.ArtifactID(t, artifact.KindEvidence, "terminal-reference-"+name)
		operations[name] = operation
		descriptors = append(descriptors, artifact.Descriptor{ID: operation})
	}
	if _, err := store.Commit(ctx, artifact.Batch{Key: "terminal/reference/authorities", Artifacts: descriptors}); err != nil {
		t.Fatal(err)
	}

	capability := publishTerminalTestCapability(
		t, ctx, store, implementation, schema, "terminal-reference/1", "terminal/reference/capability",
	)

	observation, err := PublishFailureObservation(ctx, store, FailureObservation{
		Source: "reference-test", Message: "connection refused", ExitCode: 1,
		ObservedUnixNS: failureFixtureObservedNS,
	})
	if err != nil {
		t.Fatal(err)
	}
	foreignObservation, err := PublishFailureObservation(ctx, store, FailureObservation{
		Source: "reference-test", Message: "deadline exceeded", ExitCode: 1,
		ObservedUnixNS: failureFixtureObservedNS + 1,
	})
	if err != nil {
		t.Fatal(err)
	}
	normalization, err := PublishFailureNormalization(ctx, store, NormalizeFailureObservation(observation))
	if err != nil {
		t.Fatal(err)
	}
	foreignNormalization, err := PublishFailureNormalization(ctx, store, NormalizeFailureObservation(foreignObservation))
	if err != nil {
		t.Fatal(err)
	}
	runBoundObservation, err := PublishFailureObservation(ctx, store, FailureObservation{
		Source: "reference-test", Message: "connection refused", ExitCode: 1,
		ObservedUnixNS: failureFixtureObservedNS + 2, Run: foreignRun,
	})
	if err != nil {
		t.Fatal(err)
	}
	runBoundNormalization, err := PublishFailureNormalization(ctx, store, NormalizeFailureObservation(runBoundObservation))
	if err != nil {
		t.Fatal(err)
	}
	if _, err := PublishFailureObservation(ctx, store, FailureObservation{
		Source: "reference-test", Message: "connection refused",
		ObservedUnixNS: failureFixtureObservedNS + 3, Run: operations["run-bound-observation"],
	}); err == nil {
		t.Fatal("failure observation accepted an evidence operation as its run")
	}

	// A historical classifier may legitimately have reached a different
	// classification. The receipt closes over that stored version instead of
	// silently reclassifying it with today's rules.
	historicalNormalization, err := PublishFailureNormalization(ctx, store, FailureNormalization{
		Observation:       observation.ID,
		ClassifierVersion: executionfailure.ClassifierVersion + 1,
		Cause:             executionfailure.CauseTimeout,
		Rule:              "historical-timeout",
	})
	if err != nil {
		t.Fatal(err)
	}
	if historicalNormalization.Cause == NormalizeFailureObservation(observation).Cause {
		t.Fatal("historical fixture does not differ from the current classifier")
	}
	historicalDisposition := executionfailure.Decide(executionfailure.Situation{
		Cause: historicalNormalization.Cause, Attempts: 1, MaxAttempts: 3,
	})
	historical, err := PublishTerminalAttemptReceipt(ctx, store, TerminalAttemptReceipt{
		Operation: operations["historical"], Capability: capability.ID, Outcome: OutcomeFailed,
		FailureObservation: observation.ID, FailureNormalization: historicalNormalization.ID,
		Disposition:    &historicalDisposition,
		Gaps:           []string{GapProcessTermination, GapResources, GapToolOutput, GapTranscript},
		ObservedUnixNS: failureFixtureObservedNS + 2,
	})
	if err != nil {
		t.Fatalf("historical normalization was refused: %v", err)
	}
	if _, err := RequireTerminalAttemptReceipt(ctx, store, historical.ID); err != nil {
		t.Fatalf("historical receipt could not be required: %v", err)
	}

	base := func(operation artifact.ID) TerminalAttemptReceipt {
		disposition := executionfailure.Decide(executionfailure.Situation{
			Cause: normalization.Cause, Attempts: 1, MaxAttempts: 3,
		})
		return TerminalAttemptReceipt{
			Operation: operation, Capability: capability.ID, Outcome: OutcomeFailed,
			FailureObservation: observation.ID, FailureNormalization: normalization.ID,
			Disposition:    &disposition,
			Gaps:           []string{GapProcessTermination, GapResources, GapToolOutput, GapTranscript},
			ObservedUnixNS: failureFixtureObservedNS + 3,
		}
	}
	refusals := []struct {
		name   string
		mutate func(*TerminalAttemptReceipt)
	}{
		{"foreign-normalization", func(value *TerminalAttemptReceipt) {
			value.FailureNormalization = foreignNormalization.ID
		}},
		{"foreign-cause", func(value *TerminalAttemptReceipt) {
			disposition := executionfailure.Decide(executionfailure.Situation{
				Cause: executionfailure.CauseTimeout, Attempts: 1, MaxAttempts: 3,
			})
			value.Disposition = &disposition
		}},
		{"forged-policy", func(value *TerminalAttemptReceipt) {
			mutated := *value.Disposition
			mutated.Decision = executionfailure.DecisionRefuse
			mutated.Rule = "budget-exhausted"
			value.Disposition = &mutated
		}},
		{"foreign-interruption", func(value *TerminalAttemptReceipt) {
			disposition := executionfailure.Decide(executionfailure.Situation{
				Cause: normalization.Cause, Interrupted: true, Attempts: 1, MaxAttempts: 3,
			})
			value.Disposition = &disposition
		}},
		{"foreign-attempt", func(value *TerminalAttemptReceipt) {
			disposition := executionfailure.Decide(executionfailure.Situation{
				Cause: normalization.Cause, Attempts: 2, MaxAttempts: 3,
			})
			value.Disposition = &disposition
		}},
		{"zero-budget", func(value *TerminalAttemptReceipt) {
			disposition := executionfailure.Decide(executionfailure.Situation{
				Cause: normalization.Cause, Attempts: 1,
			})
			value.Disposition = &disposition
		}},
		{"spent-budget", func(value *TerminalAttemptReceipt) {
			disposition := executionfailure.Decide(executionfailure.Situation{
				Cause: normalization.Cause, Attempts: 2, MaxAttempts: 1,
			})
			value.Disposition = &disposition
		}},
		{"foreign-exit", func(value *TerminalAttemptReceipt) {
			value.Process = &ProcessTermination{ExitCode: 2, WallNS: 1}
			value.Gaps = []string{GapResources, GapToolOutput, GapTranscript}
		}},
		{"foreign-process-interruption", func(value *TerminalAttemptReceipt) {
			value.Process = &ProcessTermination{ExitCode: observation.ExitCode, Interrupted: true, WallNS: 1}
			value.Gaps = []string{GapResources, GapToolOutput, GapTranscript}
		}},
	}
	for _, refusal := range refusals {
		t.Run("publish/"+refusal.name, func(t *testing.T) {
			value := base(operations[refusal.name])
			refusal.mutate(&value)
			if _, err := PublishTerminalAttemptReceipt(ctx, store, value); err == nil {
				t.Fatal("inconsistent receipt was published")
			}
		})
	}
	if _, err := PublishTerminalAttemptReceipt(ctx, store, base(missingOperation)); err == nil {
		t.Fatal("receipt with a missing operation was published")
	}
	runBoundReceipt := base(operations["run-bound-observation"])
	runBoundReceipt.FailureObservation = runBoundObservation.ID
	runBoundReceipt.FailureNormalization = runBoundNormalization.ID
	runBoundDisposition := executionfailure.Decide(executionfailure.Situation{
		Cause: runBoundNormalization.Cause, Attempts: 1, MaxAttempts: 3,
	})
	runBoundReceipt.Disposition = &runBoundDisposition
	if _, err := PublishTerminalAttemptReceipt(ctx, store, runBoundReceipt); err == nil {
		t.Fatal("receipt accepted a failure observation bound to a foreign run")
	}
	runOperation := base(operations["run-bound-observation"])
	runOperation.Operation = foreignRun
	if _, err := PublishTerminalAttemptReceipt(ctx, store, runOperation); err == nil {
		t.Fatal("receipt accepted a run as its evidence operation")
	}

	storedMismatch := base(operations["stored-foreign-normalization"])
	storedMismatch.FailureNormalization = foreignNormalization.ID
	storedMismatch = commitTerminalTestReceiptDirect(t, ctx, store, "terminal/reference/stored-foreign-normalization", storedMismatch)
	if _, err := RequireTerminalAttemptReceipt(ctx, store, storedMismatch.ID); err == nil {
		t.Fatal("stored receipt with a foreign normalization was accepted")
	}

	storedCapability := TerminalAttemptReceipt{
		Operation: operations["stored-foreign-capability"], Capability: foreignCapability,
		Outcome:        OutcomeSucceeded,
		Gaps:           []string{GapProcessTermination, GapResources, GapToolOutput, GapTranscript},
		ObservedUnixNS: failureFixtureObservedNS + 4,
	}
	storedCapability = commitTerminalTestReceiptDirect(t, ctx, store, "terminal/reference/stored-foreign-capability", storedCapability)
	if _, err := RequireTerminalAttemptReceipt(ctx, store, storedCapability.ID); err == nil {
		t.Fatal("stored receipt with an untyped capability was accepted")
	}

	storedOperation := TerminalAttemptReceipt{
		Operation: missingOperation, Capability: capability.ID, Outcome: OutcomeSucceeded,
		Gaps:           []string{GapProcessTermination, GapResources, GapToolOutput, GapTranscript},
		ObservedUnixNS: failureFixtureObservedNS + 5,
	}
	storedOperation = commitTerminalTestReceiptDirect(t, ctx, store, "terminal/reference/stored-missing-operation", storedOperation)
	if _, err := RequireTerminalAttemptReceipt(ctx, store, storedOperation.ID); err == nil {
		t.Fatal("stored receipt with a missing operation was accepted")
	}

	storedRunBound := base(operations["stored-run-bound-observation"])
	storedRunBound.FailureObservation = runBoundObservation.ID
	storedRunBound.FailureNormalization = runBoundNormalization.ID
	storedRunBound.Disposition = &runBoundDisposition
	storedRunBound = commitTerminalTestReceiptDirect(t, ctx, store, "terminal/reference/stored-run-bound-observation", storedRunBound)
	if _, err := RequireTerminalAttemptReceipt(ctx, store, storedRunBound.ID); err == nil {
		t.Fatal("stored receipt with a run-bound failure observation was accepted")
	}
}

// TestTerminalAttemptReceiptRecoveryLineage requires every retry or recovery
// to name the exact immutable prior receipt selected by the same operation's
// preceding attempt alias. A recovered outcome cannot originate at attempt 0.
func TestTerminalAttemptReceiptRecoveryLineage(t *testing.T) {
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	ctx := t.Context()

	implementation := testutil.ArtifactID(t, artifact.KindFile, "terminal-recovery-implementation")
	schema := testutil.ArtifactID(t, artifact.KindProfile, "terminal-recovery-schema")
	missingPrior := testutil.ArtifactID(t, artifact.KindEvidence, "terminal-recovery-missing-prior")
	operations := map[string]artifact.ID{
		"valid":   testutil.ArtifactID(t, artifact.KindEvidence, "terminal-recovery-valid"),
		"missing": testutil.ArtifactID(t, artifact.KindEvidence, "terminal-recovery-missing"),
		"foreign": testutil.ArtifactID(t, artifact.KindEvidence, "terminal-recovery-foreign"),
		"source":  testutil.ArtifactID(t, artifact.KindEvidence, "terminal-recovery-source"),
		"ordinal": testutil.ArtifactID(t, artifact.KindEvidence, "terminal-recovery-ordinal"),
	}
	descriptors := []artifact.Descriptor{{ID: implementation}, {ID: schema}, {ID: missingPrior}}
	for _, operation := range operations {
		descriptors = append(descriptors, artifact.Descriptor{ID: operation})
	}
	if _, err := store.Commit(ctx, artifact.Batch{
		Key:       "terminal/recovery/authorities",
		Artifacts: descriptors,
		Aliases: []artifact.AliasBinding{{
			Name: terminalAttemptAlias(operations["missing"], 0), Target: missingPrior,
		}},
	}); err != nil {
		t.Fatal(err)
	}
	capability := publishTerminalTestCapability(
		t, ctx, store, implementation, schema, "terminal-recovery/1", "terminal/recovery/capability",
	)
	gaps := []string{GapProcessTermination, GapResources, GapToolOutput, GapTranscript}
	receipt := func(operation artifact.ID, attempt uint32, outcome Outcome, previous artifact.ID, observed int64) TerminalAttemptReceipt {
		return TerminalAttemptReceipt{
			Operation: operation, Capability: capability.ID, Attempt: attempt, Outcome: outcome,
			Previous: previous, Gaps: gaps, ObservedUnixNS: observed,
		}
	}

	validPrior, err := PublishTerminalAttemptReceipt(ctx, store, receipt(
		operations["valid"], 0, OutcomeLost, artifact.ID{}, failureFixtureObservedNS,
	))
	if err != nil {
		t.Fatal(err)
	}
	recovered, err := PublishTerminalAttemptReceipt(ctx, store, receipt(
		operations["valid"], 1, OutcomeRecovered, validPrior.ID, failureFixtureObservedNS+1,
	))
	if err != nil {
		t.Fatalf("valid recovered chain was refused: %v", err)
	}
	if required, err := RequireTerminalAttemptReceipt(ctx, store, recovered.ID); err != nil || required.Previous != validPrior.ID {
		t.Fatalf("valid recovered chain = %+v, err=%v", required, err)
	}
	if _, err := PublishTerminalAttemptReceipt(ctx, store, receipt(
		operations["source"], 0, OutcomeRecovered, artifact.ID{}, failureFixtureObservedNS+2,
	)); err == nil {
		t.Fatal("recovered attempt 0 was accepted")
	}

	missing := commitTerminalTestReceiptDirect(t, ctx, store, "terminal/recovery/missing", receipt(
		operations["missing"], 1, OutcomeRecovered, missingPrior, failureFixtureObservedNS+3,
	))
	if _, err := RequireTerminalAttemptReceipt(ctx, store, missing.ID); err == nil {
		t.Fatal("recovered receipt with a missing prior receipt was accepted")
	}

	sourcePrior, err := PublishTerminalAttemptReceipt(ctx, store, receipt(
		operations["source"], 0, OutcomeLost, artifact.ID{}, failureFixtureObservedNS+4,
	))
	if err != nil {
		t.Fatal(err)
	}
	if _, err := PublishTerminalAttemptReceipt(ctx, store, receipt(
		operations["foreign"], 0, OutcomeLost, artifact.ID{}, failureFixtureObservedNS+5,
	)); err != nil {
		t.Fatal(err)
	}
	foreign := commitTerminalTestReceiptDirect(t, ctx, store, "terminal/recovery/foreign", receipt(
		operations["foreign"], 1, OutcomeRecovered, sourcePrior.ID, failureFixtureObservedNS+6,
	))
	if _, err := RequireTerminalAttemptReceipt(ctx, store, foreign.ID); err == nil {
		t.Fatal("recovered receipt with a foreign operation's prior was accepted")
	}

	ordinalZero, err := PublishTerminalAttemptReceipt(ctx, store, receipt(
		operations["ordinal"], 0, OutcomeLost, artifact.ID{}, failureFixtureObservedNS+7,
	))
	if err != nil {
		t.Fatal(err)
	}
	ordinalOne, err := PublishTerminalAttemptReceipt(ctx, store, receipt(
		operations["ordinal"], 1, OutcomeSucceeded, ordinalZero.ID, failureFixtureObservedNS+8,
	))
	if err != nil {
		t.Fatal(err)
	}
	if ordinalOne.ID == ordinalZero.ID {
		t.Fatal("ordinal fixture collapsed distinct receipts")
	}
	wrongOrdinal := commitTerminalTestReceiptDirect(t, ctx, store, "terminal/recovery/wrong-ordinal", receipt(
		operations["ordinal"], 2, OutcomeRecovered, ordinalZero.ID, failureFixtureObservedNS+9,
	))
	if _, err := RequireTerminalAttemptReceipt(ctx, store, wrongOrdinal.ID); err == nil {
		t.Fatal("recovered receipt with a non-adjacent prior ordinal was accepted")
	}
}

func commitTerminalTestReceiptDirect(
	t *testing.T,
	ctx context.Context,
	repository artifact.Repository,
	key string,
	value TerminalAttemptReceipt,
) TerminalAttemptReceipt {
	t.Helper()
	value.Version = artifact.InitialDocumentVersion
	identified, err := terminalAttemptCodec.New(value)
	if err != nil {
		t.Fatal(err)
	}
	batch, err := terminalAttemptCodec.Batch(key, identified, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := repository.Commit(ctx, batch); err != nil {
		t.Fatal(err)
	}
	return identified
}

func publishTerminalTestCapability(
	t *testing.T,
	ctx context.Context,
	repository artifact.Repository,
	implementation, schema artifact.ID,
	protocol, key string,
) CapabilityIdentity {
	t.Helper()
	capability, err := (CapabilityIdentity{
		Implementation: implementation, Release: "1.0.0",
		Transport: CapabilityTransport{Kind: CapabilityTransportBuiltin, Protocol: protocol},
		Schema:    schema, Platform: CapabilityPlatform{OS: runtime.GOOS, Arch: runtime.GOARCH},
		Resources: CapabilityResourceEnvelope{
			MaxInputBytes: 4096, MaxOutputBytes: 4096, MaxConcurrent: 1, CPUThreads: 1, HostBytes: 4096,
		},
	}).Identify()
	if err != nil {
		t.Fatal(err)
	}
	content, err := capability.Content()
	if err != nil {
		t.Fatal(err)
	}
	batch, err := artifact.NewDocumentBatch(key, []artifact.Content{content}, capability.Lineage(), nil)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := repository.Commit(ctx, batch); err != nil {
		t.Fatal(err)
	}
	return capability
}
