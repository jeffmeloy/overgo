package evaluation

import (
	"context"
	"reflect"
	"runtime"
	"slices"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/executionfailure"
	"overgo/internal/overgodb"
	"overgo/internal/runrecord"
	"overgo/internal/testutil"
)

const coverageObservedUnixNS = int64(1_700_000_000_000_000_000)

// TestEvidenceCoverageProjection pins missing evidence as an explicit result,
// never a favorable score. Every axis keeps its own denominator and degraded
// state, known-zero cost remains observed, classifications are about presence
// rather than cause value, recovery differs from retry, and exact pairing
// requires both declared endpoints to be complete.
func TestEvidenceCoverageProjection(t *testing.T) {
	ctx := t.Context()
	root := t.TempDir()
	store, err := overgodb.Open(root)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = store.Close() })

	id := func(kind artifact.Kind, label string) artifact.ID {
		t.Helper()
		return testutil.ArtifactID(t, kind, "coverage "+label)
	}
	recipe := id(artifact.KindRecipe, "recipe")
	otherRecipe := id(artifact.KindRecipe, "other recipe")
	dataset := id(artifact.KindDataset, "dataset")
	hardware := id(artifact.KindEvidence, "hardware")
	causalRoot := id(artifact.KindEvidence, "causal root")
	completeOutput := id(artifact.KindOutput, "complete output")
	completeRun, err := runrecord.NewRun(
		recipe, runrecord.OutcomeSucceeded, nil, []artifact.ID{completeOutput}, "",
	)
	if err != nil {
		t.Fatal(err)
	}
	complete := completeRun.ID
	partial := id(artifact.KindRun, "partial")
	classifiedRun, err := runrecord.NewRun(recipe, runrecord.OutcomeFailed, nil, nil, "classified_failure")
	if err != nil {
		t.Fatal(err)
	}
	unclassifiedRun, err := runrecord.NewRun(recipe, runrecord.OutcomeFailed, nil, nil, "unclassified_failure")
	if err != nil {
		t.Fatal(err)
	}
	succeededOutput := id(artifact.KindOutput, "succeeded output")
	succeededRun, err := runrecord.NewRun(
		recipe, runrecord.OutcomeSucceeded, nil, []artifact.ID{succeededOutput}, "",
	)
	if err != nil {
		t.Fatal(err)
	}
	classifiedUnknown, unclassified := classifiedRun.ID, unclassifiedRun.ID
	receiptAttempt := id(artifact.KindEvidence, "terminal receipt attempt")
	receiptImplementation := id(artifact.KindFile, "terminal receipt implementation")
	receiptSchema := id(artifact.KindProfile, "terminal receipt schema")
	attemptResult := id(artifact.KindEvidence, "typed attempt result")
	taskContract := id(artifact.KindRecipe, "paired task contract")
	attemptEnvironment := id(artifact.KindEvidence, "paired attempt environment")
	recoveryResult := id(artifact.KindEvidence, "recovery result")
	retryResult := id(artifact.KindEvidence, "retry result")
	missingCausalResult := id(artifact.KindEvidence, "missing causal result")
	splitBrainResult := id(artifact.KindEvidence, "split brain causal result")
	costMismatchResult := id(artifact.KindEvidence, "cost mismatch result")
	foreignTaskContract := id(artifact.KindRecipe, "foreign task contract")
	foreignPairResult := id(artifact.KindEvidence, "foreign pair result")
	typedCausal, err := runrecord.NewCausalRoot(runrecord.TriggerManual, causalRoot)
	if err != nil {
		t.Fatal(err)
	}
	typedAttempt, err := runrecord.NewAttemptRecord(runrecord.AttemptRecord{
		PlanItem: "coverage", PlanStep: "typed-causal", Result: attemptResult,
		Recipe: recipe, CodeCommit: "0123456789abcdef0123456789abcdef01234567",
		Outcome: runrecord.OutcomeSucceeded, WallNS: 1, Causal: &typedCausal,
	})
	if err != nil {
		t.Fatal(err)
	}
	recoveryCausal, err := typedCausal.Derive(runrecord.TriggerRecovery, recoveryResult)
	if err != nil {
		t.Fatal(err)
	}
	retryCausal, err := typedCausal.Derive(runrecord.TriggerRetry, retryResult)
	if err != nil {
		t.Fatal(err)
	}
	newPairedAttempt := func(step string, result artifact.ID, causal *runrecord.CausalContext, cost uint64) runrecord.AttemptRecord {
		t.Helper()
		record, err := runrecord.NewAttemptRecord(runrecord.AttemptRecord{
			PlanItem: "coverage", PlanStep: step, Result: result, Recipe: recipe,
			CodeCommit: "0123456789abcdef0123456789abcdef01234567",
			Outcome:    runrecord.OutcomeSucceeded, WallNS: 1, CostUnits: cost,
			TaskContract: taskContract, Environment: attemptEnvironment, Causal: causal,
		})
		if err != nil {
			t.Fatal(err)
		}
		return record
	}
	recoveryAttempt := newPairedAttempt("recovery", recoveryResult, &recoveryCausal, 0)
	retryAttempt := newPairedAttempt("retry", retryResult, &retryCausal, 7)
	missingCausalAttempt := newPairedAttempt("missing-causal", missingCausalResult, nil, 0)
	splitBrainAttempt := newPairedAttempt("split-brain-causal", splitBrainResult, nil, 0)
	costMismatchAttempt := newPairedAttempt("cost-mismatch", costMismatchResult, nil, 7)
	foreignPairCausal, err := typedCausal.Derive(runrecord.TriggerRetry, foreignPairResult)
	if err != nil {
		t.Fatal(err)
	}
	foreignPairAttempt := newPairedAttempt("foreign-pair", foreignPairResult, &foreignPairCausal, 0)
	foreignPairAttempt.TaskContract, foreignPairAttempt.ID = foreignTaskContract, artifact.ID{}
	foreignPairAttempt, err = runrecord.NewAttemptRecord(foreignPairAttempt)
	if err != nil {
		t.Fatal(err)
	}
	recovery, retry, missingCausal := recoveryAttempt.ID, retryAttempt.ID, missingCausalAttempt.ID
	all := []artifact.ID{
		recipe, otherRecipe, dataset, hardware, causalRoot, partial, completeOutput,
		receiptAttempt, receiptImplementation, receiptSchema, attemptResult, succeededOutput,
		taskContract, attemptEnvironment, recoveryResult, retryResult, missingCausalResult,
		splitBrainResult, costMismatchResult,
		foreignTaskContract, foreignPairResult,
	}
	descriptors := make([]artifact.Descriptor, len(all))
	for index, value := range all {
		descriptors[index] = artifact.Descriptor{ID: value}
	}
	if _, err := store.Commit(ctx, artifact.Batch{Key: "coverage/authorities", Artifacts: descriptors}); err != nil {
		t.Fatal(err)
	}
	typedAttemptContent, err := typedAttempt.Content()
	if err != nil {
		t.Fatal(err)
	}
	if _, err := store.Commit(ctx, artifact.Batch{
		Key: "coverage/attempt/typed-causal", Artifacts: []artifact.Descriptor{typedAttemptContent.Descriptor},
		Contents: []artifact.Content{typedAttemptContent}, Lineage: typedAttempt.Lineage(),
	}); err != nil {
		t.Fatal(err)
	}
	for _, publication := range []struct {
		name   string
		record runrecord.AttemptRecord
	}{
		{name: "recovery", record: recoveryAttempt},
		{name: "retry", record: retryAttempt},
		{name: "missing-causal", record: missingCausalAttempt},
		{name: "split-brain-causal", record: splitBrainAttempt},
		{name: "cost-mismatch", record: costMismatchAttempt},
		{name: "foreign-pair", record: foreignPairAttempt},
	} {
		content, err := publication.record.Content()
		if err != nil {
			t.Fatal(err)
		}
		batch := artifact.Batch{
			Key:       "coverage/attempt/" + publication.name,
			Artifacts: []artifact.Descriptor{content.Descriptor}, Contents: []artifact.Content{content},
			Lineage: publication.record.Lineage(),
		}
		if err := runrecord.BindCausality(&batch, publication.record.ID, publication.record.Causal); err != nil {
			t.Fatal(err)
		}
		if _, err := store.Commit(ctx, batch); err != nil {
			t.Fatal(err)
		}
	}
	if _, err := store.Commit(ctx, artifact.Batch{
		Key: "coverage/attempt/split-brain-causal-facet",
		Causality: []artifact.CausalLink{{
			Execution: splitBrainAttempt.ID, Root: causalRoot,
			Trigger: string(runrecord.TriggerManual),
		}},
	}); err != nil {
		t.Fatal(err)
	}
	receiptCapability, err := (runrecord.CapabilityIdentity{
		Implementation: receiptImplementation, Release: "1.0.0",
		Transport: runrecord.CapabilityTransport{
			Kind: runrecord.CapabilityTransportBuiltin, Protocol: "coverage-receipt/1",
		},
		Schema: receiptSchema,
		Platform: runrecord.CapabilityPlatform{
			OS: runtime.GOOS, Arch: runtime.GOARCH,
		},
		Resources: runrecord.CapabilityResourceEnvelope{
			MaxInputBytes: 4096, MaxOutputBytes: 4096, MaxConcurrent: 1,
			CPUThreads: 1, HostBytes: 4096,
		},
	}).Identify()
	if err != nil {
		t.Fatal(err)
	}
	receiptCapabilityContent, err := receiptCapability.Content()
	if err != nil {
		t.Fatal(err)
	}
	receiptCapabilityBatch, err := artifact.NewDocumentBatch(
		"coverage/capability/terminal-receipt", []artifact.Content{receiptCapabilityContent},
		receiptCapability.Lineage(), nil,
	)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := artifact.CommitBatch(ctx, store, receiptCapabilityBatch); err != nil {
		t.Fatal(err)
	}
	for _, publication := range []struct {
		name   string
		record runrecord.Run
	}{
		{name: "complete", record: completeRun},
		{name: "classified-failure", record: classifiedRun},
		{name: "unclassified-failure", record: unclassifiedRun},
		{name: "contradictory-success", record: succeededRun},
	} {
		batch, err := publication.record.Batch("coverage/run/" + publication.name)
		if err != nil {
			t.Fatal(err)
		}
		if _, err := artifact.CommitBatch(ctx, store, batch); err != nil {
			t.Fatal(err)
		}
	}

	completeChunk, err := runrecord.NewObservationChunk(
		runrecord.ResourceScope{
			Surface: runrecord.SurfaceEvaluation, Hardware: hardware,
			Workload: recipe, Attempt: complete,
		}, artifact.ID{}, []runrecord.ObservationSample{
			{Ordinal: 1, ElapsedNS: 1, Kind: runrecord.ObservationSampleExecution, Measures: []runrecord.ResourceMeasure{
				{Metric: runrecord.ResourceWallNS, Value: 10},
				{Metric: runrecord.ResourceCostUnits, Value: 0},
			}},
			{Ordinal: 2, ElapsedNS: 2, Kind: runrecord.ObservationSampleHardware, Measures: []runrecord.ResourceMeasure{
				{Metric: runrecord.ResourcePeakDeviceBytes, Value: 0},
			}},
		},
	)
	if err != nil {
		t.Fatal(err)
	}
	completeSummary := publishCoverageChunk(t, ctx, store, "complete", completeChunk)
	completeContinuation, err := runrecord.NewObservationChunk(
		completeChunk.Scope, completeSummary.ID, []runrecord.ObservationSample{{
			Ordinal: 3, ElapsedNS: 4, Kind: runrecord.ObservationSampleExecution,
			Measures: []runrecord.ResourceMeasure{{Metric: runrecord.ResourceWallNS, Value: 2}},
		}},
	)
	if err != nil {
		t.Fatal(err)
	}
	completeTail := publishCoverageChunk(t, ctx, store, "complete-continuation", completeContinuation)

	partialChunk, err := runrecord.NewObservationChunk(
		runrecord.ResourceScope{
			Surface: runrecord.SurfaceEvaluation, Hardware: hardware,
			Workload: recipe, Attempt: partial,
		}, artifact.ID{}, []runrecord.ObservationSample{
			{Ordinal: 1, ElapsedNS: 1, Kind: runrecord.ObservationSampleExecution, Measures: []runrecord.ResourceMeasure{
				{Metric: runrecord.ResourceWallNS, Value: 11},
			}},
			{Ordinal: 2, ElapsedNS: 3, Kind: runrecord.ObservationSampleHardware, Measures: []runrecord.ResourceMeasure{
				{Metric: runrecord.ResourcePeakHostBytes, Value: 12},
			}},
		},
	)
	if err != nil {
		t.Fatal(err)
	}
	partialSummary := publishCoverageChunk(t, ctx, store, "partial", partialChunk)

	recoveryChunk, err := runrecord.NewObservationChunk(
		runrecord.ResourceScope{
			Surface: runrecord.SurfaceEvaluation, Hardware: hardware,
			Workload: recipe, Attempt: recovery,
		}, artifact.ID{}, []runrecord.ObservationSample{{
			Ordinal: 1, ElapsedNS: 1, Kind: runrecord.ObservationSampleExecution,
			Interactions: &runrecord.InteractionWork{},
		}},
	)
	if err != nil {
		t.Fatal(err)
	}
	recoverySummary := publishCoverageChunk(t, ctx, store, "recovery", recoveryChunk)
	retryChunk, err := runrecord.NewObservationChunk(
		runrecord.ResourceScope{
			Surface: runrecord.SurfaceEvaluation, Hardware: hardware,
			Workload: recipe, Attempt: retry,
		}, artifact.ID{}, []runrecord.ObservationSample{{
			Ordinal: 1, ElapsedNS: 1, Kind: runrecord.ObservationSampleExecution,
			Interactions: &runrecord.InteractionWork{},
		}},
	)
	if err != nil {
		t.Fatal(err)
	}
	retrySummary := publishCoverageChunk(t, ctx, store, "retry", retryChunk)
	missingScopeChunk, err := runrecord.NewObservationChunk(
		runrecord.ResourceScope{
			Surface: runrecord.SurfaceAgent, Workload: otherRecipe, Attempt: missingCausal,
		}, artifact.ID{}, []runrecord.ObservationSample{{
			Ordinal: 1, ElapsedNS: 1, Kind: runrecord.ObservationSampleExecution,
			Interactions: &runrecord.InteractionWork{},
		}},
	)
	if err != nil {
		t.Fatal(err)
	}
	missingScopeSummary := publishCoverageChunk(t, ctx, store, "missing-causal", missingScopeChunk)
	costMismatchChunk, err := runrecord.NewObservationChunk(
		runrecord.ResourceScope{
			Surface: runrecord.SurfaceAgent, Workload: recipe, Attempt: costMismatchAttempt.ID,
		}, artifact.ID{}, []runrecord.ObservationSample{{
			Ordinal: 1, ElapsedNS: 1, Kind: runrecord.ObservationSampleExecution,
			Measures: []runrecord.ResourceMeasure{{Metric: runrecord.ResourceCostUnits, Value: 8}},
		}},
	)
	if err != nil {
		t.Fatal(err)
	}
	publishCoverageChunk(t, ctx, store, "cost-mismatch", costMismatchChunk)

	evaluationRecord, err := runrecord.NewEvaluation(recipe, complete, dataset, []runrecord.Metric{{
		Name: "quality", Value: 1, Direction: runrecord.DirectionMaximize,
	}})
	if err != nil {
		t.Fatal(err)
	}
	evaluationBatch, err := evaluationRecord.Batch("coverage/evaluation")
	if err != nil {
		t.Fatal(err)
	}
	if _, err := artifact.CommitBatch(ctx, store, evaluationBatch); err != nil {
		t.Fatal(err)
	}

	unknownObservation, err := runrecord.PublishFailureObservation(ctx, store, runrecord.FailureObservation{
		Source: "coverage-fixture", Message: "the operation ended unexpectedly", Detail: "no further output",
		ObservedUnixNS: coverageObservedUnixNS, Run: classifiedUnknown,
	})
	if err != nil {
		t.Fatal(err)
	}
	unknownNormalization := runrecord.NormalizeFailureObservation(unknownObservation)
	if unknownNormalization.Cause != executionfailure.CauseUnknown {
		t.Fatalf("fixture no longer exercises classified unknown: %+v", unknownNormalization)
	}
	if _, err := runrecord.PublishFailureNormalization(ctx, store, unknownNormalization); err != nil {
		t.Fatal(err)
	}
	unclassifiedObservations := make([]artifact.ID, 0, 2)
	firstUnclassified, err := runrecord.PublishFailureObservation(ctx, store, runrecord.FailureObservation{
		Source: "coverage-fixture", Message: "connection refused",
		ObservedUnixNS: coverageObservedUnixNS + 1, Run: unclassified,
	})
	if err != nil {
		t.Fatal(err)
	}
	unclassifiedObservations = append(unclassifiedObservations, firstUnclassified.ID)
	secondUnclassified, err := runrecord.PublishFailureObservation(ctx, store, runrecord.FailureObservation{
		Source: "coverage-fixture", Message: "request timed out",
		ObservedUnixNS: coverageObservedUnixNS + 2, Run: unclassified,
	})
	if err != nil {
		t.Fatal(err)
	}
	unclassifiedObservations = append(unclassifiedObservations, secondUnclassified.ID)
	contradictoryObservation, err := runrecord.PublishFailureObservation(ctx, store, runrecord.FailureObservation{
		Source: "coverage-fixture", Message: "this child contradicts terminal success",
		ObservedUnixNS: coverageObservedUnixNS + 3, Run: succeededRun.ID,
	})
	if err != nil {
		t.Fatal(err)
	}
	receiptObservation, err := runrecord.PublishFailureObservation(ctx, store, runrecord.FailureObservation{
		Source: "coverage-fixture", Message: "connection refused", ExitCode: 1,
		ObservedUnixNS: coverageObservedUnixNS + 4,
	})
	if err != nil {
		t.Fatal(err)
	}
	receiptNormalization, err := runrecord.PublishFailureNormalization(
		ctx, store, runrecord.NormalizeFailureObservation(receiptObservation),
	)
	if err != nil {
		t.Fatal(err)
	}
	receiptDisposition := executionfailure.Decide(executionfailure.Situation{
		Cause: receiptNormalization.Cause, Attempts: 1, MaxAttempts: 2,
	})
	receipt, err := runrecord.PublishTerminalAttemptReceipt(ctx, store, runrecord.TerminalAttemptReceipt{
		Operation: receiptAttempt, Capability: receiptCapability.ID, Outcome: runrecord.OutcomeFailed,
		FailureObservation: receiptObservation.ID, FailureNormalization: receiptNormalization.ID,
		Disposition: &receiptDisposition,
		Gaps: []string{
			runrecord.GapProcessTermination, runrecord.GapResources,
			runrecord.GapToolOutput, runrecord.GapTranscript,
		},
		ObservedUnixNS: coverageObservedUnixNS + 5,
	})
	if err != nil {
		t.Fatal(err)
	}
	runFactBudget, err := NewEvidenceCoverageQuery([]CoverageUnit{{
		Attempt: classifiedUnknown, Terminal: classifiedUnknown,
		FailureObservations: []artifact.ID{unknownObservation.ID},
	}}, nil, coverageTestBounds(2, 1))
	if err != nil {
		t.Fatal(err)
	}
	if _, err := runFactBudget.Project(ctx, store); err == nil {
		t.Fatal("found failure normalization escaped the shared fact budget")
	}
	receiptFactBudget, err := NewEvidenceCoverageQuery([]CoverageUnit{{
		Attempt: receiptAttempt, Terminal: receipt.ID,
	}}, nil, coverageTestBounds(3, 1))
	if err != nil {
		t.Fatal(err)
	}
	receiptFactProjection, err := receiptFactBudget.Project(ctx, store)
	if err != nil || receiptFactProjection.FailuresClassified != 1 {
		t.Fatalf("receipt failure references were not charged exactly once: projection=%+v err=%v", receiptFactProjection, err)
	}
	selectedRun, selectedEvidence, untypedEvidence := publishCoverageSelectedEvaluation(t, ctx, store)
	selectedQuery, err := NewEvidenceCoverageQuery([]CoverageUnit{{
		Attempt: selectedRun.ID, Terminal: selectedRun.ID,
		EvaluationEvidence: []artifact.ID{selectedEvidence.ID},
		Required:           []CoverageRequirement{{Axis: CoverageEvaluation}, {Axis: CoverageCausal}},
	}}, nil, coverageTestBounds(16, 1))
	if err != nil {
		t.Fatal(err)
	}
	selectedProjection, err := selectedQuery.Project(ctx, store)
	if err != nil || selectedProjection.Evaluation.Observed != 1 || selectedProjection.Causal.Observed != 1 {
		t.Fatalf("selected evaluation coverage = %+v err=%v", selectedProjection, err)
	}
	selectedUnit := CoverageUnit{
		Attempt: selectedRun.ID, EvaluationEvidence: []artifact.ID{selectedEvidence.ID},
		Required: []CoverageRequirement{{Axis: CoverageEvaluation}},
	}
	exactEvaluationBudget := newCoverageBudget(coverageTestBounds(3, 1))
	if _, err := coverageEvaluationSources(ctx, store, selectedUnit, selectedRun, &exactEvaluationBudget); err != nil || exactEvaluationBudget.facts != 0 {
		t.Fatalf("selected evaluation facts were not charged exactly once: remaining=%d err=%v", exactEvaluationBudget.facts, err)
	}
	truncatedEvaluationBudget := newCoverageBudget(coverageTestBounds(2, 1))
	if _, err := coverageEvaluationSources(ctx, store, selectedUnit, selectedRun, &truncatedEvaluationBudget); err == nil {
		t.Fatal("selected evaluation escaped its fact budget")
	}
	uncachedRunBudget := newCoverageBudget(coverageTestBounds(4, 1))
	if _, err := coverageEvaluationSources(ctx, store, selectedUnit, runrecord.Run{}, &uncachedRunBudget); err != nil || uncachedRunBudget.facts != 0 {
		t.Fatalf("uncached evaluation run was not charged exactly once: remaining=%d err=%v", uncachedRunBudget.facts, err)
	}
	planCacheBudget := newCoverageBudget(coverageTestBounds(2, 1))
	selectedAuthority := coverageAuthorityFromEvidence(selectedEvidence)
	if _, err := coverageEvaluationPairContext(ctx, store, selectedAuthority, &planCacheBudget); err != nil {
		t.Fatal(err)
	}
	cacheMismatch := selectedAuthority
	cacheMismatch.Environment = id(artifact.KindEvidence, "cached plan foreign environment")
	if _, err := coverageEvaluationPairContext(ctx, store, cacheMismatch, &planCacheBudget); err == nil {
		t.Fatal("cached plan context bypassed current evidence compatibility")
	}
	foreignContractEvidence := selectedEvidence
	foreignContractEvidence.ID = artifact.ID{}
	foreignContractEvidence.Acceptance = id(artifact.KindProfile, "foreign evidence contract acceptance")
	foreignContractEvidence, err = evaluationEvidenceCodec.New(foreignContractEvidence)
	if err != nil {
		t.Fatal(err)
	}
	foreignContractContent, err := foreignContractEvidence.Content()
	if err != nil {
		t.Fatal(err)
	}
	foreignContractContent.Descriptor.MediaType = "application/vnd.overgo.foreign-evaluation-evidence+json"
	foreignContractContent.Descriptor.Schema = "overgo/foreign-evaluation-evidence/v1"
	if _, err := store.Commit(ctx, artifact.Batch{
		Key:       "coverage/selected-evaluation/foreign-contract",
		Artifacts: []artifact.Descriptor{foreignContractContent.Descriptor},
		Contents:  []artifact.Content{foreignContractContent},
	}); err != nil {
		t.Fatal(err)
	}
	foreignContractQuery, err := NewEvidenceCoverageQuery([]CoverageUnit{{
		Attempt: selectedRun.ID, Terminal: selectedRun.ID,
		EvaluationEvidence: []artifact.ID{foreignContractEvidence.ID},
		Required:           []CoverageRequirement{{Axis: CoverageEvaluation}},
	}}, nil, coverageTestBounds(16, 1))
	if err != nil {
		t.Fatal(err)
	}
	if _, err := foreignContractQuery.Project(ctx, store); err == nil {
		t.Fatal("parseable evaluation bytes under a foreign contract were accepted")
	}
	untypedQuery, err := NewEvidenceCoverageQuery([]CoverageUnit{{
		Attempt: selectedRun.ID, Terminal: selectedRun.ID,
		EvaluationEvidence: []artifact.ID{untypedEvidence.ID},
		Required:           []CoverageRequirement{{Axis: CoverageCausal}},
	}}, nil, coverageTestBounds(16, 1))
	if err != nil {
		t.Fatal(err)
	}
	if _, err := untypedQuery.Project(ctx, store); err == nil {
		t.Fatal("indexed causality without typed evaluation evidence was accepted")
	}
	costMismatchQuery, err := NewEvidenceCoverageQuery([]CoverageUnit{{
		Attempt: costMismatchAttempt.ID, Terminal: costMismatchAttempt.ID,
	}}, nil, coverageTestBounds(4, 1))
	if err != nil {
		t.Fatal(err)
	}
	if _, err := costMismatchQuery.Project(ctx, store); err == nil {
		t.Fatal("typed and observed attempt cost disagreement was accepted")
	}
	splitBrainQuery, err := NewEvidenceCoverageQuery([]CoverageUnit{{
		Attempt: splitBrainAttempt.ID, Terminal: splitBrainAttempt.ID,
		Required: []CoverageRequirement{{Axis: CoverageCausal}},
	}}, nil, coverageTestBounds(4, 1))
	if err != nil {
		t.Fatal(err)
	}
	if _, err := splitBrainQuery.Project(ctx, store); err == nil {
		t.Fatal("indexed causality without its authoritative attempt binding was accepted")
	}

	units := []CoverageUnit{
		{Attempt: complete, Required: []CoverageRequirement{
			{Axis: CoverageMeasurement, ExpectedSamples: 1, Metrics: []runrecord.ResourceMetric{
				runrecord.ResourceWallNS, runrecord.ResourceCostUnits,
			}},
			{Axis: CoverageHardware, ExpectedSamples: 1, Metrics: []runrecord.ResourceMetric{
				runrecord.ResourcePeakDeviceBytes,
			}},
			{Axis: CoverageEvaluation},
		}},
		{Attempt: partial, Required: []CoverageRequirement{
			{Axis: CoverageMeasurement, ExpectedSamples: 1, Metrics: []runrecord.ResourceMetric{
				runrecord.ResourceWallNS, runrecord.ResourceInputTokens,
			}},
			{Axis: CoverageHardware, ExpectedSamples: 2, Metrics: []runrecord.ResourceMetric{
				runrecord.ResourcePeakHostBytes,
			}},
			{Axis: CoverageEvaluation},
		}},
		{Attempt: classifiedUnknown, Terminal: classifiedUnknown, FailureObservations: []artifact.ID{unknownObservation.ID}},
		{Attempt: unclassified, Terminal: unclassified, FailureObservations: unclassifiedObservations},
		{Attempt: recovery, Terminal: recovery, Required: []CoverageRequirement{{Axis: CoverageCausal}}},
		{Attempt: retry, Terminal: retry, Required: []CoverageRequirement{{Axis: CoverageCausal}}},
		{Attempt: missingCausal, Terminal: missingCausal, Required: []CoverageRequirement{{Axis: CoverageCausal}}},
		{Attempt: foreignPairAttempt.ID, Terminal: foreignPairAttempt.ID, Required: []CoverageRequirement{{Axis: CoverageCausal}}},
		{Attempt: typedAttempt.ID, Terminal: typedAttempt.ID, Required: []CoverageRequirement{{Axis: CoverageCausal}}},
		{Attempt: receiptAttempt, Terminal: receipt.ID},
	}
	pairs := []CoveragePair{
		{Baseline: recovery, Candidate: retry},
		{Baseline: retry, Candidate: missingCausal},
		{Baseline: recovery, Candidate: foreignPairAttempt.ID},
	}
	bounds := coverageTestBounds(64, 64)
	query, err := NewEvidenceCoverageQuery(units, pairs, bounds)
	if err != nil {
		t.Fatal(err)
	}
	explicitEmptyUnits := slices.Clone(units)
	explicitEmptyUnits[2].Required = []CoverageRequirement{}
	explicitEmpty, err := NewEvidenceCoverageQuery(explicitEmptyUnits, pairs, bounds)
	if err != nil || explicitEmpty.ID != query.ID {
		t.Fatalf("empty required form changed query identity: nil=%s empty=%s err=%v", query.ID, explicitEmpty.ID, err)
	}
	withoutTypedTerminal := slices.Clone(units)
	for index := range withoutTypedTerminal {
		if withoutTypedTerminal[index].Attempt == typedAttempt.ID {
			withoutTypedTerminal[index].Terminal = artifact.ID{}
		}
	}
	terminalOmitted, err := NewEvidenceCoverageQuery(withoutTypedTerminal, pairs, bounds)
	if err != nil || terminalOmitted.ID == query.ID {
		t.Fatalf("exact terminal did not change query identity: exact=%s omitted=%s err=%v", query.ID, terminalOmitted.ID, err)
	}
	head, sequence := store.Head()
	projection, err := query.Project(ctx, store)
	if err != nil {
		t.Fatal(err)
	}
	if projection.Version != EvidenceCoverageProjectionVersion ||
		projection.FailureClassifierVersion != executionfailure.ClassifierVersion ||
		projection.CausalityProjectionVersion != overgodb.CausalityProjectionVersion || projection.Query != query.ID ||
		projection.Head != head || projection.Sequence != sequence || projection.Units != uint64(len(units)) {
		t.Fatalf("projection authority/denominator differs: %+v", projection)
	}
	wantSources := []artifact.ID{
		completeSummary.Chunk, completeTail.Chunk, partialSummary.Chunk,
		recoverySummary.Chunk, retrySummary.Chunk, missingScopeSummary.Chunk,
	}
	slices.SortFunc(wantSources, artifact.CompareID)
	if !slices.Equal(projection.SourceChunks, wantSources) {
		t.Fatalf("projection source chunks = %v, want %v", projection.SourceChunks, wantSources)
	}
	if projection.Measurement != (AxisCoverage{Observed: 1, Degraded: 1}) ||
		projection.Hardware != (AxisCoverage{Observed: 1, Degraded: 1}) ||
		projection.Evaluation != (AxisCoverage{Observed: 1, Missing: 1}) ||
		projection.Causal != (AxisCoverage{Observed: 4, Missing: 1}) {
		t.Fatalf("independent evidence axes collapsed: %+v", projection)
	}
	if projection.Failures != 3 || projection.FailuresClassified != 2 || projection.FailuresUnclassified != 1 {
		t.Fatalf("failure coverage = %+v", projection)
	}
	if projection.Costed != 2 || projection.Uncosted != uint64(len(units)-2) {
		t.Fatalf("known-zero cost or unknown cost collapsed: %+v", projection)
	}
	if projection.Recovered != 1 {
		t.Fatalf("recovery conflated with retry: %+v", projection)
	}
	if projection.Pairs != 3 || projection.Paired != 1 || projection.Unpaired != 2 {
		t.Fatalf("explicit pair coverage = %+v", projection)
	}

	// The traversal budget is part of query identity. A small but otherwise
	// identical query must refuse truncation instead of reporting partial facts.
	factBounded, err := NewEvidenceCoverageQuery(units, pairs, coverageTestBounds(4, 64))
	if err != nil || factBounded.ID == query.ID {
		t.Fatalf("fact-bounded query identity = %s err=%v", factBounded.ID, err)
	}
	if _, err := factBounded.Project(ctx, store); err == nil {
		t.Fatal("coverage projection accepted a truncated fact traversal")
	}
	chunkBounded, err := NewEvidenceCoverageQuery(units, pairs, coverageTestBounds(64, 1))
	if err != nil || chunkBounded.ID == query.ID || chunkBounded.ID == factBounded.ID {
		t.Fatalf("chunk-bounded query identity = %s err=%v", chunkBounded.ID, err)
	}
	if _, err := chunkBounded.Project(ctx, store); err == nil {
		t.Fatal("coverage projection accepted a truncated chunk traversal")
	}
	globalChunkBounds := coverageTestBounds(16, 1)
	globalChunkQuery, err := NewEvidenceCoverageQuery([]CoverageUnit{
		{Attempt: partial}, {Attempt: recovery, Terminal: recovery},
	}, nil, globalChunkBounds)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := globalChunkQuery.Project(ctx, store); err == nil {
		t.Fatal("coverage projection reset its chunk budget between units")
	}
	byteBounds := coverageTestBounds(16, 2)
	byteBounds.Observation.MaxRawBytes = completeTail.Stats.Bytes - 1
	byteQuery, err := NewEvidenceCoverageQuery([]CoverageUnit{{Attempt: complete}}, nil, byteBounds)
	if err != nil || byteQuery.ID == query.ID {
		t.Fatalf("byte-bounded query identity = %s err=%v", byteQuery.ID, err)
	}
	if _, err := byteQuery.Project(ctx, store); err == nil {
		t.Fatal("coverage projection accepted a raw stream beyond its byte budget")
	}
	contradiction, err := NewEvidenceCoverageQuery(
		[]CoverageUnit{{
			Attempt: succeededRun.ID, Terminal: succeededRun.ID,
			FailureObservations: []artifact.ID{contradictoryObservation.ID},
		}}, nil, bounds,
	)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := contradiction.Project(ctx, store); err == nil {
		t.Fatal("failure observation contradicted a succeeded terminal run")
	}
	receiptMismatch, err := NewEvidenceCoverageQuery(
		[]CoverageUnit{{Attempt: recovery, Terminal: receipt.ID}}, nil, bounds,
	)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := receiptMismatch.Project(ctx, store); err == nil {
		t.Fatal("terminal receipt was accepted for a foreign operation")
	}

	for name, build := range map[string]func() error{
		"empty denominator": func() error {
			_, err := NewEvidenceCoverageQuery(nil, nil, coverageTestBounds(1, 1))
			return err
		},
		"zero bound": func() error {
			_, err := NewEvidenceCoverageQuery(units, pairs, CoverageBounds{})
			return err
		},
		"fact bound above population": func() error {
			_, err := NewEvidenceCoverageQuery(
				units, pairs, coverageTestBounds(runrecord.MaximumAttemptPopulation+1, 1),
			)
			return err
		},
		"chunk bound above population": func() error {
			_, err := NewEvidenceCoverageQuery(
				units, pairs, coverageTestBounds(1, runrecord.MaximumAttemptPopulation+1),
			)
			return err
		},
		"unit population above bound": func() error {
			_, err := NewEvidenceCoverageQuery(
				make([]CoverageUnit, runrecord.MaximumAttemptPopulation+1), nil,
				coverageTestBounds(1, 1),
			)
			return err
		},
		"pair population above bound": func() error {
			_, err := NewEvidenceCoverageQuery(
				units, make([]CoveragePair, runrecord.MaximumAttemptPopulation+1),
				coverageTestBounds(1, 1),
			)
			return err
		},
		"unknown axis": func() error {
			_, err := NewEvidenceCoverageQuery(
				[]CoverageUnit{{Attempt: complete, Required: []CoverageRequirement{{Axis: "score"}}}}, nil,
				coverageTestBounds(1, 1),
			)
			return err
		},
		"non-hardware metric": func() error {
			_, err := NewEvidenceCoverageQuery([]CoverageUnit{{Attempt: complete, Required: []CoverageRequirement{{
				Axis: CoverageHardware, ExpectedSamples: 1, Metrics: []runrecord.ResourceMetric{runrecord.ResourceWallNS},
			}}}}, nil, coverageTestBounds(1, 1))
			return err
		},
		"run terminal differs from unit": func() error {
			_, err := NewEvidenceCoverageQuery(
				[]CoverageUnit{{Attempt: complete, Terminal: classifiedUnknown}}, nil,
				coverageTestBounds(1, 1),
			)
			return err
		},
		"evidence unit names run terminal": func() error {
			_, err := NewEvidenceCoverageQuery(
				[]CoverageUnit{{Attempt: recovery, Terminal: classifiedUnknown}}, nil,
				coverageTestBounds(1, 1),
			)
			return err
		},
		"duplicate unit": func() error {
			_, err := NewEvidenceCoverageQuery(
				[]CoverageUnit{units[0], units[0]}, nil,
				coverageTestBounds(1, 1),
			)
			return err
		},
		"foreign pair": func() error {
			_, err := NewEvidenceCoverageQuery(
				units, []CoveragePair{{Baseline: complete, Candidate: id(artifact.KindRun, "foreign")}},
				coverageTestBounds(1, 1),
			)
			return err
		},
		"mismatched pair requirements": func() error {
			_, err := NewEvidenceCoverageQuery(
				units, []CoveragePair{{Baseline: complete, Candidate: recovery}},
				coverageTestBounds(1, 1),
			)
			return err
		},
	} {
		t.Run("refuses "+name, func(t *testing.T) {
			if err := build(); err == nil {
				t.Fatal("accepted")
			}
		})
	}

	if err := store.Close(); err != nil {
		t.Fatal(err)
	}
	cold, err := overgodb.OpenReadOnly(root)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = cold.Close() })
	coldProjection, err := query.Project(ctx, cold)
	if err != nil || !reflect.DeepEqual(coldProjection, projection) {
		t.Fatalf("cold replay projection differs: projection=%+v err=%v", coldProjection, err)
	}
	rebuiltRoot := t.TempDir()
	if _, err := overgodb.Rebuild(ctx, cold, rebuiltRoot, nil); err != nil {
		t.Fatal(err)
	}
	if err := cold.Close(); err != nil {
		t.Fatal(err)
	}
	rebuilt, err := overgodb.OpenReadOnly(rebuiltRoot)
	if err != nil {
		t.Fatal(err)
	}
	defer rebuilt.Close()
	rebuiltProjection, err := query.Project(ctx, rebuilt)
	if err != nil {
		t.Fatal(err)
	}
	rebuiltHead, rebuiltSequence := rebuilt.Head()
	if rebuiltProjection.Head != rebuiltHead || rebuiltProjection.Sequence != rebuiltSequence {
		t.Fatalf("rebuilt projection anchor differs: %+v", rebuiltProjection)
	}
	rebuiltProjection.Head, rebuiltProjection.Sequence = projection.Head, projection.Sequence
	if !reflect.DeepEqual(rebuiltProjection, projection) {
		t.Fatalf("rebuilt evidence coverage differs: rebuilt=%+v original=%+v", rebuiltProjection, projection)
	}
}

func publishCoverageChunk(
	t *testing.T,
	ctx context.Context,
	store *overgodb.Store,
	name string,
	chunk runrecord.ObservationChunk,
) runrecord.ObservationChunkSummary {
	t.Helper()
	batch := artifact.Batch{Key: "coverage/observation/" + name}
	summary, err := runrecord.BindObservationChunk(ctx, store, &batch, chunk)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := artifact.CommitBatch(ctx, store, batch); err != nil {
		t.Fatal(err)
	}
	raw, rawErr := runrecord.RequireObservationChunk(ctx, store, summary.Chunk)
	want, summaryErr := runrecord.SummarizeObservationChunk(raw)
	if rawErr != nil || summaryErr != nil || !reflect.DeepEqual(summary, want) {
		t.Fatalf("%s raw summary differs: published=%+v recomputed=%+v raw_err=%v summary_err=%v", name, summary, want, rawErr, summaryErr)
	}
	stored, err := runrecord.RequireObservationChunkSummary(ctx, store, summary.ID)
	if err != nil || stored.ID != summary.ID {
		t.Fatalf("%s observation round trip = %+v err=%v", name, stored, err)
	}
	return summary
}

func coverageTestBounds(maxFacts, maxChunks int) CoverageBounds {
	return CoverageBounds{
		MaxFacts: maxFacts,
		Observation: runrecord.ObservationStreamBounds{
			MaxChunks: maxChunks, MaxRawBytes: artifact.MaxContentBytes,
		},
	}
}

func publishCoverageSelectedEvaluation(
	t *testing.T,
	ctx context.Context,
	store *overgodb.Store,
) (runrecord.Run, EvaluationEvidence, EvaluationEvidence) {
	t.Helper()
	exact, plan := ledgerFixture(t, "coverage selected evaluation")
	publishPlanFixtureAuthorities(t, store, plan)
	report, err := EvaluateExactSharded(ctx, store, exactGenerator{pieces: []string{"o", "k"}}, exact, plan, nil)
	if err != nil {
		t.Fatal(err)
	}
	metrics := []runrecord.Metric{{Name: exactMetricName, Value: 1, Direction: runrecord.DirectionMaximize}}
	policy, err := newAcceptancePolicy(metrics)
	if err != nil {
		t.Fatal(err)
	}
	evaluator, err := NewEvaluator(plan.identity, policy)
	if err != nil {
		t.Fatal(err)
	}
	run, err := runrecord.NewBoundRun(
		plan.body.RuntimeRecipe, runrecord.OutcomeSucceeded, []artifact.ID{plan.identity}, []artifact.ID{report}, "",
		plan.body.CodeCommit, plan.body.Environment, 10,
		[]runrecord.PhaseMetric{{Phase: runrecord.PhaseValidate, DurationNS: 10}},
	)
	if err != nil {
		t.Fatal(err)
	}
	record, err := runrecord.NewEvaluation(plan.body.RuntimeRecipe, run.ID, plan.body.Dataset, metrics)
	if err != nil {
		t.Fatal(err)
	}
	for _, publication := range []struct {
		key      string
		document evidenceBatchDocument
	}{{"run", run}, {"evaluation", record}} {
		batch, err := publication.document.Batch("coverage/selected-evaluation/" + publication.key)
		if err != nil {
			t.Fatal(err)
		}
		if _, err := artifact.CommitBatch(ctx, store, batch); err != nil {
			t.Fatal(err)
		}
	}
	selectedRoot := testutil.ArtifactID(t, artifact.KindEvidence, "coverage selected evaluation root")
	untypedRoot := testutil.ArtifactID(t, artifact.KindEvidence, "coverage untyped evaluation root")
	if _, err := store.Commit(ctx, artifact.Batch{
		Key:       "coverage/selected-evaluation/causal-authorities",
		Artifacts: []artifact.Descriptor{{ID: selectedRoot}, {ID: untypedRoot}},
	}); err != nil {
		t.Fatal(err)
	}
	causal, err := runrecord.NewCausalRoot(runrecord.TriggerManual, selectedRoot)
	if err != nil {
		t.Fatal(err)
	}
	selected, err := PublishEvaluationEvidence(ctx, store, plan, policy, evaluator, report, run, record, causal)
	if err != nil {
		t.Fatal(err)
	}
	untyped, err := PublishEvaluationEvidence(ctx, store, plan, policy, evaluator, report, run, record)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := store.Commit(ctx, artifact.Batch{
		Key: "coverage/selected-evaluation/untyped-causal-facet",
		Causality: []artifact.CausalLink{{
			Execution: untyped.ID, Root: untypedRoot, Trigger: string(runrecord.TriggerManual),
		}},
	}); err != nil {
		t.Fatal(err)
	}
	return run, selected, untyped
}
