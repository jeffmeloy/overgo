package evaluation

import (
	"errors"
	"slices"
	"strings"
	"testing"

	"overgo/internal/agenttool"
	"overgo/internal/artifact"
	"overgo/internal/executionfailure"
	"overgo/internal/invocation"
	"overgo/internal/overgodb"
	"overgo/internal/recipe"
	"overgo/internal/runrecord"
	"overgo/internal/testutil"
)

func TestEvidenceRouteSelectsUniqueDominantCapability(t *testing.T) {
	fixture := newEvidenceRouteFixture(t, true)
	decision, commit, err := CompileEvidenceRoute(t.Context(), fixture.store, fixture.request, fixture.candidates)
	if err != nil {
		t.Fatal(err)
	}
	if !commit.Valid() || decision.Disposition != runrecord.RouteSelected || decision.Winner != fixture.wantWinner {
		t.Fatalf("decision = %+v, want dominant stored selection %s", decision, fixture.wantWinner)
	}
	loaded, err := runrecord.RequireRoutingDecision(t.Context(), fixture.store, decision.ID)
	if err != nil || loaded.ID != decision.ID {
		t.Fatalf("loaded decision = %+v, %v", loaded, err)
	}
	parents, err := fixture.store.Parents(t.Context(), decision.ID)
	if err != nil {
		t.Fatal(err)
	}
	if slices.ContainsFunc(parents, func(edge artifact.Lineage) bool { return edge.Parent == fixture.wantWinner }) {
		t.Fatal("derived selection was published as document lineage")
	}
}

func TestEvidenceRouteRefusesIncomparableFrontier(t *testing.T) {
	fixture := newEvidenceRouteFixture(t, false)
	decision, _, err := CompileEvidenceRoute(t.Context(), fixture.store, fixture.request, fixture.candidates)
	if err != nil {
		t.Fatal(err)
	}
	if decision.Disposition != runrecord.RouteDecisionRequired || decision.Winner.Valid() {
		t.Fatalf("decision = %+v, want explicit decision requirement", decision)
	}
}

func TestEvidenceRouteRejectsSyntheticCandidate(t *testing.T) {
	fixture := newEvidenceRouteFixture(t, true)
	synthetic := testutil.ArtifactID(t, artifact.KindEvidence, "synthetic-route-probe")
	testutil.PublishArtifact(t, fixture.store, synthetic)
	fixture.candidates[0].Probe = synthetic
	if _, _, err := CompileEvidenceRoute(t.Context(), fixture.store, fixture.request, fixture.candidates); err == nil {
		t.Fatal("descriptor-only route probe was admitted")
	}
}

func TestEvidenceRouteRejectsUnregisteredManual(t *testing.T) {
	store := newCapabilityEvaluationStore(t)
	var declaration agenttool.Manual
	declaration.Name = "unregistered.inspect"
	declaration.Description = "Inspect a capability that never entered the registered catalog."
	declaration.Effect = agenttool.EffectInspection
	declaration.Transport.Kind = agenttool.TransportBuiltin
	manual, err := agenttool.NewManual(declaration)
	if err != nil {
		t.Fatal(err)
	}
	data, err := manual.Content()
	if err != nil {
		t.Fatal(err)
	}
	content := artifact.Content{Descriptor: artifact.Descriptor{
		ID: manual.ID, Size: uint64(len(data)), MediaType: agenttool.ManualMediaType, Schema: agenttool.ManualSchema,
	}, Data: data}
	if _, err := artifact.CommitBatch(t.Context(), store, artifact.Batch{
		Key: "test/evidence-route/unregistered-manual", Contents: []artifact.Content{content},
	}); err != nil {
		t.Fatal(err)
	}
	if _, err := requireRegisteredRouteManual(t.Context(), store, manual.ID); err == nil ||
		!strings.Contains(err.Error(), "not currently registered") {
		t.Fatalf("unregistered manual was not refused: %v", err)
	}
}

func TestEvidenceRouteFallbackUsesTypedFailure(t *testing.T) {
	fixture := newEvidenceRouteFixture(t, true)
	selected, _, err := CompileEvidenceRoute(t.Context(), fixture.store, fixture.request, fixture.candidates)
	if err != nil {
		t.Fatal(err)
	}
	failedIndex := slices.IndexFunc(selected.Candidates, func(candidate runrecord.RoutingCandidateObservation) bool {
		return candidate.Selection == selected.Winner
	})
	if failedIndex < 0 {
		t.Fatal("selected decision lacks winner candidate")
	}
	terminal := publishRouteFailure(t, fixture.store, selected.ID, selected.Candidates[failedIndex].Capability,
		executionfailure.CauseNetwork, 2)
	next, commit, disposition, err := FallbackEvidenceRoute(t.Context(), fixture.store, selected.ID, terminal.ID)
	if err != nil || disposition.Decision != executionfailure.DecisionRetry || !commit.Valid() ||
		next.Previous != selected.ID || next.Failure != terminal.ID || !next.Winner.Valid() || next.Winner == selected.Winner {
		t.Fatalf("fallback = %+v, %+v, %v; want recorded alternate decision", next, disposition, err)
	}
	if _, err := runrecord.RequireRoutingDecision(t.Context(), fixture.store, next.ID); err != nil {
		t.Fatal(err)
	}
}

type evidenceRouteFixture struct {
	store      *overgodb.Store
	request    EvidenceRouteRequest
	candidates []EvidenceRouteCandidate
	probes     []CapabilityProbeEvidence
	wantWinner artifact.ID
}

func newEvidenceRouteFixture(t *testing.T, dominant bool) evidenceRouteFixture {
	t.Helper()
	store := newCapabilityEvaluationStore(t)
	return newEvidenceRouteFixtureInStore(t, store, dominant)
}

func newEvidenceRouteFixtureInStore(t *testing.T, store *overgodb.Store, dominant bool) evidenceRouteFixture {
	t.Helper()
	var err error
	first := newProductionProbeFixtureInStore(t, store, "route-first")
	second := newProductionProbeFixtureInStore(t, store, "route-second")
	fixtures := []*productionProbeFixture{&first, &second}
	observations := make([]runrecord.ServingObservation, len(fixtures))
	for index, fixture := range fixtures {
		observations[index], err = runrecord.RequireServingAttemptObservation(t.Context(), store, fixture.request.Observation)
		if err != nil {
			t.Fatal(err)
		}
	}
	low, high := 0, 1
	if observations[1].MeasuredNS < observations[0].MeasuredNS {
		low, high = 1, 0
	}
	extras := [2]int{}
	if dominant {
		extras[low] = 2
	} else {
		extras[high] = 2
	}
	probes := make([]CapabilityProbeResult, len(fixtures))
	probeEvidence := make([]CapabilityProbeEvidence, len(fixtures))
	manuals := make([]agenttool.Manual, len(fixtures))
	for index, fixture := range fixtures {
		fixture.request.Case = first.testCase.ID
		evidence := []artifact.ID{first.testCase.Input, fixture.request.Trace, fixture.receipt.ID, fixture.request.Observation}
		for ordinal := range extras[index] {
			extra := testutil.ArtifactID(t, artifact.KindEvidence, fixture.alias+"-route-extra-"+string(rune('a'+ordinal)))
			testutil.PublishArtifact(t, store, extra)
			evidence = append(evidence, extra)
		}
		check, createErr := recipe.NewDecision(
			first.testCase.ID, recipe.DecisionAccepted, recipe.EvidenceProduction, "",
			recipe.Decider{CodeCommit: "0123456789abcdef0123456789abcdef01234567", Derivation: first.testCase.EvidenceCheck}, evidence,
		)
		if createErr != nil {
			t.Fatal(createErr)
		}
		batch, batchErr := check.Batch("route/check/" + fixture.alias)
		if batchErr != nil {
			t.Fatal(batchErr)
		}
		if _, commitErr := artifact.CommitBatch(t.Context(), store, batch); commitErr != nil &&
			!errors.Is(commitErr, artifact.ErrNoChange) {
			t.Fatal(commitErr)
		}
		fixture.request.CheckDecision = check.ID
		probeEvidence[index] = fixture.request
		probes[index], _, err = PublishProductionCapabilityProbe(t.Context(), store, fixture.request)
		if err != nil {
			t.Fatalf("publish route probe %d: %v", index, err)
		}
		manuals[index], err = agenttool.NewManual(agenttool.Manual{
			Name: "route.inspect." + string(rune('a'+index)), Description: "Inspect one exact production capability.",
			Effect:     agenttool.EffectInspection,
			Transport:  agenttool.Transport{Kind: agenttool.TransportBuiltin, Protocol: "probe-model/1"},
			Capability: fixture.request.Placement.Capability.ID,
		})
		if err != nil {
			t.Fatal(err)
		}
	}
	if _, err := agenttool.PublishManualCatalog(t.Context(), store, manuals); err != nil {
		t.Fatalf("publish route manuals: %v", err)
	}
	derivation := testutil.ArtifactID(t, artifact.KindProfile, "route-policy")
	testutil.PublishArtifact(t, store, derivation)
	authorityEvidence := []artifact.ID{probes[0].ID, probes[1].ID, manuals[0].ID, manuals[1].ID}
	authority, err := recipe.NewDecision(
		first.testCase.ID, recipe.DecisionAccepted, recipe.EvidenceProduction, "",
		recipe.Decider{CodeCommit: "0123456789abcdef0123456789abcdef01234567", Derivation: derivation}, authorityEvidence,
	)
	if err != nil {
		t.Fatal(err)
	}
	authorityBatch, err := authority.Batch("route/authority")
	if err != nil {
		t.Fatal(err)
	}
	if _, err := artifact.CommitBatch(t.Context(), store, authorityBatch); err != nil {
		t.Fatalf("publish route authority: %v", err)
	}
	winner := probes[low].Selection
	return evidenceRouteFixture{
		store:      store,
		request:    EvidenceRouteRequest{Intent: "evidence-lookup", Authority: authority.ID, AllowedBoundaries: []invocation.Boundary{invocation.BoundaryInternal}},
		candidates: []EvidenceRouteCandidate{{Probe: probes[0].ID, Manual: manuals[0].ID}, {Probe: probes[1].ID, Manual: manuals[1].ID}},
		probes:     probeEvidence,
		wantWinner: winner,
	}
}

func publishRouteFailure(
	t *testing.T,
	store *overgodb.Store,
	operation, capability artifact.ID,
	cause executionfailure.Cause,
	maxAttempts uint32,
) runrecord.TerminalAttemptReceipt {
	t.Helper()
	observation, err := runrecord.PublishFailureObservation(t.Context(), store, runrecord.FailureObservation{
		Source: "evidence-route", Message: "connection refused", ObservedUnixNS: 1_700_000_000_000_000_001,
	})
	if err != nil {
		t.Fatal(err)
	}
	normalization := runrecord.NormalizeFailureObservation(observation)
	// Bind the requested deterministic cause without asking fallback callers to
	// classify free-form errors. Tests use the classifier's own cause below.
	if normalization.Cause != cause {
		cause = normalization.Cause
	}
	normalization, err = runrecord.PublishFailureNormalization(t.Context(), store, normalization)
	if err != nil {
		t.Fatal(err)
	}
	disposition := executionfailure.Decide(executionfailure.Situation{Cause: cause, Attempts: 1, MaxAttempts: maxAttempts})
	receipt, err := runrecord.PublishTerminalAttemptReceipt(t.Context(), store, runrecord.TerminalAttemptReceipt{
		Operation: operation, Capability: capability, Outcome: runrecord.OutcomeFailed,
		FailureObservation: observation.ID, FailureNormalization: normalization.ID, Disposition: &disposition,
		Gaps:           []string{runrecord.GapProcessTermination, runrecord.GapResources, runrecord.GapToolOutput, runrecord.GapTranscript},
		ObservedUnixNS: 1_700_000_000_000_000_002,
	})
	if err != nil {
		t.Fatal(err)
	}
	return receipt
}
