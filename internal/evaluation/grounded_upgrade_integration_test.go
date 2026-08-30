package evaluation

import (
	"slices"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/dataset"
	"overgo/internal/executionfailure"
	"overgo/internal/overgodb"
	"overgo/internal/runrecord"
	"overgo/internal/testutil"
)

func TestGroundedCapabilityUpgradeEndToEnd(t *testing.T) {
	knowledge := newKnowledgePromotionFixture(t)
	store := knowledge.store
	route := newEvidenceRouteFixtureInStore(t, store, true)
	supervisor := newTrajectorySupervisorFixtureInStore(t, store)
	previous := supervisor.exchange(t, "upgrade-previous", `{}`, nil, nil, nil)
	stuck := supervisor.exchange(t, "upgrade-stuck", `{}`, nil, nil, nil)
	recovered := supervisor.exchange(t, "upgrade-recovered", `{"pivot":true}`, nil, nil, nil)
	episodeSource := publishGroundedReplayEpisodeSource(t, store, previous)

	tokenizer := testutil.ArtifactID(t, artifact.KindTokenizer, "grounded-replay-tokenizer")
	counter := testutil.ArtifactID(t, artifact.KindProfile, "grounded-replay-token-counter")
	if _, err := artifact.CommitBatch(t.Context(), store, artifact.Batch{
		Key:       "evaluation/grounded-replay/measurement-authorities",
		Artifacts: []artifact.Descriptor{{ID: tokenizer}, {ID: counter}},
	}); err != nil {
		t.Fatal(err)
	}
	caseValue, err := retrievalCaseCodec.RequireExactLineage(
		t.Context(), store, knowledge.baseline.Case, RetrievalCase.Lineage,
	)
	if err != nil {
		t.Fatal(err)
	}
	request := SupervisedGroundedReplayRequest{
		RunRecipe: knowledge.transform.Spec().Operation,
		EpisodeBounds: dataset.CapabilityEpisodeProjectionBounds{
			MaximumEpisodes: 1, MaximumEventsPerEpisode: 8,
			MaximumCallsPerEpisode: 2, MaximumReferencesPerEpisode: 16,
		},
		EpisodeSources: []runrecord.CapabilityEpisodeAuthoritySource{episodeSource},
		ArcTrajectory:  previous.ID,
		ArcBounds: dataset.InteractionArcProjectionBounds{
			MaximumBranches: 2, MaximumArcs: 8, MaximumProjectedSequences: 32,
			MaximumToolPairs: 4, MaximumReferences: 32,
		},
		ArcSelection: dataset.InteractionSelectionBounds{
			MaxTokens: 1 << 20, MaxBytes: 1 << 20, MaxDocuments: 8, MaxDepth: 4, MaxResults: 8,
		},
		ArcTokenizer: tokenizer,
		ArcCounter:   counter,
		RetrievalSources: []GroundedReplaySource{
			{Kind: artifact.KindFile, Documents: []dataset.AgentRetrievalDocument{{
				Text: "target distractor", Facet: dataset.AgentRetrievalFacetText,
				Structure: []string{"fixture", "knowledge-baseline"},
			}}},
			{Kind: artifact.KindFile, Documents: []dataset.AgentRetrievalDocument{{
				Text: "target answer", Facet: dataset.AgentRetrievalFacetText,
				Structure: []string{"fixture", "knowledge-trial"},
			}}},
		},
		BaselineCase:    caseValue,
		BaselineReceipt: knowledge.baseline.Receipt,
		TrialCase:       caseValue,
		TrialReceipt:    knowledge.trial.Receipt,
		Transform:       knowledge.transform.Spec(),
		Probes:          route.probes,
		Route:           route.request,
		RouteCandidates: route.candidates,
		RouteFailure: runrecord.FailureObservation{
			Source: "grounded-replay", Message: "connection refused",
			ObservedUnixNS: 1_700_000_000_000_000_101,
		},
		RouteMaxAttempts:   2,
		Trajectories:       [3]artifact.ID{previous.ID, stuck.ID, recovered.ID},
		KnowledgeAdmission: knowledge.admission.ID,
	}
	result, err := RunSupervisedGroundedReplay(t.Context(), store, request)
	if err != nil {
		t.Fatal(err)
	}
	if result.Run.Outcome != runrecord.OutcomeSucceeded ||
		!slices.Equal(result.Run.Outputs, []artifact.ID{result.Summary}) ||
		result.Route.Disposition != runrecord.RouteSelected || result.Route.Winner != route.wantWinner ||
		result.Fallback.Winner == result.Route.Winner || !result.Fallback.Winner.Valid() ||
		result.Trajectories[0].Decision != executionfailure.DecisionRetireSession ||
		result.Trajectories[1].Decision != executionfailure.DecisionResume ||
		result.Promotion.Output != knowledge.proposal.Output || len(result.Promotion.Benefits) == 0 {
		t.Fatalf("supervised grounded replay = %+v", result)
	}

	if err := store.Close(); err != nil {
		t.Fatal(err)
	}
	reopened, err := overgodb.Open(knowledge.path)
	if err != nil {
		t.Fatal(err)
	}
	defer reopened.Close()
	requireColdGroundedReplayResult(t, reopened, request, result)
}

func publishGroundedReplayEpisodeSource(
	t *testing.T,
	store *overgodb.Store,
	trace runrecord.InteractionTrace,
) runrecord.CapabilityEpisodeAuthoritySource {
	t.Helper()
	gateRecipe := testutil.ArtifactID(t, artifact.KindRecipe, "grounded-replay-gate-recipe")
	environment := testutil.ArtifactID(t, artifact.KindEvidence, "grounded-replay-environment")
	const codeCommit = "0123456789abcdef0123456789abcdef01234567"
	gate, err := runrecord.NewGateRecord(
		gateRecipe, environment, codeCommit, runrecord.OutcomeSucceeded, "", 1,
		[]runrecord.GateStep{{Name: "verify", Phase: runrecord.PhaseValidate, Outcome: runrecord.StepSucceeded, DurationNS: 1}},
	)
	if err != nil {
		t.Fatal(err)
	}
	attempt, err := runrecord.NewAttemptRecord(runrecord.AttemptRecord{
		PlanItem: "grounded-upgrade", PlanStep: "supervised-replay", Strategy: "supervised-grounded",
		Result: gate.Result.ID, Recipe: gateRecipe, CodeCommit: codeCommit,
		Outcome: runrecord.OutcomeSucceeded, WallNS: 1, TaskContract: trace.TaskContract,
		Environment: environment, Trajectory: trace.ID, CostUnits: 1,
	})
	if err != nil {
		t.Fatal(err)
	}
	gateContent, err := gate.Result.Content()
	if err != nil {
		t.Fatal(err)
	}
	attemptContent, err := attempt.Content()
	if err != nil {
		t.Fatal(err)
	}
	lineage := append(gate.Result.Lineage(), attempt.Lineage()...)
	batch, err := artifact.NewDocumentBatch(
		"evaluation/grounded-replay/episode-source",
		[]artifact.Content{gateContent, attemptContent}, lineage, nil,
	)
	if err != nil {
		t.Fatal(err)
	}
	batch.Artifacts = []artifact.Descriptor{{ID: gateRecipe}, {ID: environment}}
	if _, err := artifact.CommitBatch(t.Context(), store, batch); err != nil {
		t.Fatal(err)
	}
	return runrecord.CapabilityEpisodeAuthoritySource{Attempt: attempt.ID, Trajectory: trace.ID}
}

func requireColdGroundedReplayResult(
	t *testing.T,
	store *overgodb.Store,
	request SupervisedGroundedReplayRequest,
	result SupervisedGroundedReplayResult,
) {
	t.Helper()
	if _, err := requireGroundedReplayRun(t.Context(), store, result.Run.ID); err != nil {
		t.Fatal(err)
	}
	if _, err := requireGroundedReplaySummary(t.Context(), store, result.Summary, result.Run.ID); err != nil {
		t.Fatal(err)
	}
	if _, err := runrecord.RequireCapabilityEpisodeProjection(t.Context(), store, result.Episode.ID); err != nil {
		t.Fatal(err)
	}
	if _, _, err := runrecord.RequireInteractionArcProjection(t.Context(), store, result.ArcProjection.ID); err != nil {
		t.Fatal(err)
	}
	if _, err := runrecord.RequireInteractionArcSelection(t.Context(), store, result.ArcSelection.ID); err != nil {
		t.Fatal(err)
	}
	for _, probe := range result.Probes {
		if _, err := RequireCapabilityProbeResult(t.Context(), store, probe.ID); err != nil {
			t.Fatal(err)
		}
	}
	for _, id := range []artifact.ID{result.Route.ID, result.Fallback.ID} {
		if _, err := runrecord.RequireRoutingDecision(t.Context(), store, id); err != nil {
			t.Fatal(err)
		}
	}
	if _, err := runrecord.RequireTerminalAttemptReceipt(t.Context(), store, result.RouteFailure.ID); err != nil {
		t.Fatal(err)
	}
	for _, decision := range result.Trajectories {
		if _, err := RequireTrajectoryHealthDecision(t.Context(), store, decision.ID); err != nil {
			t.Fatal(err)
		}
	}
	for _, evaluation := range []RetrievalEvaluation{result.Baseline, result.Trial} {
		if _, err := RequireRetrievalEvaluation(t.Context(), store, evaluation.ID); err != nil {
			t.Fatal(err)
		}
	}
	for _, receipt := range []artifact.ID{request.BaselineReceipt, request.TrialReceipt} {
		if _, err := runrecord.RequireConsumedRetrievalReceipt(t.Context(), store, receipt); err != nil {
			t.Fatal(err)
		}
	}
	if _, err := dataset.RequireDatasetTransform(t.Context(), store, result.Transform.ID()); err != nil {
		t.Fatal(err)
	}
	if _, err := RequireEpisodeKnowledgeProposal(t.Context(), store, result.Proposal.ID); err != nil {
		t.Fatal(err)
	}
	if _, err := RequireKnowledgePromotionEvidence(t.Context(), store, result.Promotion.ID); err != nil {
		t.Fatal(err)
	}
}
