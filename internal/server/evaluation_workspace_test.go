package server

import (
	"context"
	"encoding/json"
	"net/http"
	"slices"
	"strings"
	"sync"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/capabilityruntime"
	"overgo/internal/dataset"
	"overgo/internal/evaluation"
	"overgo/internal/invocation"
	"overgo/internal/operation"
	"overgo/internal/runrecord"
	"overgo/internal/testutil"
)

const evaluationTestCommit = "0123456789abcdef0123456789abcdef01234567"

type evaluationWorkspaceFixture struct {
	capability EvaluationCapability
	run        artifact.ID
	mu         sync.Mutex
	model      artifact.ID
	plans      []artifact.ID
	replay     *evaluation.SupervisedGroundedReplayRequest
}

func (fixture *evaluationWorkspaceFixture) EvaluationCapabilities(context.Context) ([]EvaluationCapability, error) {
	return []EvaluationCapability{fixture.capability}, nil
}

func (fixture *evaluationWorkspaceFixture) ExecuteEvaluation(
	_ context.Context,
	model artifact.ID,
	plans []artifact.ID,
	reporter operation.Reporter,
) (operation.Completion, error) {
	fixture.mu.Lock()
	fixture.model, fixture.plans = model, slices.Clone(plans)
	fixture.mu.Unlock()
	total := uint64(len(plans))
	reporter.Progress(total, &total)
	return operation.Completion{Run: fixture.run}, nil
}

func (fixture *evaluationWorkspaceFixture) ExecuteSupervisedGroundedReplay(
	_ context.Context,
	model artifact.ID,
	request evaluation.SupervisedGroundedReplayRequest,
	reporter operation.Reporter,
) (operation.Completion, error) {
	fixture.mu.Lock()
	fixture.model, fixture.replay = model, &request
	fixture.mu.Unlock()
	total := uint64(1)
	reporter.Progress(total, &total)
	return operation.Completion{Run: fixture.run}, nil
}

func (*evaluationWorkspaceFixture) EvaluationHistory(context.Context, artifact.ID) ([]EvaluationHistoryEntry, error) {
	return nil, nil
}

func (*evaluationWorkspaceFixture) EvaluationReport(context.Context, artifact.ID) (EvaluationReport, error) {
	return EvaluationReport{}, nil
}

func (*evaluationWorkspaceFixture) EvaluationFailures(context.Context, artifact.ID) ([]EvaluationFailure, error) {
	return nil, nil
}

func (*evaluationWorkspaceFixture) CompareEvaluations(context.Context, artifact.ID, artifact.ID) (EvaluationComparison, error) {
	return EvaluationComparison{}, nil
}

func TestEvaluationCampaignAPIUsesCompiledPlans(t *testing.T) {
	modelID := testutil.ArtifactID(t, artifact.KindModel, "evaluation-api-model")
	recipeID := testutil.ArtifactID(t, artifact.KindRecipe, "evaluation-api-recipe")
	data, err := json.Marshal(evaluation.ExactSuite{
		Schema: "fixture/v1", Source: "fixture",
		Cases: []evaluation.ExactCase{{
			Name: "case", Prompt: "prompt", MaxTokens: 1,
			Text: "A", PromptTokens: 2, GeneratedTokens: 1,
		}},
	})
	if err != nil {
		t.Fatal(err)
	}
	compiled, err := evaluation.CompileSuite(data, evaluation.ExactAuthorities{
		ModelDefinition: testutil.ArtifactID(t, artifact.KindModelDefinition, "evaluation-api-definition"),
		RuntimeRecipe:   recipeID, CodeCommit: evaluationTestCommit,
		Environment: testutil.ArtifactID(t, artifact.KindEvidence, "evaluation-api-environment"),
		Execution:   evaluation.ExecutionPolicy{Lifecycle: evaluation.LifecycleResident},
	})
	if err != nil {
		t.Fatal(err)
	}
	fixture := &evaluationWorkspaceFixture{
		capability: EvaluationCapability{Model: modelID, Recipe: recipeID, Suite: compiled.Descriptor()},
		run:        testutil.ArtifactID(t, artifact.KindRun, "evaluation-api-run"),
	}
	handler, err := New(Config{Evaluation: fixture}, &fakeGenerator{})
	if err != nil {
		t.Fatal(err)
	}
	defer handler.Close()
	capabilities := serveTestRequest(handler, http.MethodGet, "/evaluations/capabilities", "")
	if capabilities.Code != http.StatusOK || !strings.Contains(capabilities.Body.String(), compiled.Descriptor().Plan.String()) {
		t.Fatalf("capabilities status=%d body=%s", capabilities.Code, capabilities.Body.String())
	}
	body, err := json.Marshal(evaluationRunRequest{Model: modelID, Plans: []artifact.ID{compiled.Descriptor().Plan}})
	if err != nil {
		t.Fatal(err)
	}
	accepted := serveTestRequest(handler, http.MethodPost, "/evaluations/run", string(body))
	if accepted.Code != http.StatusAccepted {
		t.Fatalf("run status=%d body=%s", accepted.Code, accepted.Body.String())
	}
	var submission workflowResponse
	if err := json.Unmarshal(accepted.Body.Bytes(), &submission); err != nil {
		t.Fatal(err)
	}
	waited := serveTestRequest(handler, http.MethodGet, "/operations/wait?id="+submission.Operation.String(), "")
	var status operation.Status
	if err := json.Unmarshal(waited.Body.Bytes(), &status); err != nil {
		t.Fatal(err)
	}
	fixture.mu.Lock()
	selectedModel, selectedPlans := fixture.model, slices.Clone(fixture.plans)
	fixture.mu.Unlock()
	if status.State != operation.StateCompleted || selectedModel != modelID || !slices.Equal(selectedPlans, []artifact.ID{compiled.Descriptor().Plan}) {
		t.Fatalf("status=%+v model=%s plans=%v", status, selectedModel, selectedPlans)
	}
	if len(status.Metrics) != 0 || status.Run == nil || *status.Run != fixture.run {
		t.Fatalf("operation result = %+v", status)
	}
}

func TestEvaluationCampaignAPIRunsSupervisedGroundedReplay(t *testing.T) {
	modelID := testutil.ArtifactID(t, artifact.KindModel, "grounded-replay-api-model")
	recipeID := testutil.ArtifactID(t, artifact.KindRecipe, "grounded-replay-api-recipe")
	fixture := &evaluationWorkspaceFixture{
		capability: EvaluationCapability{
			Model: modelID, Recipe: recipeID,
			Suite: evaluation.SuiteDescriptor{Plan: testutil.ArtifactID(t, artifact.KindProfile, "grounded-replay-api-plan")},
		},
		run: testutil.ArtifactID(t, artifact.KindRun, "grounded-replay-api-run"),
	}
	handler, err := New(Config{Evaluation: fixture, MaxStoredResponses: 4}, &fakeGenerator{})
	if err != nil {
		t.Fatal(err)
	}
	defer handler.Close()
	replay := groundedReplayAPIRequest(t, recipeID)
	body := groundedReplayAPIBody(t, modelID, replay, false)
	accepted := serveTestRequest(handler, http.MethodPost, "/evaluations/run", string(body))
	if accepted.Code != http.StatusAccepted {
		t.Fatalf("run status=%d body=%s", accepted.Code, accepted.Body.String())
	}
	var submission workflowResponse
	if err := json.Unmarshal(accepted.Body.Bytes(), &submission); err != nil {
		t.Fatal(err)
	}
	waited := serveTestRequest(handler, http.MethodGet, "/operations/wait?id="+submission.Operation.String(), "")
	var status operation.Status
	if err := json.Unmarshal(waited.Body.Bytes(), &status); err != nil {
		t.Fatal(err)
	}
	fixture.mu.Lock()
	selectedModel, selectedReplay := fixture.model, fixture.replay
	fixture.mu.Unlock()
	if status.State != operation.StateCompleted || status.Run == nil || *status.Run != fixture.run ||
		selectedModel != modelID || selectedReplay == nil || selectedReplay.RunRecipe != recipeID {
		t.Fatalf("status=%+v model=%s replay=%+v", status, selectedModel, selectedReplay)
	}
}

func TestEvaluationCampaignAPIRejectsMixedOrUnboundedGroundedReplay(t *testing.T) {
	modelID := testutil.ArtifactID(t, artifact.KindModel, "grounded-replay-refusal-model")
	recipeID := testutil.ArtifactID(t, artifact.KindRecipe, "grounded-replay-refusal-recipe")
	planID := testutil.ArtifactID(t, artifact.KindProfile, "grounded-replay-refusal-plan")
	fixture := &evaluationWorkspaceFixture{
		capability: EvaluationCapability{Model: modelID, Recipe: recipeID, Suite: evaluation.SuiteDescriptor{Plan: planID}},
		run:        testutil.ArtifactID(t, artifact.KindRun, "grounded-replay-refusal-run"),
	}
	handler, err := New(Config{Evaluation: fixture, MaxStoredResponses: 2}, &fakeGenerator{})
	if err != nil {
		t.Fatal(err)
	}
	defer handler.Close()
	replay := groundedReplayAPIRequest(t, recipeID)
	for _, test := range []struct {
		name  string
		mixed bool
		alter func(*evaluation.SupervisedGroundedReplayRequest)
	}{
		{name: "mixed plans and replay", mixed: true},
		{name: "foreign recipe", alter: func(value *evaluation.SupervisedGroundedReplayRequest) {
			value.RunRecipe = testutil.ArtifactID(t, artifact.KindRecipe, "grounded-replay-foreign-recipe")
		}},
		{name: "oversized source population", alter: func(value *evaluation.SupervisedGroundedReplayRequest) {
			value.EpisodeSources = append(value.EpisodeSources,
				runrecord.CapabilityEpisodeAuthoritySource{
					Attempt:    testutil.ArtifactID(t, artifact.KindEvidence, "grounded-replay-extra-attempt"),
					Trajectory: testutil.ArtifactID(t, artifact.KindEvidence, "grounded-replay-extra-trajectory"),
				})
			value.EpisodeBounds.MaximumEpisodes = 3
		}},
	} {
		t.Run(test.name, func(t *testing.T) {
			candidate := replay
			candidate.EpisodeSources = slices.Clone(replay.EpisodeSources)
			if test.alter != nil {
				test.alter(&candidate)
			}
			response := serveTestRequest(handler, http.MethodPost, "/evaluations/run", string(groundedReplayAPIBody(t, modelID, candidate, test.mixed)))
			if response.Code != http.StatusBadRequest {
				t.Fatalf("status=%d body=%s", response.Code, response.Body.String())
			}
		})
	}
	fixture.mu.Lock()
	defer fixture.mu.Unlock()
	if fixture.replay != nil || fixture.model.Valid() {
		t.Fatalf("refused request reached workspace: model=%s replay=%+v", fixture.model, fixture.replay)
	}
}

func groundedReplayAPIRequest(t *testing.T, recipeID artifact.ID) evaluation.SupervisedGroundedReplayRequest {
	t.Helper()
	id := func(kind artifact.Kind, name string) artifact.ID {
		return testutil.ArtifactID(t, kind, "grounded-replay-"+name)
	}
	baseline := evaluation.RetrievalCase{
		Query: id(artifact.KindEvidence, "baseline-query"), Split: id(artifact.KindDatasetShard, "baseline-split"),
		Group: "held-out", ExpectAbstain: true,
	}
	trial := evaluation.RetrievalCase{
		Query: id(artifact.KindEvidence, "trial-query"), Split: id(artifact.KindDatasetShard, "trial-split"),
		Group: "held-out", ExpectAbstain: true,
	}
	return evaluation.SupervisedGroundedReplayRequest{
		RunRecipe: recipeID,
		EpisodeBounds: dataset.CapabilityEpisodeProjectionBounds{
			MaximumEpisodes: 2, MaximumEventsPerEpisode: 2,
			MaximumCallsPerEpisode: 2, MaximumReferencesPerEpisode: 2,
		},
		EpisodeSources: []runrecord.CapabilityEpisodeAuthoritySource{
			{Attempt: id(artifact.KindEvidence, "attempt-a"), Trajectory: id(artifact.KindEvidence, "episode-trajectory-a")},
			{Attempt: id(artifact.KindEvidence, "attempt-b"), Trajectory: id(artifact.KindEvidence, "episode-trajectory-b")},
		},
		ArcTrajectory: id(artifact.KindEvidence, "arc-trajectory"),
		ArcBounds: dataset.InteractionArcProjectionBounds{
			MaximumBranches: 2, MaximumArcs: 2, MaximumProjectedSequences: 2,
			MaximumToolPairs: 2, MaximumReferences: 2,
		},
		ArcBranch: "candidate/a",
		ArcSelection: dataset.InteractionSelectionBounds{
			MaxTokens: 2, MaxBytes: 2, MaxDocuments: 2, MaxDepth: 2, MaxResults: 2,
		},
		ArcTokenizer: id(artifact.KindTokenizer, "tokenizer"),
		ArcCounter:   id(artifact.KindProfile, "counter"),
		RetrievalSources: []evaluation.GroundedReplaySource{{
			Kind: artifact.KindFile, Documents: []dataset.AgentRetrievalDocument{{Text: "grounded source"}},
		}},
		BaselineCase: baseline, BaselineReceipt: id(artifact.KindEvidence, "baseline-receipt"),
		TrialCase: trial, TrialReceipt: id(artifact.KindEvidence, "trial-receipt"),
		Transform: dataset.DatasetTransformSpec{
			Registration: id(artifact.KindProfile, "transform-registration"), Operation: recipeID,
			Parameters: id(artifact.KindRecipe, "transform-parameters"), Code: id(artifact.KindEvidence, "transform-code"),
			Environment: id(artifact.KindEvidence, "transform-environment"),
			Inputs:      []artifact.ID{id(artifact.KindFile, "transform-input")}, Outputs: []artifact.ID{id(artifact.KindDatasetShard, "transform-output")},
			ReplayEvidence: id(artifact.KindEvidence, "transform-replay"),
		},
		Probes: []evaluation.CapabilityProbeEvidence{
			{
				Case: id(artifact.KindRecipe, "probe-case-a"), Profile: id(artifact.KindProfile, "probe-profile-a"),
				Placement:   capabilityruntime.ExactCapabilityPlacement{ID: id(artifact.KindProfile, "probe-placement-a")},
				Environment: id(artifact.KindEvidence, "probe-environment-a"), Trace: id(artifact.KindEvidence, "probe-trace-a"),
				StageReceipt: id(artifact.KindEvidence, "probe-stage-a"), CheckDecision: id(artifact.KindEvidence, "probe-check-a"),
				Observation: id(artifact.KindEvidence, "probe-observation-a"),
			},
			{
				Case: id(artifact.KindRecipe, "probe-case-b"), Profile: id(artifact.KindProfile, "probe-profile-b"),
				Placement:   capabilityruntime.ExactCapabilityPlacement{ID: id(artifact.KindProfile, "probe-placement-b")},
				Environment: id(artifact.KindEvidence, "probe-environment-b"), Trace: id(artifact.KindEvidence, "probe-trace-b"),
				StageReceipt: id(artifact.KindEvidence, "probe-stage-b"), CheckDecision: id(artifact.KindEvidence, "probe-check-b"),
				Observation: id(artifact.KindEvidence, "probe-observation-b"),
			},
		},
		Route: evaluation.EvidenceRouteRequest{
			Intent: "select grounded capability", Authority: id(artifact.KindEvidence, "route-authority"),
			AllowedBoundaries: []invocation.Boundary{invocation.BoundaryInternal},
		},
		RouteCandidates: []evaluation.EvidenceRouteCandidate{
			{Probe: id(artifact.KindEvidence, "route-probe-a"), Manual: id(artifact.KindRecipe, "route-manual-a")},
			{Probe: id(artifact.KindEvidence, "route-probe-b"), Manual: id(artifact.KindRecipe, "route-manual-b")},
		},
		RouteFailure: runrecord.FailureObservation{
			Source: "grounded-replay", Message: "connection refused", ObservedUnixNS: 1_700_000_000_000_000_001,
		},
		RouteMaxAttempts: 2,
		Trajectories: [3]artifact.ID{
			id(artifact.KindEvidence, "supervisor-a"), id(artifact.KindEvidence, "supervisor-b"), id(artifact.KindEvidence, "supervisor-c"),
		},
		KnowledgeAdmission: id(artifact.KindEvidence, "knowledge-admission"),
	}
}

func groundedReplayAPIBody(
	t *testing.T,
	model artifact.ID,
	request evaluation.SupervisedGroundedReplayRequest,
	mixed bool,
) []byte {
	t.Helper()
	ids := func(values []artifact.ID) []string {
		result := make([]string, len(values))
		for index, value := range values {
			result[index] = value.String()
		}
		return result
	}
	episodeSources := make([]map[string]any, len(request.EpisodeSources))
	for index, source := range request.EpisodeSources {
		episodeSources[index] = map[string]any{"attempt": source.Attempt.String(), "trajectory": source.Trajectory.String()}
	}
	retrievalSources := make([]map[string]any, len(request.RetrievalSources))
	for index, source := range request.RetrievalSources {
		documents := make([]map[string]any, len(source.Documents))
		for documentIndex, document := range source.Documents {
			documents[documentIndex] = map[string]any{"text": document.Text, "structure": document.Structure}
		}
		retrievalSources[index] = map[string]any{"kind": uint8(source.Kind), "documents": documents}
	}
	probes := make([]map[string]any, len(request.Probes))
	for index, probe := range request.Probes {
		probes[index] = map[string]any{
			"case": probe.Case.String(), "profile": probe.Profile.String(),
			"placement":   map[string]any{"id": probe.Placement.ID.String()},
			"environment": probe.Environment.String(), "trace": probe.Trace.String(),
			"stage_receipt": probe.StageReceipt.String(), "check_decision": probe.CheckDecision.String(),
			"observation": probe.Observation.String(),
		}
	}
	candidates := make([]map[string]any, len(request.RouteCandidates))
	for index, candidate := range request.RouteCandidates {
		candidates[index] = map[string]any{"probe": candidate.Probe.String(), "manual": candidate.Manual.String()}
	}
	boundaries := make([]string, len(request.Route.AllowedBoundaries))
	for index, boundary := range request.Route.AllowedBoundaries {
		boundaries[index] = string(boundary)
	}
	replay := map[string]any{
		"run_recipe": request.RunRecipe.String(), "episode_bounds": request.EpisodeBounds, "episode_sources": episodeSources,
		"arc_trajectory": request.ArcTrajectory.String(), "arc_bounds": request.ArcBounds, "arc_branch": request.ArcBranch,
		"arc_selection": request.ArcSelection, "arc_tokenizer": request.ArcTokenizer.String(), "arc_counter": request.ArcCounter.String(),
		"retrieval_sources": retrievalSources, "baseline_case": request.BaselineCase,
		"baseline_receipt": request.BaselineReceipt.String(), "trial_case": request.TrialCase,
		"trial_receipt": request.TrialReceipt.String(),
		"transform": map[string]any{
			"registration": request.Transform.Registration.String(), "operation": request.Transform.Operation.String(),
			"parameters": request.Transform.Parameters.String(), "code": request.Transform.Code.String(),
			"environment": request.Transform.Environment.String(), "inputs": ids(request.Transform.Inputs),
			"outputs": ids(request.Transform.Outputs), "replay_evidence": request.Transform.ReplayEvidence.String(),
		},
		"probes": probes,
		"route": map[string]any{
			"intent": request.Route.Intent, "authority": request.Route.Authority.String(), "allowed_boundaries": boundaries,
		},
		"route_candidates": candidates,
		"route_failure": map[string]any{
			"source": request.RouteFailure.Source, "message": request.RouteFailure.Message,
			"observed_unix_ns": request.RouteFailure.ObservedUnixNS,
		},
		"route_max_attempts": request.RouteMaxAttempts, "trajectories": ids(request.Trajectories[:]),
		"knowledge_admission": request.KnowledgeAdmission.String(),
	}
	body := map[string]any{"model": model.String(), "grounded_replay": replay}
	if mixed {
		body["plans"] = []string{testutil.ArtifactID(t, artifact.KindProfile, "grounded-replay-refusal-plan").String()}
	}
	data, err := json.Marshal(body)
	if err != nil {
		t.Fatal(err)
	}
	return data
}
