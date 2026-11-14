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
	"overgo/internal/evaluation"
	"overgo/internal/operation"
	"overgo/internal/testcheck"
	"overgo/internal/testutil"
)

const evaluationTestCommit = "0123456789abcdef0123456789abcdef01234567"

type evaluationWorkspaceFixture struct {
	capability evaluationCapability
	run        artifact.ID
	mu         sync.Mutex
	model      artifact.ID
	plans      []artifact.ID
}

func (fixture *evaluationWorkspaceFixture) EvaluationCapabilities(context.Context) ([]evaluationCapability, error) {
	return []evaluationCapability{fixture.capability}, nil
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

func (*evaluationWorkspaceFixture) EvaluationHistory(context.Context, artifact.ID) ([]evaluationHistoryEntry, error) {
	return nil, nil
}

func (*evaluationWorkspaceFixture) EvaluationReport(context.Context, artifact.ID) (evaluationReport, error) {
	return evaluationReport{}, nil
}

func (*evaluationWorkspaceFixture) EvaluationFailures(context.Context, artifact.ID) ([]evaluationFailure, error) {
	return nil, nil
}

func (*evaluationWorkspaceFixture) CompareEvaluations(context.Context, artifact.ID, artifact.ID) (evaluationComparison, error) {
	return evaluationComparison{}, nil
}

func TestEvaluationCampaignAPIUsesCompiledPlans(t *testing.T) {
	t.Parallel()
	modelID := testutil.ArtifactID(t, artifact.KindModel, "evaluation-api-model")
	recipeID := testutil.ArtifactID(t, artifact.KindRecipe, "evaluation-api-recipe")
	data := testcheck.Must(json.Marshal(evaluation.ExactSuite{
		Schema: "fixture/v1", Source: "fixture",
		Cases: []evaluation.ExactCase{{
			Name: "case", Prompt: "prompt", MaxTokens: 1,
			Text: "A", PromptTokens: 2, GeneratedTokens: 1,
		}},
	}))(t)
	compiled := testcheck.Must(evaluation.CompileSuite(data, evaluation.ExactAuthorities{
		ModelDefinition: testutil.ArtifactID(t, artifact.KindModelDefinition, "evaluation-api-definition"),
		RuntimeRecipe:   recipeID, CodeCommit: evaluationTestCommit,
		Environment: testutil.ArtifactID(t, artifact.KindEvidence, "evaluation-api-environment"),
		Execution:   evaluation.ExecutionPolicy{Lifecycle: evaluation.LifecycleResident},
	}))(t)
	fixture := &evaluationWorkspaceFixture{
		capability: evaluationCapability{Model: modelID, Recipe: recipeID, Suite: compiled.Descriptor()},
		run:        testutil.ArtifactID(t, artifact.KindRun, "evaluation-api-run"),
	}
	handler := testcheck.Must(New(Config{Evaluation: fixture}, &fakeGenerator{}))(t)
	defer handler.Close()
	capabilities := serveTestRequest(handler, http.MethodGet, "/evaluations/capabilities", "")
	if capabilities.Code != http.StatusOK || !strings.Contains(capabilities.Body.String(), compiled.Descriptor().Plan.String()) {
		t.Fatalf("capabilities status=%d body=%s", capabilities.Code, capabilities.Body.String())
	}
	body := testcheck.Must(json.Marshal(evaluationRunRequest{Model: modelID, Plans: []artifact.ID{compiled.Descriptor().Plan}}))(t)
	accepted := serveTestRequest(handler, http.MethodPost, "/evaluations/run", string(body))
	if accepted.Code != http.StatusAccepted {
		t.Fatalf("run status=%d body=%s", accepted.Code, accepted.Body.String())
	}
	var submission workflowResponse
	testcheck.NoError(t, json.Unmarshal(accepted.Body.Bytes(), &submission))
	waited := serveTestRequest(handler, http.MethodGet, "/operations/wait?id="+submission.Operation.String(), "")
	var status operation.Status
	testcheck.NoError(t, json.Unmarshal(waited.Body.Bytes(), &status))
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
