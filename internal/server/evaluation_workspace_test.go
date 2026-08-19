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
	"overgo/internal/testutil"
)

const evaluationTestCommit = "0123456789abcdef0123456789abcdef01234567"

type evaluationWorkspaceFixture struct {
	capability EvaluationCapability
	run        artifact.ID
	mu         sync.Mutex
	model      artifact.ID
	plans      []artifact.ID
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
