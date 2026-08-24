package server

import (
	"context"
	"encoding/json"
	"net/http"
	"strings"
	"sync"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/operation"
	"overgo/internal/recipe"
	"overgo/internal/repodb"
	"overgo/internal/testutil"
)

type generationWorkspaceGenerator struct {
	*fakeGenerator
	capability WorkflowCapability
	run        artifact.ID
	repository artifact.Repository
	mu         sync.Mutex
	request    generationWorkspaceInput
	executions int
}

type generationWorkspaceInput struct {
	Text string `json:"text"`
	Seed *int64 `json:"seed,omitempty"`
}

func (generator *generationWorkspaceGenerator) WorkflowCapabilities(_ context.Context, kind WorkflowKind) ([]WorkflowCapability, error) {
	if kind != WorkflowGeneration {
		return nil, nil
	}
	return []WorkflowCapability{generator.capability}, nil
}

func (generator *generationWorkspaceGenerator) ExecuteWorkflow(
	ctx context.Context,
	_ WorkflowKind,
	task recipe.Task,
	recipeID artifact.ID,
	raw json.RawMessage,
	reporter operation.Reporter,
) (operation.Completion, error) {
	var input generationWorkspaceInput
	if err := json.Unmarshal(raw, &input); err != nil {
		return operation.Completion{Run: generator.run}, err
	}
	generator.mu.Lock()
	generator.request = input
	generator.executions++
	generator.mu.Unlock()
	if _, err := generator.repository.Commit(ctx, artifact.Batch{
		Key: "fixture/workflow/" + generator.run.String(), Artifacts: []artifact.Descriptor{{ID: generator.run}},
	}); err != nil {
		return operation.Completion{Run: generator.run}, err
	}
	reporter.Publishing()
	return operation.Completion{Run: generator.run}, nil
}

func TestGenerationWorkspaceUsesRecipeCapabilities(t *testing.T) {
	store, err := repodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = store.Close() })
	recipeID := testutil.ArtifactID(t, artifact.KindRecipe, "generation-workspace-recipe")
	runID := testutil.ArtifactID(t, artifact.KindRun, "generation-workspace-run")
	generator := &generationWorkspaceGenerator{
		fakeGenerator: &fakeGenerator{},
		run:           runID,
		repository:    store,
		capability: WorkflowCapability{
			Task: recipe.TaskSpeech, Recipe: recipeID,
			Stages: []recipe.Stage{{Node: recipe.Node{ID: "generate", Module: "test.generate"}}},
			Controls: []WorkflowControl{
				{Name: "text", Type: WorkflowControlText, Required: true},
				{Name: "seed", Type: WorkflowControlInteger},
			},
		},
	}
	handler, err := New(Config{Repository: store}, generator)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = handler.Close() })
	capabilities := serveTestRequest(handler, http.MethodGet, "/generation/capabilities", "")
	if capabilities.Code != http.StatusOK || !strings.Contains(capabilities.Body.String(), `"type":"integer"`) {
		t.Fatalf("capabilities status=%d body=%s", capabilities.Code, capabilities.Body.String())
	}
	requestBody := `{"task":"speech","recipe":"` + recipeID.String() + `","input":{"text":"hello","seed":17}}`
	accepted := serveTestRequest(handler, http.MethodPost, "/generation/run", requestBody)
	if accepted.Code != http.StatusAccepted {
		t.Fatalf("run status=%d body=%s", accepted.Code, accepted.Body.String())
	}
	var submission workflowResponse
	if err := json.Unmarshal(accepted.Body.Bytes(), &submission); err != nil {
		t.Fatal(err)
	}
	waited := serveTestRequest(handler, http.MethodGet, "/operations/wait?id="+submission.Operation.String(), "")
	if waited.Code != http.StatusOK {
		t.Fatalf("wait status=%d body=%s", waited.Code, waited.Body.String())
	}
	var status operation.Status
	if err := json.Unmarshal(waited.Body.Bytes(), &status); err != nil {
		t.Fatal(err)
	}
	if status.State != operation.StateCompleted || status.Run == nil || *status.Run != runID {
		t.Fatalf("operation = %+v", status)
	}
	generator.mu.Lock()
	input := generator.request
	generator.mu.Unlock()
	if input.Text != "hello" || input.Seed == nil || *input.Seed != 17 {
		t.Fatalf("executor input = %+v", input)
	}

	module := serveTestRequest(handler, http.MethodGet, "/workflow.js", "").Body.String()
	for _, token := range []string{"capability.controls", "capability.recipe", "overgo.waitOperation"} {
		if !strings.Contains(module, token) {
			t.Errorf("generation module missing %q", token)
		}
	}
	for _, taskLiteral := range []string{"speech", "image-gen", "video-gen"} {
		registration := serveTestRequest(handler, http.MethodGet, "/mod/generation.js", "").Body.String()
		if strings.Contains(module, taskLiteral) || strings.Contains(registration, taskLiteral) {
			t.Errorf("generation module embeds task %q", taskLiteral)
		}
	}
	unavailable := serveTestRequest(handler, http.MethodGet, "/training/capabilities", "")
	if unavailable.Code != http.StatusNotImplemented || !strings.Contains(unavailable.Body.String(), "training workspace is unavailable") {
		t.Fatalf("unavailable workspace status=%d body=%s", unavailable.Code, unavailable.Body.String())
	}
}

func TestWorkflowSubmissionDoesNotReplayCompletedOperation(t *testing.T) {
	store, err := repodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = store.Close() })
	recipeID := testutil.ArtifactID(t, artifact.KindRecipe, "workflow recovery recipe")
	generator := &generationWorkspaceGenerator{
		fakeGenerator: &fakeGenerator{}, repository: store,
		run: testutil.ArtifactID(t, artifact.KindRun, "workflow recovery run"),
		capability: WorkflowCapability{
			Task: recipe.TaskGeneration, Recipe: recipeID,
			Stages:   []recipe.Stage{{Node: recipe.Node{ID: "generate", Module: "test.generate"}}},
			Controls: []WorkflowControl{{Name: "text", Type: WorkflowControlText, Required: true}},
		},
	}
	handler, err := New(Config{Repository: store}, generator)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = handler.Close() })
	run := func() workflowResponse {
		body, err := json.Marshal(workflowRequest{
			Task: recipe.TaskGeneration, Recipe: recipeID,
			Input: json.RawMessage(`{"text":"run"}`),
		})
		if err != nil {
			t.Fatal(err)
		}
		response := serveTestRequest(handler, http.MethodPost, "/generation/run", string(body))
		if response.Code != http.StatusAccepted {
			t.Fatalf("recovery status=%d body=%s", response.Code, response.Body.String())
		}
		var submission workflowResponse
		if err := json.Unmarshal(response.Body.Bytes(), &submission); err != nil {
			t.Fatal(err)
		}
		waited := serveTestRequest(handler, http.MethodGet, "/operations/wait?id="+submission.Operation.String(), "")
		if waited.Code != http.StatusOK {
			t.Fatalf("recovery wait status=%d body=%s", waited.Code, waited.Body.String())
		}
		return submission
	}
	first := run()
	second := run()
	generator.mu.Lock()
	executions := generator.executions
	generator.mu.Unlock()
	if second.Operation == first.Operation || executions != 2 {
		t.Fatalf("second operation=%s first=%s executions=%d", second.Operation, first.Operation, executions)
	}
}
