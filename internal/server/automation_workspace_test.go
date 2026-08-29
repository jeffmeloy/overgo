package server

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/modelrecipe"
	"overgo/internal/operation"
	"overgo/internal/overgodb"
	"overgo/internal/recipe"
	"overgo/internal/workflowrecipe"
	"overgo/internal/workflowruntime"
)

type automationWorkspaceGenerator struct {
	*fakeGenerator
	*AutomationWorkspace
}

type automationServerFixture struct {
	handler    *Handler
	store      *overgodb.Store
	definition recipe.Definition
}

func TestAutomationWorkspaceVertical(t *testing.T) {
	fixture := newAutomationServerFixture(t)
	defer fixture.store.Close()
	definition := publishAutomationFromAPI(t, fixture)
	activate := serveTestRequest(fixture.handler, http.MethodPost, "/automations/activate", marshalAutomationJSON(t, map[string]any{
		"definition": definition,
	}))
	if activate.Code != http.StatusOK {
		t.Fatalf("activate status=%d body=%s", activate.Code, activate.Body.String())
	}
	inventory := serveTestRequest(fixture.handler, http.MethodGet, "/automations", "")
	if inventory.Code != http.StatusOK || !strings.Contains(inventory.Body.String(), "daily-report") ||
		!strings.Contains(inventory.Body.String(), `"plan"`) {
		t.Fatalf("inventory status=%d body=%s", inventory.Code, inventory.Body.String())
	}
	run := serveTestRequest(fixture.handler, http.MethodPost, "/automations/run", marshalAutomationJSON(t, map[string]any{
		"name": "daily-report", "key": "vertical", "inputs": map[string]any{"tokens": "hello"},
	}))
	if run.Code != http.StatusAccepted {
		t.Fatalf("run status=%d body=%s", run.Code, run.Body.String())
	}
	var execution workflowruntime.AutomationExecution
	if err := json.Unmarshal(run.Body.Bytes(), &execution); err != nil || !execution.Operation.Valid() {
		t.Fatalf("execution = (%+v, %v)", execution, err)
	}
	status, err := fixture.handler.operations.Wait(t.Context(), execution.Operation)
	if err != nil || status.State != operation.StateCompleted || len(status.Outputs) != 1 {
		t.Fatalf("automation operation = (%+v, %v)", status, err)
	}
	history := serveTestRequest(fixture.handler, http.MethodGet, "/automations/history", "")
	if history.Code != http.StatusOK || !strings.Contains(history.Body.String(), status.Run.String()) {
		t.Fatalf("history status=%d body=%s", history.Code, history.Body.String())
	}
	content := serveTestRequest(
		fixture.handler, http.MethodGet, "/artifacts/content?id="+status.Outputs[0].String(), "",
	)
	if content.Code != http.StatusOK {
		t.Fatalf("output status=%d body=%s", content.Code, content.Body.String())
	}
}

func TestAutomationWorkspaceSSE(t *testing.T) {
	fixture := newAutomationServerFixture(t)
	defer fixture.store.Close()
	ctx, cancel := context.WithCancel(t.Context())
	recorder := &countingRecorder{ResponseRecorder: httptest.NewRecorder(), flushes: make(chan struct{}, 4)}
	done := make(chan struct{})
	go func() {
		fixture.handler.ServeHTTP(recorder, httptest.NewRequest(http.MethodGet, "/automations/stream", nil).WithContext(ctx))
		close(done)
	}()
	for range 2 {
		<-recorder.flushes
	}
	cancel()
	<-done
	if recorder.Header().Get("Content-Type") != "text/event-stream" ||
		!strings.Contains(recorder.Body.String(), "event: automation.inventory") ||
		!strings.Contains(recorder.Body.String(), "event: operation.snapshot") {
		t.Fatalf("automation SSE headers=%v body=%s", recorder.Header(), recorder.Body.String())
	}
}

func TestAutomationWorkspaceRefusal(t *testing.T) {
	fixture := newAutomationServerFixture(t)
	defer fixture.store.Close()
	publishAutomationFromAPI(t, fixture)
	invalid := serveTestRequest(fixture.handler, http.MethodPost, "/automations/activate", `{"definition":"recipe:sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"}`)
	if invalid.Code != http.StatusUnprocessableEntity || !strings.Contains(invalid.Body.String(), "automation_refused") {
		t.Fatalf("activation refusal status=%d body=%s", invalid.Code, invalid.Body.String())
	}
	unavailable := serveTestRequest(newTestHandler(t, &fakeGenerator{}), http.MethodGet, "/automations", "")
	if unavailable.Code != http.StatusNotImplemented {
		t.Fatalf("unavailable workspace status=%d body=%s", unavailable.Code, unavailable.Body.String())
	}
}

func TestAutomationWorkspaceNoDirectExecutor(t *testing.T) {
	fixture := newAutomationServerFixture(t)
	defer fixture.store.Close()
	routes := serveTestRequest(fixture.handler, http.MethodGet, "/mod/automations.js", "").Body.String()
	for _, expected := range []string{
		"/automations/definitions", "/automations/activate", "/automations/run", "/automations/schedule",
		"/automations/history", "/automations/stream", "/operations/cancel", "/operations/decision", "schemaForm",
	} {
		if !strings.Contains(routes, expected) {
			t.Errorf("automation GUI lacks %q", expected)
		}
	}
	source := readServerSource(t, "automation_routes.go")
	for _, forbidden := range []string{"NewForProgram", "AutomationRuntime{", "NewExecutor", "ExecuteProgram"} {
		if strings.Contains(source, forbidden) {
			t.Errorf("automation routes construct runtime directly with %q", forbidden)
		}
	}
}

func newAutomationServerFixture(t *testing.T) automationServerFixture {
	t.Helper()
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	model := commitAutomationServerBlob(t, store, artifact.KindModel, "server-automation-model")
	definition, err := recipe.NewDefinitionWithDependencies(
		recipe.TaskGeneration, []recipe.Dependency{{Role: recipe.DependencyModel, Artifact: model}},
		[]recipe.Node{{
			ID: "generate", Module: workflowrecipe.ModuleGenerate,
			Placement: recipe.PlacementHost, Session: recipe.SessionCapacity,
		}}, nil,
		[]recipe.Input{{Name: "tokens", Data: recipe.DataTokens, Target: recipe.Endpoint{Node: "generate", Port: "tokens"}}},
		[]recipe.Output{{Name: "tokens", Data: recipe.DataTokens, Source: recipe.Endpoint{Node: "generate", Port: "tokens"}}},
	)
	if err != nil {
		store.Close()
		t.Fatal(err)
	}
	content, err := definition.ArtifactContent()
	if err == nil {
		_, err = artifact.CommitBatch(t.Context(), store, artifact.Batch{
			Key: "automation/server/recipe", Contents: []artifact.Content{content},
		})
	}
	if err == nil {
		err = modelrecipe.EnsureRuntimePolicy(t.Context(), store, definition)
	}
	if err != nil {
		store.Close()
		t.Fatal(err)
	}
	adapter := workflowruntime.AdapterFunc(func(_ context.Context, request workflowruntime.StepRequest) (map[recipe.PortName]workflowruntime.Value, error) {
		input := request.Inputs["tokens"].Items[0]
		encoded, marshalErr := json.Marshal(input.Value)
		if marshalErr != nil {
			return nil, marshalErr
		}
		id, identifyErr := artifact.IdentifyBytes(artifact.KindOutput, encoded)
		if identifyErr != nil {
			return nil, identifyErr
		}
		output := artifact.Content{Descriptor: artifact.Descriptor{
			ID: id, Size: uint64(len(encoded)), MediaType: "application/json",
		}, Data: encoded}
		input.Content, input.Artifact = &output, output.Descriptor
		return map[recipe.PortName]workflowruntime.Value{
			"tokens": {Kind: recipe.DataTokens, Items: []workflowruntime.Datum{input}},
		}, nil
	})
	workspace, err := (AutomationWorkspaceConfig{
		Store: store, Catalog: workflowrecipe.Catalog(),
		Adapters: map[recipe.ModuleID]workflowruntime.Adapter{workflowrecipe.ModuleGenerate: adapter},
		Limit:    len(definition.Nodes) + 16,
	}).Open()
	if err != nil {
		store.Close()
		t.Fatal(err)
	}
	generator := &automationWorkspaceGenerator{fakeGenerator: &fakeGenerator{}, AutomationWorkspace: workspace}
	handler := newTestHandlerForRepository(t, store, generator)
	return automationServerFixture{handler: handler, store: store, definition: definition}
}

func publishAutomationFromAPI(t *testing.T, fixture automationServerFixture) artifact.ID {
	t.Helper()
	request := AutomationDefinitionInput{
		Name: "daily-report", Recipe: fixture.definition.ID,
		Trigger:  recipe.AutomationTriggerPolicy{Kind: recipe.AutomationTriggerManual},
		Delivery: recipe.AutomationDeliveryPolicy{Kind: recipe.AutomationDeliveryArtifact},
	}
	response := serveTestRequest(fixture.handler, http.MethodPost, "/automations/definitions", marshalAutomationJSON(t, request))
	if response.Code != http.StatusCreated {
		t.Fatalf("publish status=%d body=%s", response.Code, response.Body.String())
	}
	var result struct {
		ID artifact.ID `json:"id"`
	}
	if err := json.Unmarshal(response.Body.Bytes(), &result); err != nil || !result.ID.Valid() {
		t.Fatalf("published definition = (%s, %v)", result.ID, err)
	}
	return result.ID
}

func marshalAutomationJSON(t *testing.T, value any) string {
	t.Helper()
	encoded, err := json.Marshal(value)
	if err != nil {
		t.Fatal(err)
	}
	return string(encoded)
}

func commitAutomationServerBlob(t *testing.T, store artifact.Repository, kind artifact.Kind, label string) artifact.ID {
	t.Helper()
	data := []byte(label)
	id, err := artifact.IdentifyBytes(kind, data)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := artifact.CommitBatch(t.Context(), store, artifact.Batch{
		Key: "automation/server/blob/" + label,
		Contents: []artifact.Content{{Descriptor: artifact.Descriptor{
			ID: id, Size: uint64(len(data)), MediaType: "application/octet-stream",
		}, Data: data}},
	}); err != nil {
		t.Fatal(err)
	}
	return id
}

func readServerSource(t *testing.T, name string) string {
	t.Helper()
	data, err := os.ReadFile(name)
	if err != nil {
		t.Fatal(err)
	}
	return string(data)
}
