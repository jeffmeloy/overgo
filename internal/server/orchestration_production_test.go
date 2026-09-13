package server

import (
	"encoding/json"
	"net/http"
	"strings"
	"testing"
	"time"

	"overgo/internal/artifact"
	"overgo/internal/operation"
	"overgo/internal/recipe"
	"overgo/internal/workflowruntime"
)

var orchestrationValidationAnchor = time.Date(2026, time.August, 24, 8, 0, 0, 0, time.UTC)

type orchestrationValidationClock struct{ now time.Time }

func (clock orchestrationValidationClock) Now() time.Time { return clock.now }

func TestOrchestrationProductionVerticals(t *testing.T) {
	fixture := newAutomationServerFixture(t)
	defer fixture.store.Close()
	manualDefinition := publishAutomationFromAPI(t, fixture)
	activateAutomationFromAPI(t, fixture.handler, manualDefinition)
	manual := runAutomationFromAPI(t, fixture.handler, "/automations/run", map[string]any{
		"name": "daily-report", "key": "production-manual", "inputs": map[string]any{"tokens": "manual"},
	})
	manualStatus := waitAutomationOperation(t, fixture.handler, manual.Operation)

	scheduledInput := AutomationDefinitionInput{
		Name: "scheduled-report", Recipe: fixture.definition.ID,
		Trigger: recipe.AutomationTriggerPolicy{
			Kind: recipe.AutomationTriggerSchedule, Schedule: time.Hour.String(),
			AnchorUnixNano: orchestrationValidationAnchor.UnixNano(), Missed: recipe.AutomationMissedLatest,
		},
		Delivery: recipe.AutomationDeliveryPolicy{Kind: recipe.AutomationDeliveryArtifact},
	}
	publish := serveTestRequest(fixture.handler, http.MethodPost, "/automations/definitions", marshalAutomationJSON(t, scheduledInput))
	var scheduledDefinition struct {
		ID artifact.ID `json:"id"`
	}
	if err := json.Unmarshal(publish.Body.Bytes(), &scheduledDefinition); err != nil || publish.Code != http.StatusCreated {
		t.Fatalf("scheduled definition status=%d body=%s err=%v", publish.Code, publish.Body.String(), err)
	}
	activateAutomationFromAPI(t, fixture.handler, scheduledDefinition.ID)
	fixture.handler.generator.(*automationWorkspaceGenerator).clock = orchestrationValidationClock{
		now: orchestrationValidationAnchor.Add(time.Hour),
	}
	scheduled := runAutomationFromAPI(t, fixture.handler, "/automations/schedule", map[string]any{
		"name": "scheduled-report", "inputs": map[string]any{"tokens": "scheduled"},
	})
	if !scheduled.Claim.Valid() {
		t.Fatal("scheduled production run omitted its durable claim")
	}
	scheduledStatus := waitAutomationOperation(t, fixture.handler, scheduled.Operation)

	history := serveTestRequest(fixture.handler, http.MethodGet, "/automations/history", "")
	manualEvidence := serveTestRequest(fixture.handler, http.MethodGet, "/operations/evidence?id="+manual.Operation.String(), "")
	scheduledEvidence := serveTestRequest(fixture.handler, http.MethodGet, "/operations/evidence?id="+scheduled.Operation.String(), "")
	gui := serveTestRequest(fixture.handler, http.MethodGet, "/mod/automations.js", "")
	if history.Code != http.StatusOK || !strings.Contains(history.Body.String(), manualStatus.Run.String()) ||
		!strings.Contains(history.Body.String(), scheduledStatus.Run.String()) || manualEvidence.Code != http.StatusOK ||
		scheduledEvidence.Code != http.StatusOK || !strings.Contains(manualEvidence.Body.String(), `"completed":1`) ||
		!strings.Contains(scheduledEvidence.Body.String(), `"completed":1`) || gui.Code != http.StatusOK ||
		!strings.Contains(gui.Body.String(), "/automations/stream") || !strings.Contains(gui.Body.String(), "overgo.decideOperation(") {
		t.Fatalf("production projections history=(%d %s) manual=(%d %s) scheduled=(%d %s) gui=%d",
			history.Code, history.Body.String(), manualEvidence.Code, manualEvidence.Body.String(),
			scheduledEvidence.Code, scheduledEvidence.Body.String(), gui.Code)
	}
	_, fired, err := fixture.handler.generator.(*automationWorkspaceGenerator).ScheduleAutomation(
		t.Context(), fixture.handler.operations, AutomationExecutionInput{
			Name: "scheduled-report", Inputs: map[string]json.RawMessage{"tokens": json.RawMessage(`"scheduled"`)},
		},
	)
	if err != nil || fired {
		t.Fatalf("duplicate schedule=(%t, %v)", fired, err)
	}
}

func activateAutomationFromAPI(t *testing.T, handler *Handler, definition artifact.ID) {
	t.Helper()
	response := serveTestRequest(handler, http.MethodPost, "/automations/activate", marshalAutomationJSON(t, map[string]any{
		"definition": definition,
	}))
	if response.Code != http.StatusOK {
		t.Fatalf("automation activation status=%d body=%s", response.Code, response.Body.String())
	}
}

func runAutomationFromAPI(t *testing.T, handler *Handler, path string, request map[string]any) workflowruntime.AutomationExecution {
	t.Helper()
	response := serveTestRequest(handler, http.MethodPost, path, marshalAutomationJSON(t, request))
	if response.Code != http.StatusAccepted {
		t.Fatalf("automation run %s status=%d body=%s", path, response.Code, response.Body.String())
	}
	var execution workflowruntime.AutomationExecution
	if path == "/automations/schedule" {
		var result struct {
			Execution workflowruntime.AutomationExecution `json:"execution"`
			Fired     bool                                `json:"fired"`
		}
		if err := json.Unmarshal(response.Body.Bytes(), &result); err != nil || !result.Fired {
			t.Fatalf("scheduled execution=(%+v, %v)", result, err)
		}
		execution = result.Execution
	} else if err := json.Unmarshal(response.Body.Bytes(), &execution); err != nil {
		t.Fatal(err)
	}
	if !execution.Operation.Valid() {
		t.Fatal("automation execution omitted operation authority")
	}
	return execution
}

func waitAutomationOperation(t *testing.T, handler *Handler, id artifact.ID) operation.Status {
	t.Helper()
	status, err := handler.operations.Wait(t.Context(), id)
	if err != nil || status.State != operation.StateCompleted || status.Run == nil || len(status.Outputs) == 0 {
		t.Fatalf("automation operation=(%+v, %v)", status, err)
	}
	return status
}
