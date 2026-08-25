package server

import (
	"context"
	"encoding/json"
	"net/http"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/operation"
	"overgo/internal/operatoraction"
	"overgo/internal/recipe"
	"overgo/internal/runrecord"
	"overgo/internal/workflowruntime"
)

const operationEvidenceSecret = "transcript-content-must-stay-private"

func TestOperationEvidenceProjection(t *testing.T) {
	fixture, execution, status := operationEvidenceFixture(t)
	defer fixture.store.Close()
	response := serveTestRequest(
		fixture.handler, http.MethodGet, "/operations/evidence?id="+execution.Operation.String(), "",
	)
	if response.Code != http.StatusOK {
		t.Fatalf("projection status=%d body=%s", response.Code, response.Body.String())
	}
	var projection OperationEvidenceProjection
	if err := json.Unmarshal(response.Body.Bytes(), &projection); err != nil {
		t.Fatal(err)
	}
	if projection.ID != execution.Operation || projection.Operation == nil || projection.Operation.Run == nil ||
		*projection.Operation.Run != *status.Run || len(projection.Stages) != len(fixture.definition.Nodes) ||
		len(projection.Serving) != 1 || len(projection.Interactions) != 1 || len(projection.Decisions) != 1 ||
		projection.Summary.Stages.Completed != len(fixture.definition.Nodes) || projection.Summary.ServingAttempts != 1 {
		t.Fatalf("operation projection differs: %+v", projection)
	}
}

func TestOperationSummary(t *testing.T) {
	projection := OperationEvidenceProjection{
		Operation: &operation.Status{State: operation.StateCompleted, Attempts: []artifact.ID{{}, {}}},
		Stages: []OperationEvidenceDocument[runrecord.StageReceipt]{
			{Value: runrecord.StageReceipt{Attempt: 2, State: runrecord.StageCompleted}},
			{Value: runrecord.StageReceipt{Attempt: 3, State: runrecord.StageFailed}},
		},
		Serving: []OperationEvidenceDocument[runrecord.ServingObservation]{
			{Value: runrecord.ServingObservation{
				MeasuredNS: 5, Usage: runrecord.ServingUsage{InputTokens: 2, OutputTokens: 3},
				Resources: runrecord.ServingResources{PeakHostBytes: 7, PeakDeviceBytes: 11, HostToDeviceBytes: 13},
			}},
			{Value: runrecord.ServingObservation{
				MeasuredNS: 17, Usage: runrecord.ServingUsage{InputTokens: 19, OutputTokens: 23},
				Resources: runrecord.ServingResources{PeakHostBytes: 29, PeakDeviceBytes: 5, HostToDeviceBytes: 31},
			}},
		},
		Interactions: make([]OperationEvidenceDocument[runrecord.Interaction], 2),
		Decisions:    make([]OperationEvidenceDocument[runrecord.HumanDecision], 1),
	}
	summary, err := summarizeOperationEvidence(projection)
	if err != nil || summary.State != operation.StateCompleted || summary.OperationAttempts != 2 ||
		summary.StageAttempts != 5 || summary.Stages.Completed != 1 || summary.Stages.Failed != 1 ||
		summary.ServingAttempts != 2 || summary.Interactions != 2 || summary.Decisions != 1 ||
		summary.TotalMeasuredNS != 22 || summary.Usage.InputTokens != 21 || summary.Usage.OutputTokens != 26 ||
		summary.Resources.PeakHostBytes != 29 || summary.Resources.PeakDeviceBytes != 11 ||
		summary.Resources.HostToDeviceBytes != 44 {
		t.Fatalf("summary=(%+v, %v)", summary, err)
	}
}

func TestOperationProjectionBounds(t *testing.T) {
	fixture, execution, status := operationEvidenceFixture(t)
	defer fixture.store.Close()
	publishOperationInteraction(t, fixture, execution.Operation, *status.Run, "resp_operation_second", "second private payload")
	projection, err := fixture.handler.operationEvidenceSnapshot(context.Background(), execution.Operation, 1)
	if err != nil || len(projection.Interactions) != 1 || !projection.Truncated {
		t.Fatalf("bounded projection=(%+v, %v)", projection, err)
	}
}

func TestTracePayloadRedaction(t *testing.T) {
	fixture, execution, _ := operationEvidenceFixture(t)
	defer fixture.store.Close()
	projection, err := fixture.handler.operationEvidenceSnapshot(context.Background(), execution.Operation, fixture.handler.config.MaxStoredResponses)
	if err != nil {
		t.Fatal(err)
	}
	encoded, err := json.Marshal(projection)
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(string(encoded), operationEvidenceSecret) || !strings.Contains(string(encoded), `"trace"`) {
		t.Fatalf("trace projection leaked payload or omitted trace identity: %s", encoded)
	}
}

func operationEvidenceFixture(
	t *testing.T,
) (automationServerFixture, workflowruntime.AutomationExecution, operation.Status) {
	t.Helper()
	fixture := newAutomationServerFixture(t)
	definitionID := publishAutomationFromAPI(t, fixture)
	activation := serveTestRequest(fixture.handler, http.MethodPost, "/automations/activate", marshalAutomationJSON(t, map[string]any{
		"definition": definitionID,
	}))
	if activation.Code != http.StatusOK {
		fixture.store.Close()
		t.Fatalf("activation status=%d body=%s", activation.Code, activation.Body.String())
	}
	run := serveTestRequest(fixture.handler, http.MethodPost, "/automations/run", marshalAutomationJSON(t, map[string]any{
		"name": "daily-report", "key": "operation-evidence", "inputs": map[string]any{"tokens": "hello"},
	}))
	var execution workflowruntime.AutomationExecution
	if run.Code != http.StatusAccepted || json.Unmarshal(run.Body.Bytes(), &execution) != nil {
		fixture.store.Close()
		t.Fatalf("run status=%d body=%s", run.Code, run.Body.String())
	}
	status, err := fixture.handler.operations.Wait(context.Background(), execution.Operation)
	if err != nil || status.State != operation.StateCompleted || status.Run == nil {
		fixture.store.Close()
		t.Fatalf("operation=(%+v, %v)", status, err)
	}
	model := fixture.definition.Dependencies[0].Artifact
	if _, err := runrecord.PublishServingObservation(context.Background(), fixture.store, runrecord.ServingObservation{
		Model: model, Recipe: fixture.definition.ID,
		Environment: fixture.handler.environment.ID,
		Operation:   execution.Operation, Run: *status.Run, Task: recipe.TaskGeneration,
		Outcome: runrecord.OutcomeSucceeded, StartedUnixNS: 1, MeasuredNS: 2,
		Usage:     runrecord.ServingUsage{InputTokens: 3, OutputTokens: 5},
		Resources: runrecord.ServingResources{PeakHostBytes: 7, PeakDeviceBytes: 11},
	}); err != nil {
		fixture.store.Close()
		t.Fatal(err)
	}
	publishOperationInteraction(t, fixture, execution.Operation, *status.Run, "resp_operation_evidence", operationEvidenceSecret)
	action := operatoraction.Action{Code: "continue", Summary: "Continue operation", Argv: []string{"overgo", "continue"}}
	request, err := operatoraction.NewApprovalRequest(execution.Operation, fixture.definition.ID, action, artifact.ID{})
	if err != nil {
		fixture.store.Close()
		t.Fatal(err)
	}
	decision, err := runrecord.NewHumanDecision(request, operatoraction.AnswerGrant)
	if err == nil {
		err = runrecord.PublishHumanDecision(context.Background(), fixture.store, request, decision)
	}
	if err != nil {
		fixture.store.Close()
		t.Fatal(err)
	}
	return fixture, execution, status
}

func publishOperationInteraction(
	t *testing.T,
	fixture automationServerFixture,
	operationID artifact.ID,
	run artifact.ID,
	response, privatePayload string,
) {
	t.Helper()
	if _, err := runrecord.PublishInteraction(context.Background(), fixture.store, runrecord.Interaction{
		Response: response, Recipe: fixture.definition.ID, Model: fixture.definition.Dependencies[0].Artifact,
		Node: fixture.definition.Nodes[0].ID, Operation: operationID, Run: run,
	}, []runrecord.InteractionMessage{{Role: "user", Content: privatePayload}, {Role: "assistant", Content: "answer"}}); err != nil {
		t.Fatal(err)
	}
}
