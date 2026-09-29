package server

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/operation"
	"overgo/internal/operatoraction"
	"overgo/internal/overgodb"
	"overgo/internal/recipe"
	"overgo/internal/runrecord"
	"overgo/internal/workflowrecipe"
	"overgo/internal/workflowruntime"
)

const operationEvidenceSecret = "transcript-content-must-stay-private"

func TestOperationEvidenceProjection(t *testing.T) {
	t.Parallel()
	fixture, operationID, status := operationEvidenceFixture(t)
	defer fixture.store.Close()
	response := serveTestRequest(
		fixture.handler, http.MethodGet, "/operations/evidence?id="+operationID.String(), "",
	)
	if response.Code != http.StatusOK {
		t.Fatalf("projection status=%d body=%s", response.Code, response.Body.String())
	}
	var projection operationEvidenceProjection
	if err := json.Unmarshal(response.Body.Bytes(), &projection); err != nil {
		t.Fatal(err)
	}
	if projection.ID != operationID || projection.Operation == nil || projection.Operation.Run == nil ||
		*projection.Operation.Run != *status.Run || len(projection.Stages) != len(fixture.definition.Nodes) ||
		len(projection.Serving) != 1 || len(projection.Interactions) != 1 || len(projection.Decisions) != 1 ||
		projection.Summary.Stages.Completed != len(fixture.definition.Nodes) || projection.Summary.ServingAttempts != 1 {
		t.Fatalf("operation projection differs: %+v", projection)
	}
}

func TestOperationSummary(t *testing.T) {
	t.Parallel()
	projection := operationEvidenceProjection{
		Operation: &operation.Status{State: operation.StateCompleted, Attempts: []artifact.ID{{}, {}}},
		Stages: []operationEvidenceDocument[runrecord.StageReceipt]{
			{Value: runrecord.StageReceipt{Attempt: 2, State: runrecord.StageCompleted}},
			{Value: runrecord.StageReceipt{Attempt: 3, State: runrecord.StageFailed}},
		},
		Serving: []operationEvidenceDocument[runrecord.ServingObservation]{
			{Value: runrecord.ServingObservation{
				MeasuredNS: 5, Usage: runrecord.ServingUsage{InputTokens: 2, OutputTokens: 3},
				Resources: runrecord.ServingResources{PeakHostBytes: 7, PeakDeviceBytes: 11, HostToDeviceBytes: 13},
			}},
			{Value: runrecord.ServingObservation{
				MeasuredNS: 17, Usage: runrecord.ServingUsage{InputTokens: 19, OutputTokens: 23},
				Resources: runrecord.ServingResources{PeakHostBytes: 29, PeakDeviceBytes: 5, HostToDeviceBytes: 31},
			}},
		},
		Interactions: make([]operationEvidenceDocument[runrecord.Interaction], 2),
		Decisions:    make([]operationEvidenceDocument[runrecord.HumanDecision], 1),
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
	t.Parallel()
	fixture, operationID, status := operationEvidenceFixture(t)
	defer fixture.store.Close()
	publishOperationInteraction(t, fixture, operationID, *status.Run, "resp_operation_second", "second private payload")
	projection, err := fixture.handler.operationEvidenceSnapshot(t.Context(), operationID, 1)
	if err != nil || len(projection.Interactions) != 1 || !projection.Truncated {
		t.Fatalf("bounded projection=(%+v, %v)", projection, err)
	}
}

func TestTracePayloadRedaction(t *testing.T) {
	t.Parallel()
	fixture, operationID, _ := operationEvidenceFixture(t)
	defer fixture.store.Close()
	projection, err := fixture.handler.operationEvidenceSnapshot(t.Context(), operationID, fixture.handler.config.MaxStoredResponses)
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

// operationEvidenceServer is a store-backed handler beside the one-node
// recipe whose workflow run its operations execute.
type operationEvidenceServer struct {
	handler    *Handler
	store      *overgodb.Store
	definition recipe.Definition
}

// operationEvidenceFixture runs a one-node generation recipe through the
// handler's operation manager and the workflow runtime, then records the
// serving observation, interaction and decision the projection joins.
func operationEvidenceFixture(
	t *testing.T,
) (operationEvidenceServer, artifact.ID, operation.Status) {
	t.Helper()
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	fixture := operationEvidenceServer{store: store, handler: newTestHandlerForRepository(t, store, &fakeGenerator{})}
	model := commitTestBlob(t, store, artifact.KindModel, "operation-evidence-model")
	fixture.definition, err = recipe.NewDefinitionWithDependencies(
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
	definitionContent, err := fixture.definition.ArtifactContent()
	if err == nil {
		_, err = artifact.CommitBatch(t.Context(), store, artifact.Batch{
			Key: "operation-evidence/recipe", Contents: []artifact.Content{definitionContent},
		})
	}
	program, programErr := recipe.CompileProgram(fixture.definition, workflowrecipe.Catalog())
	if err = errors.Join(err, programErr); err != nil {
		store.Close()
		t.Fatal(err)
	}
	input := []byte(`"hello"`)
	inputID, err := artifact.IdentifyBytes(artifact.KindFile, input)
	if err != nil {
		store.Close()
		t.Fatal(err)
	}
	inputs := map[recipe.PortName]workflowruntime.Value{"tokens": workflowruntime.ArtifactValue(recipe.DataTokens, "hello", artifact.Content{
		Descriptor: artifact.Descriptor{ID: inputID, Size: uint64(len(input)), MediaType: "application/json"}, Data: input,
	})}
	// The generate stage answers its input as a stored output.
	generate := workflowruntime.AdapterFunc(func(_ context.Context, request workflowruntime.StepRequest) (map[recipe.PortName]workflowruntime.Value, error) {
		datum := request.Inputs["tokens"].Items[0]
		encoded, marshalErr := json.Marshal(datum.Value)
		if marshalErr != nil {
			return nil, marshalErr
		}
		id, identifyErr := artifact.IdentifyBytes(artifact.KindOutput, encoded)
		if identifyErr != nil {
			return nil, identifyErr
		}
		return map[recipe.PortName]workflowruntime.Value{"tokens": workflowruntime.ArtifactValue(recipe.DataTokens, datum.Value, artifact.Content{
			Descriptor: artifact.Descriptor{ID: id, Size: uint64(len(encoded)), MediaType: "application/json"}, Data: encoded,
		})}, nil
	})
	operationID, err := fixture.handler.operations.Submit(t.Context(), operation.Request{Task: recipe.TaskGeneration, Recipe: fixture.definition.ID},
		func(ctx context.Context, reporter operation.Reporter) (operation.Completion, error) {
			engine, engineErr := workflowruntime.NewForProgram(store, program)
			if engineErr == nil {
				engineErr = engine.Register(workflowrecipe.ModuleGenerate, generate)
			}
			if engineErr != nil {
				return operation.Completion{}, engineErr
			}
			result, runErr := engine.ExecuteProgram(ctx, "operation-evidence", reporter.OperationID(), reporter, program, inputs)
			return operation.Completion{Run: result.Run.ID, Outputs: result.Run.Outputs}, runErr
		})
	if err != nil {
		store.Close()
		t.Fatal(err)
	}
	status, err := fixture.handler.operations.Wait(t.Context(), operationID)
	if err != nil || status.State != operation.StateCompleted || status.Run == nil {
		fixture.store.Close()
		t.Fatalf("operation=(%+v, %v)", status, err)
	}
	if _, err := runrecord.PublishServingObservation(t.Context(), fixture.store, runrecord.ServingObservation{
		Model: model, Recipe: fixture.definition.ID,
		Environment: fixture.handler.environment.ID,
		Operation:   operationID, Run: *status.Run, Task: recipe.TaskGeneration,
		Outcome: runrecord.OutcomeSucceeded, StartedUnixNS: 1, MeasuredNS: 2,
		Usage:     runrecord.ServingUsage{InputTokens: 3, OutputTokens: 5},
		Resources: runrecord.ServingResources{PeakHostBytes: 7, PeakDeviceBytes: 11},
	}); err != nil {
		fixture.store.Close()
		t.Fatal(err)
	}
	publishOperationInteraction(t, fixture, operationID, *status.Run, "resp_operation_evidence", operationEvidenceSecret)
	action := operatoraction.Action{Code: "continue", Summary: "Continue operation", Argv: []string{"overgo", "continue"}}
	request, err := operatoraction.NewApprovalRequest(operationID, fixture.definition.ID, action, artifact.ID{})
	if err != nil {
		fixture.store.Close()
		t.Fatal(err)
	}
	decision, err := runrecord.NewHumanDecision(request, operatoraction.AnswerGrant)
	if err == nil {
		err = runrecord.PublishHumanDecision(t.Context(), fixture.store, request, decision)
	}
	if err != nil {
		fixture.store.Close()
		t.Fatal(err)
	}
	return fixture, operationID, status
}

func publishOperationInteraction(
	t *testing.T,
	fixture operationEvidenceServer,
	operationID artifact.ID,
	run artifact.ID,
	response, privatePayload string,
) {
	t.Helper()
	if _, err := runrecord.PublishInteraction(t.Context(), fixture.store, runrecord.Interaction{
		Response: response, Recipe: fixture.definition.ID, Model: fixture.definition.Dependencies[0].Artifact,
		Node: fixture.definition.Nodes[0].ID, Operation: operationID, Run: run,
	}, []runrecord.InteractionMessage{{Role: "user", Content: privatePayload}, {Role: "assistant", Content: "answer"}}, ""); err != nil {
		t.Fatal(err)
	}
}
