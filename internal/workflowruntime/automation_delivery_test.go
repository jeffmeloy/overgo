package workflowruntime

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"

	"overgo/internal/agenttool"
	"overgo/internal/artifact"
	"overgo/internal/operation"
	"overgo/internal/operatoraction"
	"overgo/internal/overgodb"
	"overgo/internal/recipe"
	"overgo/internal/runrecord"
	"overgo/internal/workflowcontract"
)

type deliveryAutomationFixture struct {
	automationRuntimeFixture
	executor *agenttool.Executor
	manual   agenttool.Manual
	calls    *atomic.Uint32
	close    func()
}

func TestAutomationDelivery(t *testing.T) {
	fixture := newDeliveryAutomationFixture(t)
	defer fixture.close()
	manager := newAutomationOperationManager(t, fixture.definition)
	defer manager.Close()
	runtime := fixture.deliveryRuntime(t, manager)
	execution, status := executeApprovedDelivery(t, runtime, manager, fixture, "delivery")
	if status.State != operation.StateCompleted || fixture.calls.Load() != 1 {
		t.Fatalf("delivery completion = (%+v, calls=%d)", status, fixture.calls.Load())
	}
	idempotency := deliveryIdempotency(t, execution, fixture.manual.ID, "ops://daily", status.Outputs)
	attempt, found, err := (runrecord.AutomationDeliveryAuthority{Repository: fixture.store}).Current(
		context.Background(), idempotency,
	)
	if err != nil || !found || attempt.State != runrecord.AutomationDeliverySucceeded || !attempt.Result.Valid() {
		t.Fatalf("delivery evidence = (%+v, %v, %v)", attempt, found, err)
	}
}

func TestAutomationDeliveryApproval(t *testing.T) {
	fixture := newDeliveryAutomationFixture(t)
	defer fixture.close()
	manager := newAutomationOperationManager(t, fixture.definition)
	defer manager.Close()
	runtime := fixture.deliveryRuntime(t, manager)
	execution, err := runtime.SubmitManualTo(
		context.Background(), fixture.name, "approval", "ops://daily", fixture.inputs,
	)
	if err != nil {
		t.Fatal(err)
	}
	status, err := manager.Wait(context.Background(), execution.Operation)
	if err != nil || status.State != operation.StateBlocked || status.Recovery == nil || fixture.calls.Load() != 0 {
		t.Fatalf("unapproved delivery = (%+v, calls=%d, err=%v)", status, fixture.calls.Load(), err)
	}
	approveAutomationDelivery(t, fixture.store, manager, status)
	status, err = manager.Wait(context.Background(), execution.Operation)
	if err != nil || status.State != operation.StateCompleted || fixture.calls.Load() != 1 {
		t.Fatalf("approved delivery = (%+v, calls=%d, err=%v)", status, fixture.calls.Load(), err)
	}
}

func TestAutomationDeliveryRefusesDestination(t *testing.T) {
	fixture := newDeliveryAutomationFixture(t)
	defer fixture.close()
	manager := newAutomationOperationManager(t, fixture.definition)
	defer manager.Close()
	runtime := fixture.deliveryRuntime(t, manager)
	if _, err := runtime.SubmitManualTo(
		context.Background(), fixture.name, "destination", "ops://unlisted", fixture.inputs,
	); err == nil || fixture.calls.Load() != 0 {
		t.Fatalf("unlisted destination = (calls=%d, err=%v)", fixture.calls.Load(), err)
	}
}

func TestAutomationDeliveryIdempotency(t *testing.T) {
	fixture := newDeliveryAutomationFixture(t)
	defer fixture.close()
	manager := newAutomationOperationManager(t, fixture.definition)
	defer manager.Close()
	runtime := fixture.deliveryRuntime(t, manager)
	first, firstStatus := executeApprovedDelivery(t, runtime, manager, fixture, "idempotent")
	second, err := runtime.SubmitManualTo(
		context.Background(), fixture.name, "idempotent", "ops://daily", fixture.inputs,
	)
	if err != nil || second.Operation != first.Operation {
		t.Fatalf("idempotent readmission = (%+v, %v)", second, err)
	}
	secondStatus, err := manager.Wait(context.Background(), second.Operation)
	if err != nil || secondStatus.State != operation.StateCompleted || fixture.calls.Load() != 1 ||
		len(firstStatus.Outputs) != len(secondStatus.Outputs) {
		t.Fatalf("idempotent delivery = (%+v, calls=%d, err=%v)", secondStatus, fixture.calls.Load(), err)
	}
}

func TestAutomationDeliveryRechecksArgvAuthority(t *testing.T) {
	ctx := context.Background()
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	manual, err := agenttool.NewManual(agenttool.Manual{
		Name: "automation.argv", Description: "Exercise live argv delivery authority.",
		Effect:    agenttool.EffectMutation,
		Transport: agenttool.Transport{Kind: agenttool.TransportArgv, Program: "overgo-deliver"},
	})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := agenttool.PublishArgvPolicy(ctx, store, []string{"overgo-deliver"}); err != nil {
		t.Fatal(err)
	}
	if _, err := agenttool.PublishManualCatalog(ctx, store, []agenttool.Manual{manual}); err != nil {
		t.Fatal(err)
	}
	if _, err := agenttool.PublishArgvPolicy(ctx, store, nil); err != nil {
		t.Fatal(err)
	}
	runtime := AutomationRuntime{Store: store, Tools: agenttool.NewOperatorExecutor()}
	plan := workflowcontract.AutomationExecutionPlan{Delivery: recipe.AutomationDeliveryPolicy{Tool: manual.ID}}
	if err := runtime.deliverAutomation(ctx, artifact.ID{}, plan, "", nil); err == nil ||
		!strings.Contains(err.Error(), "outside the committed policy") {
		t.Fatalf("tightened argv delivery result = %v", err)
	}
}

func newDeliveryAutomationFixture(t *testing.T) deliveryAutomationFixture {
	t.Helper()
	base := newAutomationRuntimeFixture(t)
	var calls atomic.Uint32
	server := httptest.NewServer(http.HandlerFunc(func(response http.ResponseWriter, request *http.Request) {
		defer request.Body.Close()
		var arguments map[string]json.RawMessage
		if err := json.NewDecoder(request.Body).Decode(&arguments); err != nil {
			http.Error(response, err.Error(), http.StatusBadRequest)
			return
		}
		calls.Add(1)
		response.Header().Set("Content-Type", "application/json")
		_, _ = response.Write([]byte(`{"delivered":true}`))
	}))
	manual, err := agenttool.NewManual(agenttool.Manual{
		Name: "automation.deliver", Description: "Deliver automation artifacts to an approved destination.",
		Effect: agenttool.EffectMutation,
		Arguments: []agenttool.Field{
			{Name: "destination", Kind: agenttool.FieldString, Required: true},
			{Name: "payload", Kind: agenttool.FieldObject, Required: true},
			{Name: "idempotency_key", Kind: agenttool.FieldString, Required: true},
		},
		Transport: agenttool.Transport{Kind: agenttool.TransportHTTP, URL: server.URL},
	})
	if err != nil {
		server.Close()
		base.store.Close()
		t.Fatal(err)
	}
	if _, err := agenttool.PublishManualCatalog(context.Background(), base.store, []agenttool.Manual{manual}); err != nil {
		server.Close()
		base.store.Close()
		t.Fatal(err)
	}
	authorization := commitAutomationRuntimeBlob(t, base.store, artifact.KindEvidence, "automation-delivery-authorization")
	delivery, err := (recipe.AutomationDeliveryPolicy{
		Kind: recipe.AutomationDeliveryTool, Tool: manual.ID, Authorization: authorization,
		Destinations:        []string{"ops://secondary", "ops://daily"},
		DestinationArgument: "destination", PayloadArgument: "payload", IdempotencyArgument: "idempotency_key",
	}).Identify()
	if err != nil {
		server.Close()
		base.store.Close()
		t.Fatal(err)
	}
	content, err := delivery.ArtifactContent()
	if err == nil {
		_, err = artifact.CommitBatch(context.Background(), base.store, artifact.Batch{
			Key: "automation/delivery/policy", Contents: []artifact.Content{content},
		})
	}
	if err != nil {
		server.Close()
		base.store.Close()
		t.Fatal(err)
	}
	authority := runrecord.AutomationAuthority{Repository: base.store}
	active, found, err := authority.Resolve(context.Background(), base.name)
	if err != nil || !found {
		server.Close()
		base.store.Close()
		t.Fatalf("active automation = (%v, %v)", found, err)
	}
	replacement, err := recipe.NewAutomationDefinition(recipe.AutomationDefinition{
		Name: base.name, Recipe: base.definition.ID,
		TriggerPolicy: active.Definition.TriggerPolicy, DeliveryPolicy: delivery.ID,
	})
	if err != nil {
		server.Close()
		base.store.Close()
		t.Fatal(err)
	}
	replacementAuthority := commitAutomationRuntimeBlob(t, base.store, artifact.KindEvidence, "automation-delivery-replacement")
	if _, err := authority.Activate(
		context.Background(), "automation/delivery/activate", replacement, replacementAuthority,
	); err != nil {
		server.Close()
		base.store.Close()
		t.Fatal(err)
	}
	return deliveryAutomationFixture{
		automationRuntimeFixture: base, executor: agenttool.NewOperatorExecutor(), manual: manual, calls: &calls,
		close: func() { server.Close(); base.store.Close() },
	}
}

func (fixture deliveryAutomationFixture) deliveryRuntime(t *testing.T, manager *operation.Manager) AutomationRuntime {
	runtime := fixture.runtime(t, manager, generationAutomationAdapters(t, false, nil))
	runtime.Tools = fixture.executor
	return runtime
}

func newAutomationOperationManager(t *testing.T, definition recipe.Definition) *operation.Manager {
	t.Helper()
	manager, err := operation.NewManager(len(definition.Nodes) + 1)
	if err != nil {
		t.Fatal(err)
	}
	return manager
}

func executeApprovedDelivery(
	t *testing.T,
	runtime AutomationRuntime,
	manager *operation.Manager,
	fixture deliveryAutomationFixture,
	key string,
) (AutomationExecution, operation.Status) {
	t.Helper()
	execution, err := runtime.SubmitManualTo(
		context.Background(), fixture.name, key, "ops://daily", fixture.inputs,
	)
	if err != nil {
		t.Fatal(err)
	}
	status, err := manager.Wait(context.Background(), execution.Operation)
	if err != nil || status.State != operation.StateBlocked {
		t.Fatalf("delivery approval block = (%+v, %v)", status, err)
	}
	approveAutomationDelivery(t, fixture.store, manager, status)
	status, err = manager.Wait(context.Background(), execution.Operation)
	if err != nil {
		t.Fatal(err)
	}
	return execution, status
}

func approveAutomationDelivery(
	t *testing.T,
	store artifact.Repository,
	manager *operation.Manager,
	status operation.Status,
) {
	t.Helper()
	action := status.Recovery.Actions[0]
	request, err := operatoraction.NewApprovalRequest(status.ID, status.Recipe, action, artifact.ID{})
	if err != nil {
		t.Fatal(err)
	}
	decision, err := runrecord.NewHumanDecision(request, operatoraction.AnswerGrant)
	if err == nil {
		err = runrecord.PublishHumanDecision(context.Background(), store, request, decision)
	}
	if err != nil {
		t.Fatal(err)
	}
	if _, err := manager.RecoverAfterDecision(context.Background(), decision); err != nil {
		t.Fatal(err)
	}
}

func deliveryIdempotency(
	t *testing.T,
	execution AutomationExecution,
	manual artifact.ID,
	destination string,
	outputs []artifact.ID,
) artifact.ID {
	t.Helper()
	id, err := artifact.JSONID(artifact.KindEvidence, struct {
		Plan        artifact.ID   `json:"plan"`
		Operation   artifact.ID   `json:"operation"`
		Tool        artifact.ID   `json:"tool"`
		Destination string        `json:"destination"`
		Outputs     []artifact.ID `json:"outputs"`
	}{Plan: execution.Plan.ID, Operation: execution.Operation, Tool: manual, Destination: destination, Outputs: outputs})
	if err != nil {
		t.Fatal(err)
	}
	return id
}
