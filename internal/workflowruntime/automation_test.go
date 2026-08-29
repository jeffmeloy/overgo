package workflowruntime

import (
	"context"
	"errors"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"overgo/internal/artifact"
	"overgo/internal/modelrecipe"
	"overgo/internal/operation"
	"overgo/internal/overgodb"
	"overgo/internal/recipe"
	"overgo/internal/runrecord"
	"overgo/internal/workflowrecipe"
)

type automationRuntimeFixture struct {
	store      *overgodb.Store
	root       string
	definition recipe.Definition
	name       string
	inputs     map[recipe.PortName]Value
}

func TestAutomationManualExecution(t *testing.T) {
	fixture := newAutomationRuntimeFixture(t)
	defer fixture.store.Close()
	manager, err := operation.NewManager(len(fixture.definition.Nodes) + 1)
	if err != nil {
		t.Fatal(err)
	}
	defer manager.Close()
	runtime := fixture.runtime(t, manager, generationAutomationAdapters(t, false, nil))
	execution, err := runtime.SubmitManual(t.Context(), fixture.name, "manual", fixture.inputs)
	if err != nil {
		t.Fatal(err)
	}
	status, err := manager.Wait(t.Context(), execution.Operation)
	if err != nil || status.State != operation.StateCompleted || status.Run == nil || len(status.Outputs) != 1 {
		t.Fatalf("manual automation status = (%+v, %v)", status, err)
	}
	if _, found, err := artifact.ReadContent(t.Context(), fixture.store, status.Outputs[0]); err != nil || !found {
		t.Fatalf("durable automation output = (%v, %v)", found, err)
	}
	for _, node := range fixture.definition.Nodes {
		receipt, found, err := runrecord.ResolveStageReceipt(t.Context(), fixture.store, execution.Operation, node.ID)
		if err != nil || !found || receipt.State != runrecord.StageCompleted {
			t.Fatalf("stage %s receipt = (%+v, %v, %v)", node.ID, receipt, found, err)
		}
	}
}

func TestAutomationCancellation(t *testing.T) {
	fixture := newAutomationRuntimeFixture(t)
	defer fixture.store.Close()
	manager, err := operation.NewManager(len(fixture.definition.Nodes) + 1)
	if err != nil {
		t.Fatal(err)
	}
	defer manager.Close()
	entered := make(chan struct{})
	runtime := fixture.runtime(t, manager, generationAutomationAdapters(t, false, entered))
	execution, err := runtime.SubmitManual(t.Context(), fixture.name, "cancel", fixture.inputs)
	if err != nil {
		t.Fatal(err)
	}
	<-entered
	if !manager.Cancel(execution.Operation) {
		t.Fatal("active automation was not cancelled")
	}
	status, err := manager.Wait(t.Context(), execution.Operation)
	if err != nil || status.State != operation.StateCancelled || status.Run == nil {
		t.Fatalf("cancelled automation status = (%+v, %v)", status, err)
	}
	content, found, err := artifact.ReadContent(t.Context(), fixture.store, *status.Run)
	if err != nil || !found {
		t.Fatalf("cancelled automation run = (%v, %v)", found, err)
	}
	run, err := runrecord.ParseRun(content.Data)
	if err != nil || run.Outcome != runrecord.OutcomeCancelled {
		t.Fatalf("cancelled automation run = (%+v, %v)", run, err)
	}
}

func TestAutomationRecovery(t *testing.T) {
	fixture := newAutomationRuntimeFixture(t)
	defer fixture.store.Close()
	firstManager, err := operation.NewManager(len(fixture.definition.Nodes) + 1)
	if err != nil {
		t.Fatal(err)
	}
	first := fixture.runtime(t, firstManager, generationAutomationAdapters(t, true, nil))
	execution, err := first.SubmitManual(t.Context(), fixture.name, "recover", fixture.inputs)
	if err != nil {
		t.Fatal(err)
	}
	status, err := firstManager.Wait(t.Context(), execution.Operation)
	firstManager.Close()
	if err != nil || status.State != operation.StateFailed {
		t.Fatalf("failed automation status = (%+v, %v)", status, err)
	}

	secondManager, err := operation.NewManager(len(fixture.definition.Nodes) + 1)
	if err != nil {
		t.Fatal(err)
	}
	defer secondManager.Close()
	adapters := generationAutomationAdapters(t, false, nil)
	adapters[workflowrecipe.ModuleTokenize] = AdapterFunc(func(context.Context, StepRequest) (map[recipe.PortName]Value, error) {
		return nil, errors.New("completed automation stage reran")
	})
	second := fixture.runtime(t, secondManager, adapters)
	recovered, err := second.RecoverAutomation(t.Context(), execution.Plan.ID, "recover", fixture.inputs)
	if err != nil || recovered.Operation != execution.Operation {
		t.Fatalf("automation recovery admission = (%+v, %v)", recovered, err)
	}
	status, err = secondManager.Wait(t.Context(), recovered.Operation)
	if err != nil || status.State != operation.StateCompleted {
		t.Fatalf("recovered automation status = (%+v, %v)", status, err)
	}
}

func TestWebhookAutomationDispatchAndRecovery(t *testing.T) {
	fixture := newWebhookAutomationRuntimeFixture(t)
	defer func() {
		if fixture.store != nil {
			_ = fixture.store.Close()
		}
	}()
	ctx := t.Context()
	firstManager, err := operation.NewManager(len(fixture.definition.Nodes) + 1)
	if err != nil {
		t.Fatal(err)
	}
	var tokenizeCalls atomic.Uint32
	firstAdapters := generationAutomationAdapters(t, true, nil)
	tokenize := firstAdapters[workflowrecipe.ModuleTokenize]
	firstAdapters[workflowrecipe.ModuleTokenize] = AdapterFunc(func(ctx context.Context, request StepRequest) (map[recipe.PortName]Value, error) {
		tokenizeCalls.Add(1)
		return tokenize.Execute(ctx, request)
	})
	first := fixture.runtime(t, firstManager, firstAdapters)
	plan, err := first.PrepareWebhook(ctx, fixture.name)
	if err != nil {
		firstManager.Close()
		t.Fatal(err)
	}
	payload := []byte(`{"prompt":"hello"}`)
	ledger := runrecord.WebhookDeliveryLedger{Repository: fixture.store}
	accepted, err := ledger.Publish(ctx, runrecord.WebhookDeliveryInput{
		Policy: plan.Trigger.ID, Plan: plan.ID, Source: plan.Name, IdempotencyKey: "delivery-1",
		Payload: payload, Signature: runrecord.WebhookSignatureVerified, Disposition: runrecord.WebhookDeliveryAccepted,
	})
	if err != nil {
		firstManager.Close()
		t.Fatal(err)
	}
	failed, err := first.DispatchWebhook(ctx, accepted.ID)
	if err != nil {
		firstManager.Close()
		t.Fatal(err)
	}
	status, err := firstManager.Wait(ctx, failed.Operation)
	if err != nil || status.State != operation.StateFailed || tokenizeCalls.Load() != 1 {
		firstManager.Close()
		t.Fatalf("failed webhook operation = (%+v, %v), tokenize calls = %d", status, err, tokenizeCalls.Load())
	}
	duplicate, err := ledger.Publish(ctx, runrecord.WebhookDeliveryInput{
		Policy: plan.Trigger.ID, Plan: plan.ID, Source: plan.Name, IdempotencyKey: "delivery-1",
		Payload: payload, Signature: runrecord.WebhookSignatureVerified, Disposition: runrecord.WebhookDeliveryAccepted,
	})
	if err != nil || duplicate.Disposition != runrecord.WebhookDeliveryDuplicate || duplicate.Plan != plan.ID {
		firstManager.Close()
		t.Fatalf("duplicate webhook delivery = (%+v, %v)", duplicate, err)
	}
	converged, err := first.DispatchWebhook(ctx, duplicate.ID)
	if err != nil || converged.Operation != failed.Operation || len(firstManager.List()) != 1 || tokenizeCalls.Load() != 1 {
		firstManager.Close()
		t.Fatalf("duplicate webhook convergence = (%+v, %v), operations = %d, tokenize calls = %d",
			converged, err, len(firstManager.List()), tokenizeCalls.Load())
	}
	firstManager.Close()
	if err := fixture.store.Close(); err != nil {
		t.Fatal(err)
	}
	fixture.store = nil

	reopened, err := overgodb.Open(fixture.root)
	if err != nil {
		t.Fatal(err)
	}
	fixture.store = reopened
	activateReplacementWebhook(t, fixture)
	secondManager, err := operation.NewManager(len(fixture.definition.Nodes) + 1)
	if err != nil {
		t.Fatal(err)
	}
	defer secondManager.Close()
	secondAdapters := generationAutomationAdapters(t, false, nil)
	secondAdapters[workflowrecipe.ModuleTokenize] = AdapterFunc(func(context.Context, StepRequest) (map[recipe.PortName]Value, error) {
		return nil, errors.New("completed webhook stage reran")
	})
	second := fixture.runtime(t, secondManager, secondAdapters)
	replacementPlan, err := second.PrepareWebhook(ctx, fixture.name)
	if err != nil || replacementPlan.ID == plan.ID {
		t.Fatalf("replacement webhook plan = (%s, %v), original = %s", replacementPlan.ID, err, plan.ID)
	}
	retry, err := (runrecord.WebhookDeliveryLedger{Repository: fixture.store}).Publish(ctx, runrecord.WebhookDeliveryInput{
		Policy: plan.Trigger.ID, Plan: replacementPlan.ID, Source: plan.Name, IdempotencyKey: "delivery-1",
		Payload: payload, Signature: runrecord.WebhookSignatureVerified, Disposition: runrecord.WebhookDeliveryAccepted,
	})
	if err != nil || retry.Disposition != runrecord.WebhookDeliveryDuplicate || retry.Plan != plan.ID {
		t.Fatalf("replacement-era retry = (%+v, %v)", retry, err)
	}
	recovered, err := second.DispatchWebhook(ctx, retry.ID)
	if err != nil || recovered.Operation != failed.Operation || recovered.Plan.ID != plan.ID {
		t.Fatalf("webhook recovery admission = (%+v, %v)", recovered, err)
	}
	status, err = secondManager.Wait(ctx, recovered.Operation)
	if err != nil || status.State != operation.StateCompleted || len(secondManager.List()) != 1 || tokenizeCalls.Load() != 1 {
		t.Fatalf("recovered webhook status = (%+v, %v), operations = %d, tokenize calls = %d",
			status, err, len(secondManager.List()), tokenizeCalls.Load())
	}
	causality, err := fixture.store.QueryCausality(ctx, overgodb.CausalityQuery{
		Execution: &recovered.Operation, MaxResults: 1,
	})
	if err != nil || len(causality.Links) != 1 || causality.Links[0].Root != accepted.ID ||
		causality.Links[0].Trigger != string(runrecord.TriggerWebhook) {
		t.Fatalf("webhook operation causality = (%+v, %v)", causality, err)
	}
}

type fixedAutomationClock struct{ now time.Time }

func (clock fixedAutomationClock) Now() time.Time { return clock.now }

func TestAutomationSchedule(t *testing.T) {
	anchor := time.Date(2026, time.August, 24, 8, 0, 0, 0, time.UTC)
	fixture := newScheduledAutomationRuntimeFixture(t, anchor)
	defer fixture.store.Close()
	manager, err := operation.NewManager(len(fixture.definition.Nodes) + 1)
	if err != nil {
		t.Fatal(err)
	}
	defer manager.Close()
	runtime := fixture.runtime(t, manager, generationAutomationAdapters(t, false, nil))
	clock := fixedAutomationClock{now: anchor.Add(2*time.Hour + 5*time.Minute)}
	execution, fired, err := runtime.ScheduleAutomation(t.Context(), fixture.name, clock, fixture.inputs)
	if err != nil || !fired || !execution.Claim.Valid() {
		t.Fatalf("scheduled automation admission = (%+v, %v, %v)", execution, fired, err)
	}
	status, err := manager.Wait(t.Context(), execution.Operation)
	if err != nil || status.State != operation.StateCompleted {
		t.Fatalf("scheduled automation status = (%+v, %v)", status, err)
	}
	if _, fired, err := runtime.ScheduleAutomation(t.Context(), fixture.name, clock, fixture.inputs); err != nil || fired {
		t.Fatalf("duplicate schedule fire = (%v, %v)", fired, err)
	}
}

func TestAutomationScheduleRecovery(t *testing.T) {
	anchor := time.Date(2026, time.August, 24, 8, 0, 0, 0, time.UTC)
	fixture := newScheduledAutomationRuntimeFixture(t, anchor)
	defer fixture.store.Close()
	firstManager, err := operation.NewManager(len(fixture.definition.Nodes) + 1)
	if err != nil {
		t.Fatal(err)
	}
	first := fixture.runtime(t, firstManager, generationAutomationAdapters(t, true, nil))
	execution, fired, err := first.ScheduleAutomation(
		t.Context(), fixture.name, fixedAutomationClock{now: anchor.Add(time.Hour)}, fixture.inputs,
	)
	if err != nil || !fired {
		t.Fatalf("failed schedule admission = (%v, %v)", fired, err)
	}
	status, err := firstManager.Wait(t.Context(), execution.Operation)
	firstManager.Close()
	if err != nil || status.State != operation.StateFailed {
		t.Fatalf("failed schedule status = (%+v, %v)", status, err)
	}
	secondManager, err := operation.NewManager(len(fixture.definition.Nodes) + 1)
	if err != nil {
		t.Fatal(err)
	}
	defer secondManager.Close()
	second := fixture.runtime(t, secondManager, generationAutomationAdapters(t, false, nil))
	recovered, err := second.RecoverScheduledAutomation(t.Context(), execution.Claim, fixture.inputs)
	if err != nil || recovered.Operation != execution.Operation || recovered.Claim != execution.Claim {
		t.Fatalf("scheduled recovery = (%+v, %v)", recovered, err)
	}
	status, err = secondManager.Wait(t.Context(), recovered.Operation)
	if err != nil || status.State != operation.StateCompleted {
		t.Fatalf("recovered schedule status = (%+v, %v)", status, err)
	}
}

func newAutomationRuntimeFixture(t *testing.T) automationRuntimeFixture {
	return newAutomationRuntimeFixtureWithTrigger(t, recipe.AutomationTriggerPolicy{Kind: recipe.AutomationTriggerManual})
}

func newScheduledAutomationRuntimeFixture(t *testing.T, anchor time.Time) automationRuntimeFixture {
	return newAutomationRuntimeFixtureWithTrigger(t, recipe.AutomationTriggerPolicy{
		Kind: recipe.AutomationTriggerSchedule, Schedule: time.Hour.String(),
		AnchorUnixNano: anchor.UnixNano(), Missed: recipe.AutomationMissedLatest,
	})
}

func newWebhookAutomationRuntimeFixture(t *testing.T) automationRuntimeFixture {
	fixture := newAutomationRuntimeFixture(t)
	key := commitAutomationRuntimeBlob(t, fixture.store, artifact.KindProfile, "automation-webhook-key")
	schema := commitAutomationRuntimeBlob(t, fixture.store, artifact.KindProfile, "automation-webhook-schema")
	authority := commitAutomationRuntimeBlob(t, fixture.store, artifact.KindEvidence, "automation-webhook-authority")
	trigger, err := (recipe.AutomationTriggerPolicy{Kind: recipe.AutomationTriggerWebhook, Webhook: &recipe.WebhookTriggerPolicy{
		Signature: recipe.WebhookSignaturePolicy{Algorithm: "hmac-sha256", Header: "X-Overgo-Signature", Key: key},
		Payload: recipe.WebhookPayloadPolicy{
			ContentType: "application/json", Schema: schema, MaxBytes: 4096, IdempotencyHeader: "X-Overgo-Delivery",
		},
		Rate:      recipe.WebhookRatePolicy{WindowSeconds: 60, MaxDeliveries: 10},
		Authority: authority, Workflow: fixture.definition.ID,
	}}).Identify()
	if err != nil {
		fixture.store.Close()
		t.Fatal(err)
	}
	content, err := trigger.ArtifactContent()
	if err == nil {
		var batch artifact.Batch
		batch, err = artifact.NewDocumentBatch("automation/runtime/webhook-trigger", []artifact.Content{content}, trigger.Lineage(), nil)
		if err == nil {
			_, err = artifact.CommitBatch(t.Context(), fixture.store, batch)
		}
	}
	if err != nil {
		fixture.store.Close()
		t.Fatal(err)
	}
	current, found, err := (runrecord.AutomationAuthority{Repository: fixture.store}).Resolve(t.Context(), fixture.name)
	if err != nil || !found {
		fixture.store.Close()
		t.Fatalf("manual fixture activation = (%v, %v)", found, err)
	}
	declaration := current.Definition
	declaration.TriggerPolicy = trigger.ID
	declaration, err = recipe.NewAutomationDefinition(declaration)
	if err != nil {
		fixture.store.Close()
		t.Fatal(err)
	}
	decision := commitAutomationRuntimeBlob(t, fixture.store, artifact.KindEvidence, "automation-webhook-activation")
	if _, err := (runrecord.AutomationAuthority{Repository: fixture.store}).Activate(
		t.Context(), "automation/runtime/activate-webhook", declaration, decision,
	); err != nil {
		fixture.store.Close()
		t.Fatal(err)
	}
	return fixture
}

func activateReplacementWebhook(
	t *testing.T,
	fixture automationRuntimeFixture,
) {
	t.Helper()
	current, found, err := (runrecord.AutomationAuthority{Repository: fixture.store}).Resolve(t.Context(), fixture.name)
	if err != nil || !found {
		t.Fatalf("webhook fixture activation = (%v, %v)", found, err)
	}
	decision := commitAutomationRuntimeBlob(t, fixture.store, artifact.KindEvidence, "automation-webhook-replacement-activation")
	if _, err := (runrecord.AutomationAuthority{Repository: fixture.store}).Activate(
		t.Context(), "automation/runtime/activate-webhook-replacement", current.Definition, decision,
	); err != nil {
		t.Fatal(err)
	}
}

func newAutomationRuntimeFixtureWithTrigger(
	t *testing.T,
	triggerDeclaration recipe.AutomationTriggerPolicy,
) automationRuntimeFixture {
	t.Helper()
	ctx := t.Context()
	root := t.TempDir()
	store, err := overgodb.Open(root)
	if err != nil {
		t.Fatal(err)
	}
	model := commitAutomationRuntimeBlob(t, store, artifact.KindModel, "automation-runtime-model")
	nodes := []recipe.Node{
		{ID: "tokenize", Module: workflowrecipe.ModuleTokenize, Placement: recipe.PlacementHost, Session: recipe.SessionCapacity},
		{ID: "generate", Module: workflowrecipe.ModuleGenerate, Placement: recipe.PlacementHost, Session: recipe.SessionCapacity},
		{ID: "detokenize", Module: workflowrecipe.ModuleDetokenize, Placement: recipe.PlacementHost, Session: recipe.SessionCapacity},
	}
	definition, err := recipe.NewDefinitionWithDependencies(
		recipe.TaskGeneration, []recipe.Dependency{{Role: recipe.DependencyModel, Artifact: model}}, nodes,
		[]recipe.Edge{
			{From: recipe.Endpoint{Node: "tokenize", Port: "tokens"}, To: recipe.Endpoint{Node: "generate", Port: "tokens"}},
			{From: recipe.Endpoint{Node: "generate", Port: "tokens"}, To: recipe.Endpoint{Node: "detokenize", Port: "tokens"}},
		},
		[]recipe.Input{{Name: "prompt", Data: recipe.DataText, Target: recipe.Endpoint{Node: "tokenize", Port: "text"}}},
		[]recipe.Output{{Name: "text", Data: recipe.DataText, Source: recipe.Endpoint{Node: "detokenize", Port: "text"}}},
	)
	if err != nil {
		store.Close()
		t.Fatal(err)
	}
	trigger, err := triggerDeclaration.Identify()
	if err != nil {
		store.Close()
		t.Fatal(err)
	}
	delivery, err := (recipe.AutomationDeliveryPolicy{Kind: recipe.AutomationDeliveryArtifact}).Identify()
	if err != nil {
		store.Close()
		t.Fatal(err)
	}
	definitionContent, _ := definition.ArtifactContent()
	triggerContent, _ := trigger.ArtifactContent()
	deliveryContent, _ := delivery.ArtifactContent()
	batch, err := artifact.NewDocumentBatch(
		"automation/runtime/fixture", []artifact.Content{definitionContent, triggerContent, deliveryContent}, nil, nil,
	)
	if err == nil {
		_, err = artifact.CommitBatch(ctx, store, batch)
	}
	if err != nil {
		store.Close()
		t.Fatal(err)
	}
	if err := modelrecipe.EnsureRuntimePolicy(ctx, store, definition); err != nil {
		store.Close()
		t.Fatal(err)
	}
	automation, err := recipe.NewAutomationDefinition(recipe.AutomationDefinition{
		Name: "runtime-report", Recipe: definition.ID, TriggerPolicy: trigger.ID, DeliveryPolicy: delivery.ID,
	})
	if err != nil {
		store.Close()
		t.Fatal(err)
	}
	authority := commitAutomationRuntimeBlob(t, store, artifact.KindEvidence, "automation-runtime-authority")
	if _, err := (runrecord.AutomationAuthority{Repository: store}).Activate(
		ctx, "automation/runtime/activate", automation, authority,
	); err != nil {
		store.Close()
		t.Fatal(err)
	}
	prompt := fixtureContent(t, artifact.KindFile, "hello")
	return automationRuntimeFixture{
		store: store, root: root, definition: definition, name: automation.Name,
		inputs: map[recipe.PortName]Value{
			"prompt": {Kind: recipe.DataText, Items: []Datum{{Content: &prompt, Value: "hello"}}},
		},
	}
}

func (fixture automationRuntimeFixture) runtime(
	t *testing.T,
	manager *operation.Manager,
	adapters map[recipe.ModuleID]Adapter,
) AutomationRuntime {
	t.Helper()
	return AutomationRuntime{
		Store: fixture.store, Operations: manager, Catalog: workflowrecipe.Catalog(), Adapters: adapters,
	}
}

func generationAutomationAdapters(t *testing.T, fail bool, entered chan struct{}) map[recipe.ModuleID]Adapter {
	t.Helper()
	copyPort := func(port recipe.PortName, kind recipe.DataKind) AdapterFunc {
		return func(_ context.Context, request StepRequest) (map[recipe.PortName]Value, error) {
			for _, input := range request.Inputs {
				return map[recipe.PortName]Value{port: {Kind: kind, Items: input.Items}}, nil
			}
			return nil, errors.New("automation input is absent")
		}
	}
	generate := Adapter(copyPort("tokens", recipe.DataTokens))
	if fail {
		generate = AdapterFunc(func(context.Context, StepRequest) (map[recipe.PortName]Value, error) {
			return nil, errors.New("automation fixture interruption")
		})
	} else if entered != nil {
		generate = AdapterFunc(func(ctx context.Context, _ StepRequest) (map[recipe.PortName]Value, error) {
			close(entered)
			<-ctx.Done()
			return nil, ctx.Err()
		})
	}
	return map[recipe.ModuleID]Adapter{
		workflowrecipe.ModuleTokenize: copyPort("tokens", recipe.DataTokens),
		workflowrecipe.ModuleGenerate: generate,
		workflowrecipe.ModuleDetokenize: AdapterFunc(func(_ context.Context, request StepRequest) (map[recipe.PortName]Value, error) {
			item := request.Inputs["tokens"].Items[0]
			value := strings.ToUpper(item.Value.(string))
			content := fixtureContent(t, artifact.KindOutput, value)
			item.Content, item.Artifact = &content, content.Descriptor
			return map[recipe.PortName]Value{"text": {Kind: recipe.DataText, Items: []Datum{item}}}, nil
		}),
	}
}

func commitAutomationRuntimeBlob(t *testing.T, store artifact.Repository, kind artifact.Kind, label string) artifact.ID {
	t.Helper()
	data := []byte(label)
	id, err := artifact.IdentifyBytes(kind, data)
	if err != nil {
		t.Fatal(err)
	}
	content := artifact.Content{Descriptor: artifact.Descriptor{
		ID: id, Size: uint64(len(data)), MediaType: "application/octet-stream",
	}, Data: data}
	if _, err := artifact.CommitBatch(t.Context(), store, artifact.Batch{
		Key: "automation/runtime/blob/" + label, Contents: []artifact.Content{content},
	}); err != nil {
		t.Fatal(err)
	}
	return id
}
