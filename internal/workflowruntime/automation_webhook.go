package workflowruntime

import (
	"context"
	"encoding/json"
	"errors"

	"overgo/internal/artifact"
	"overgo/internal/operation"
	"overgo/internal/recipe"
	"overgo/internal/runrecord"
	"overgo/internal/strictjson"
	"overgo/internal/workflowcontract"
)

// PrepareWebhook compiles and publishes the exact active webhook plan that an
// ingress delivery will record. Dispatch never repeats this active resolution.
func (runtime AutomationRuntime) PrepareWebhook(
	ctx context.Context,
	name string,
) (workflowcontract.AutomationExecutionPlan, error) {
	compiler := workflowcontract.AutomationCompiler{Repository: runtime.Store, Catalog: runtime.Catalog}
	plan, err := compiler.Compile(ctx, name)
	if err != nil {
		return workflowcontract.AutomationExecutionPlan{}, err
	}
	if plan.Trigger.Kind != recipe.AutomationTriggerWebhook || plan.Trigger.Webhook == nil {
		return workflowcontract.AutomationExecutionPlan{}, errors.New("workflow runtime: automation is not webhook triggerable")
	}
	if err := runtime.publishAutomationPlan(ctx, plan); err != nil {
		return workflowcontract.AutomationExecutionPlan{}, err
	}
	return plan, nil
}

// DispatchWebhook admits one durable delivery. Exact duplicates converge on
// their accepted original's operation; an explicit replay retains the original
// plan and payload but receives its own operation identity.
func (runtime AutomationRuntime) DispatchWebhook(
	ctx context.Context,
	deliveryID artifact.ID,
) (AutomationExecution, error) {
	if ctx == nil || runtime.Store == nil || runtime.Operations == nil || runtime.Catalog == nil {
		return AutomationExecution{}, errors.New("workflow runtime: automation runtime authority is absent")
	}
	ledger := runrecord.WebhookDeliveryLedger{Repository: runtime.Store}
	delivery, err := ledger.Require(ctx, deliveryID)
	if err != nil {
		return AutomationExecution{}, err
	}
	authority, operationKey, err := webhookExecutionAuthority(ctx, ledger, delivery)
	if err != nil {
		return AutomationExecution{}, err
	}
	compiler := workflowcontract.AutomationCompiler{Repository: runtime.Store, Catalog: runtime.Catalog}
	plan, err := compiler.Require(ctx, authority.Plan)
	if err != nil {
		return AutomationExecution{}, err
	}
	if plan.Trigger.Kind != recipe.AutomationTriggerWebhook || plan.Trigger.Webhook == nil ||
		plan.Trigger.ID != authority.Policy || plan.Trigger.Webhook.Workflow != plan.Recipe {
		return AutomationExecution{}, errors.New("workflow runtime: webhook delivery authority differs from recorded plan")
	}
	definition, err := recipe.RequireDefinition(ctx, runtime.Store, plan.Recipe)
	if err != nil {
		return AutomationExecution{}, err
	}
	inputs, err := runtime.webhookInputs(ctx, authority, definition)
	if err != nil {
		return AutomationExecution{}, err
	}
	_, executionID, err := automationExecutionIdentity(plan, operationKey.String(), "")
	if err != nil {
		return AutomationExecution{}, err
	}
	if execution, found, statusErr := existingAutomationExecution(runtime.Operations, executionID, plan, false); statusErr != nil || found {
		return execution, statusErr
	}
	causal, err := webhookCausalContext(delivery, authority)
	if err != nil {
		return AutomationExecution{}, err
	}
	execution, err := runtime.admitCausal(ctx, operationKey.String(), plan, "", inputs, &causal)
	if err == nil {
		return execution, nil
	}
	// A concurrent delivery can win admission after the optimistic status
	// check. Returning that exact operation makes the race observationally one.
	if existing, found, statusErr := existingAutomationExecution(runtime.Operations, executionID, plan, true); statusErr != nil || found {
		return existing, statusErr
	}
	return AutomationExecution{}, err
}

func webhookCausalContext(
	delivery, authority runrecord.WebhookDelivery,
) (runrecord.CausalContext, error) {
	if delivery.Disposition == runrecord.WebhookDeliveryReplay {
		return runrecord.NewCausalRoot(runrecord.TriggerWebhook, authority.ID, delivery.ID)
	}
	return runrecord.NewCausalRoot(runrecord.TriggerWebhook, authority.ID)
}

func webhookExecutionAuthority(
	ctx context.Context,
	ledger runrecord.WebhookDeliveryLedger,
	delivery runrecord.WebhookDelivery,
) (runrecord.WebhookDelivery, artifact.ID, error) {
	switch delivery.Disposition {
	case runrecord.WebhookDeliveryAccepted:
		return delivery, delivery.ID, nil
	case runrecord.WebhookDeliveryDuplicate, runrecord.WebhookDeliveryReplay:
		original, err := ledger.Require(ctx, delivery.Original)
		if err != nil || original.Disposition != runrecord.WebhookDeliveryAccepted ||
			original.Policy != delivery.Policy || original.Plan != delivery.Plan ||
			original.Source != delivery.Source || original.Idempotency != delivery.Idempotency ||
			original.Payload != delivery.Payload || original.PayloadBytes != delivery.PayloadBytes ||
			!delivery.PayloadStored {
			return runrecord.WebhookDelivery{}, artifact.ID{}, errors.Join(
				errors.New("workflow runtime: webhook retry authority differs"), err,
			)
		}
		operationKey := original.ID
		if delivery.Disposition == runrecord.WebhookDeliveryReplay {
			operationKey = delivery.ID
		}
		return original, operationKey, nil
	default:
		return runrecord.WebhookDelivery{}, artifact.ID{}, errors.New("workflow runtime: webhook delivery is not executable")
	}
}

func (runtime AutomationRuntime) webhookInputs(
	ctx context.Context,
	delivery runrecord.WebhookDelivery,
	definition recipe.Definition,
) (map[recipe.PortName]Value, error) {
	if !delivery.PayloadStored {
		return nil, errors.New("workflow runtime: webhook payload is not stored")
	}
	content, found, err := artifact.ReadContent(ctx, runtime.Store, delivery.Payload)
	if err != nil || !found {
		return nil, errors.Join(errors.New("workflow runtime: webhook payload is absent"), err)
	}
	if content.Descriptor.Size != delivery.PayloadBytes ||
		content.Descriptor.MediaType != runrecord.WebhookPayloadMediaType ||
		content.Descriptor.Schema != runrecord.WebhookPayloadSchema {
		return nil, errors.New("workflow runtime: webhook payload contract differs")
	}
	var raw map[string]json.RawMessage
	if err := strictjson.DecodeBytes(content.Data, &raw); err != nil || raw == nil {
		return nil, errors.Join(errors.New("workflow runtime: webhook payload is not an input object"), err)
	}
	inputs, err := BindAutomationInputs(definition, raw)
	if err != nil {
		return nil, err
	}
	return inputs, nil
}

func existingAutomationExecution(
	manager *operation.Manager,
	executionID artifact.ID,
	plan workflowcontract.AutomationExecutionPlan,
	convergeTerminal bool,
) (AutomationExecution, bool, error) {
	status, found := manager.Status(executionID)
	if !found {
		return AutomationExecution{}, false, nil
	}
	if status.Task != plan.Task || status.Recipe != plan.Recipe {
		return AutomationExecution{}, false, errors.New("workflow runtime: webhook operation authority differs")
	}
	if !convergeTerminal {
		switch status.State {
		case operation.StateFailed, operation.StateCancelled, operation.StateBlocked:
			return AutomationExecution{}, false, nil
		}
	}
	return AutomationExecution{Operation: executionID, Plan: plan}, true, nil
}
