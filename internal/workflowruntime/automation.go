package workflowruntime

import (
	"context"
	"encoding/json"
	"errors"
	"slices"
	"time"

	"overgo/internal/agenttool"
	"overgo/internal/artifact"
	"overgo/internal/operation"
	"overgo/internal/operatoraction"
	"overgo/internal/recipe"
	"overgo/internal/runrecord"
	"overgo/internal/workflowcontract"
)

// AutomationRuntime admits active automation plans into the common operation
// manager and executes their recipe stages through Runtime.
type AutomationRuntime struct {
	Store      artifact.Repository
	Operations *operation.Manager
	Catalog    *recipe.Catalog
	Adapters   map[recipe.ModuleID]Adapter
	Tools      *agenttool.Executor
}

// AutomationExecution identifies one admitted operation and its exact plan.
type AutomationExecution struct {
	Operation artifact.ID
	Plan      workflowcontract.AutomationExecutionPlan
	Claim     artifact.ID
}

// AutomationClock supplies scheduler time and is injectable in tests and hosts.
type AutomationClock interface {
	Now() time.Time
}

// SubmitManual compiles active authority and admits a manual run.
func (runtime AutomationRuntime) SubmitManual(
	ctx context.Context,
	name, key string,
	inputs map[recipe.PortName]Value,
) (AutomationExecution, error) {
	compiler := workflowcontract.AutomationCompiler{Repository: runtime.Store, Catalog: runtime.Catalog}
	plan, err := compiler.Compile(ctx, name)
	if err != nil {
		return AutomationExecution{}, err
	}
	if plan.Trigger.Kind != recipe.AutomationTriggerManual {
		return AutomationExecution{}, errors.New("workflow runtime: automation is not manually triggerable")
	}
	if err := runtime.publishAutomationPlan(ctx, plan); err != nil {
		return AutomationExecution{}, err
	}
	return runtime.admit(ctx, key, plan, "", inputs)
}

// SubmitManualTo admits a manual run with one policy-allowlisted destination.
func (runtime AutomationRuntime) SubmitManualTo(
	ctx context.Context,
	name, key, destination string,
	inputs map[recipe.PortName]Value,
) (AutomationExecution, error) {
	compiler := workflowcontract.AutomationCompiler{Repository: runtime.Store, Catalog: runtime.Catalog}
	plan, err := compiler.Compile(ctx, name)
	if err != nil {
		return AutomationExecution{}, err
	}
	if plan.Trigger.Kind != recipe.AutomationTriggerManual {
		return AutomationExecution{}, errors.New("workflow runtime: automation is not manually triggerable")
	}
	if err := runtime.publishAutomationPlan(ctx, plan); err != nil {
		return AutomationExecution{}, err
	}
	return runtime.admit(ctx, key, plan, destination, inputs)
}

// RecoverAutomation resumes one exact persisted plan and operation identity.
func (runtime AutomationRuntime) RecoverAutomation(
	ctx context.Context,
	planID artifact.ID,
	key string,
	inputs map[recipe.PortName]Value,
) (AutomationExecution, error) {
	compiler := workflowcontract.AutomationCompiler{Repository: runtime.Store, Catalog: runtime.Catalog}
	plan, err := compiler.Require(ctx, planID)
	if err != nil {
		return AutomationExecution{}, err
	}
	return runtime.admit(ctx, key, plan, "", inputs)
}

// ScheduleAutomation claims and admits the due slot for one active schedule.
// Fired is false when no slot is due or another caller already owns it.
func (runtime AutomationRuntime) ScheduleAutomation(
	ctx context.Context,
	name string,
	clock AutomationClock,
	inputs map[recipe.PortName]Value,
) (execution AutomationExecution, fired bool, err error) {
	if clock == nil {
		return execution, false, errors.New("workflow runtime: automation clock is absent")
	}
	compiler := workflowcontract.AutomationCompiler{Repository: runtime.Store, Catalog: runtime.Catalog}
	plan, err := compiler.Compile(ctx, name)
	if err != nil {
		return execution, false, err
	}
	if plan.Trigger.Kind != recipe.AutomationTriggerSchedule {
		return execution, false, errors.New("workflow runtime: automation is not scheduled")
	}
	if err := runtime.publishAutomationPlan(ctx, plan); err != nil {
		return execution, false, err
	}
	authority := runrecord.AutomationScheduleAuthority{Repository: runtime.Store}
	current, found, err := authority.Current(ctx, name)
	if err != nil {
		return execution, false, err
	}
	var previous *time.Time
	if found {
		due := current.Due()
		previous = &due
	}
	due, ready, err := plan.Trigger.Due(clock.Now(), previous)
	if err != nil || !ready {
		return execution, false, err
	}
	claim, won, err := authority.Claim(ctx, name, plan.ID, due)
	if err != nil || !won {
		return execution, false, err
	}
	execution, err = runtime.admit(ctx, claim.ID.String(), plan, "", inputs)
	execution.Claim = claim.ID
	return execution, err == nil, err
}

// RecoverScheduledAutomation resumes an exact claimed schedule slot after a
// process interruption without consulting a newer active alias.
func (runtime AutomationRuntime) RecoverScheduledAutomation(
	ctx context.Context,
	claimID artifact.ID,
	inputs map[recipe.PortName]Value,
) (AutomationExecution, error) {
	authority := runrecord.AutomationScheduleAuthority{Repository: runtime.Store}
	claim, err := authority.Require(ctx, claimID)
	if err != nil {
		return AutomationExecution{}, err
	}
	compiler := workflowcontract.AutomationCompiler{Repository: runtime.Store, Catalog: runtime.Catalog}
	plan, err := compiler.Require(ctx, claim.Plan)
	if err != nil || plan.Name != claim.Name || plan.Trigger.Kind != recipe.AutomationTriggerSchedule {
		return AutomationExecution{}, errors.Join(errors.New("workflow runtime: scheduled recovery authority differs"), err)
	}
	execution, err := runtime.admit(ctx, claim.ID.String(), plan, "", inputs)
	execution.Claim = claim.ID
	return execution, err
}

func (runtime AutomationRuntime) admit(
	ctx context.Context,
	key string,
	plan workflowcontract.AutomationExecutionPlan,
	destination string,
	inputs map[recipe.PortName]Value,
) (AutomationExecution, error) {
	if ctx == nil || runtime.Store == nil || runtime.Operations == nil || runtime.Catalog == nil {
		return AutomationExecution{}, errors.New("workflow runtime: automation runtime authority is absent")
	}
	definition, err := recipe.RequireDefinition(ctx, runtime.Store, plan.Recipe)
	if err != nil {
		return AutomationExecution{}, err
	}
	program, err := recipe.CompileProgram(definition, runtime.Catalog)
	if err != nil {
		return AutomationExecution{}, err
	}
	destination, err = automationDeliveryDestination(plan.Delivery, destination)
	if err != nil {
		return AutomationExecution{}, err
	}
	admissionKey, err := automationAdmissionKey(key, destination)
	if err != nil {
		return AutomationExecution{}, err
	}
	executionID, err := plan.ExecutionID(admissionKey)
	if err != nil {
		return AutomationExecution{}, err
	}
	execute := func(runContext context.Context, reporter operation.Reporter) (operation.Completion, error) {
		engine, createErr := NewForProgram(runtime.Store, program)
		if createErr != nil {
			return operation.Completion{}, createErr
		}
		registered := make(map[recipe.ModuleID]struct{})
		for _, stage := range program.Stages() {
			module := stage.Node.Module
			if _, found := registered[module]; found {
				continue
			}
			adapter := runtime.Adapters[module]
			if adapter == nil {
				return operation.Completion{}, errors.New("workflow runtime: automation adapter is absent")
			}
			if registerErr := engine.Register(module, adapter); registerErr != nil {
				return operation.Completion{}, registerErr
			}
			registered[module] = struct{}{}
		}
		result, executeErr := engine.ExecuteProgram(
			runContext, "automation/run/"+executionID.String(), reporter.OperationID(), reporter, program, inputs,
		)
		if executeErr == nil && plan.Delivery.Kind == recipe.AutomationDeliveryTool {
			executeErr = runtime.deliverAutomation(
				runContext, reporter.OperationID(), plan, destination, result.Run.Outputs,
			)
		}
		reporter.Publishing()
		return operation.Completion{Run: result.Run.ID, Outputs: slices.Clone(result.Run.Outputs)}, executeErr
	}
	operationID, err := runtime.Operations.Recover(
		ctx, executionID, operation.Request{Task: plan.Task, Recipe: plan.Recipe}, execute,
	)
	if err != nil {
		return AutomationExecution{}, err
	}
	return AutomationExecution{Operation: operationID, Plan: plan}, nil
}

func (runtime AutomationRuntime) deliverAutomation(
	ctx context.Context,
	operationID artifact.ID,
	plan workflowcontract.AutomationExecutionPlan,
	destination string,
	outputs []artifact.ID,
) error {
	if runtime.Tools == nil {
		return errors.New("workflow runtime: automation delivery executor is absent")
	}
	manual, err := agenttool.RequireManual(ctx, runtime.Store, plan.Delivery.Tool)
	if err != nil {
		return err
	}
	idempotency, err := artifact.JSONID(artifact.KindEvidence, struct {
		Plan        artifact.ID   `json:"plan"`
		Operation   artifact.ID   `json:"operation"`
		Tool        artifact.ID   `json:"tool"`
		Destination string        `json:"destination"`
		Outputs     []artifact.ID `json:"outputs"`
	}{Plan: plan.ID, Operation: operationID, Tool: manual.ID, Destination: destination, Outputs: outputs})
	if err != nil {
		return err
	}
	action := operatoraction.Action{
		Code: manual.Name, Summary: "Deliver automation outputs",
		Argv: []string{manual.Name, destination, idempotency.String()},
	}
	decision, found, err := runrecord.ResolveHumanDecision(ctx, runtime.Store, operationID)
	if err != nil {
		return err
	}
	prior := artifact.ID{}
	if found {
		prior = decision.Prior
	}
	request, err := operatoraction.NewApprovalRequest(operationID, plan.Recipe, action, prior)
	if err != nil {
		return err
	}
	if !found || decision.Answer != operatoraction.AnswerGrant || !decision.Binds(request) {
		return operatoraction.Recoverable(errors.New("workflow runtime: automation delivery approval required"), operatoraction.Block{
			Subject: plan.Recipe, Reason: "External automation delivery requires exact approval",
			Evidence: []artifact.ID{plan.ID, plan.Delivery.Authorization}, Actions: []operatoraction.Action{action},
		})
	}
	authority := runrecord.AutomationDeliveryAuthority{Repository: runtime.Store}
	attempt, won, err := authority.Begin(ctx, runrecord.AutomationDeliveryAttempt{
		Plan: plan.ID, Operation: operationID, Tool: manual.ID, Destination: destination,
		Outputs: slices.Clone(outputs), Idempotency: idempotency, State: runrecord.AutomationDeliveryAdmitted,
	})
	if err != nil {
		return err
	}
	if !won {
		if attempt.State == runrecord.AutomationDeliverySucceeded {
			return nil
		}
		return errors.New("workflow runtime: automation delivery has an unresolved prior attempt")
	}
	payload := make([]string, len(outputs))
	for index, output := range outputs {
		payload[index] = output.String()
	}
	arguments, err := json.Marshal(map[string]any{
		plan.Delivery.DestinationArgument: destination,
		plan.Delivery.PayloadArgument:     map[string]any{"artifacts": payload},
		plan.Delivery.IdempotencyArgument: idempotency.String(),
	})
	if err != nil {
		return err
	}
	response, invokeErr := runtime.Tools.Invoke(ctx, manual, arguments)
	if invokeErr != nil {
		_, finishErr := authority.Finish(context.WithoutCancel(ctx), attempt, nil, invokeErr.Error())
		return errors.Join(invokeErr, finishErr)
	}
	result, err := (recipe.ToolResult{
		CallID: idempotency.String(), Module: recipe.ModuleID(manual.Name), Output: response,
	}).ArtifactContent()
	if err != nil {
		return err
	}
	_, err = authority.Finish(context.WithoutCancel(ctx), attempt, &result, "")
	return err
}

func automationDeliveryDestination(policy recipe.AutomationDeliveryPolicy, requested string) (string, error) {
	if policy.Kind == recipe.AutomationDeliveryArtifact {
		if requested != "" {
			return "", errors.New("workflow runtime: artifact delivery does not admit a destination")
		}
		return "", nil
	}
	if requested == "" && len(policy.Destinations) == 1 {
		requested = policy.Destinations[0]
	}
	if !slices.Contains(policy.Destinations, requested) {
		return "", errors.New("workflow runtime: automation delivery destination is not allowlisted")
	}
	return requested, nil
}

func automationAdmissionKey(key, destination string) (string, error) {
	if key == "" {
		return "", errors.New("workflow runtime: automation admission key is absent")
	}
	id, err := artifact.JSONID(artifact.KindProfile, struct {
		Key         string `json:"key"`
		Destination string `json:"destination,omitempty"`
	}{Key: key, Destination: destination})
	return id.String(), err
}

func (runtime AutomationRuntime) publishAutomationPlan(
	ctx context.Context,
	plan workflowcontract.AutomationExecutionPlan,
) error {
	content, err := plan.Content()
	if err != nil {
		return err
	}
	resources, err := plan.Resources.Content()
	if err != nil {
		return err
	}
	batch, err := artifact.NewDocumentBatch(
		"automation/plan/"+plan.ID.String(), []artifact.Content{resources, content}, plan.Lineage(), nil,
	)
	if err != nil {
		return err
	}
	_, err = artifact.CommitBatch(ctx, runtime.Store, batch)
	if errors.Is(err, artifact.ErrNoChange) {
		return nil
	}
	return err
}
