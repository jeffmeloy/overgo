package workflowruntime

import (
	"context"
	"errors"
	"slices"
	"time"

	"overgo/internal/artifact"
	"overgo/internal/operation"
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
	return runtime.admit(ctx, key, plan, inputs)
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
	return runtime.admit(ctx, key, plan, inputs)
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
	execution, err = runtime.admit(ctx, claim.ID.String(), plan, inputs)
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
	execution, err := runtime.admit(ctx, claim.ID.String(), plan, inputs)
	execution.Claim = claim.ID
	return execution, err
}

func (runtime AutomationRuntime) admit(
	ctx context.Context,
	key string,
	plan workflowcontract.AutomationExecutionPlan,
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
	executionID, err := plan.ExecutionID(key)
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
