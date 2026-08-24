package workflowruntime

import (
	"context"
	"errors"
	"fmt"

	"overgo/internal/artifact"
	"overgo/internal/operation"
	"overgo/internal/recipe"
	"overgo/internal/strictjson"
)

// ToolExecution contains the typed result and durable workflow identity.
type ToolExecution struct {
	Result  recipe.ToolResult
	Run     artifact.ID
	Outputs []artifact.ID
}

// ToolExecutor owns one admitted tool recipe and its operation lifecycle.
type ToolExecutor struct {
	runtime    *Runtime
	program    recipe.Program
	admission  recipe.ToolAdmission
	operations *operation.Manager
}

// NewToolExecutor binds execution to one compiled tool recipe.
func NewToolExecutor(
	store artifact.Repository,
	program recipe.Program,
	retention int,
	execute func(context.Context, recipe.ToolCall) (recipe.ToolResult, error),
) (*ToolExecutor, error) {
	if execute == nil {
		return nil, errors.New("workflow runtime: nil tool adapter")
	}
	admission, err := recipe.AdmitTool(program)
	if err != nil {
		return nil, err
	}
	runtime, err := NewForProgram(store, program)
	if err != nil {
		return nil, err
	}
	operations, err := operation.NewManager(retention)
	if err != nil {
		return nil, err
	}
	executor := &ToolExecutor{runtime: runtime, program: program, admission: admission, operations: operations}
	err = RegisterContextStage(runtime, admission.Module, admission.Model,
		func(ctx context.Context, call recipe.ToolCall) (recipe.ToolResult, error) {
			if err := call.Validate(); err != nil {
				return recipe.ToolResult{}, err
			}
			if call.Module != admission.Module {
				return recipe.ToolResult{}, fmt.Errorf("workflow runtime: tool module %q is not admitted", call.Module)
			}
			result, err := execute(ctx, call)
			if err != nil {
				return recipe.ToolResult{}, err
			}
			if result.CallID != call.ID || result.Module != call.Module {
				return recipe.ToolResult{}, errors.New("workflow runtime: tool result identity differs")
			}
			if _, err := result.ArtifactContent(); err != nil {
				return recipe.ToolResult{}, err
			}
			return result, nil
		}, func(result recipe.ToolResult) (artifact.Content, error) { return result.ArtifactContent() },
	)
	if err != nil {
		operations.Close()
		return nil, err
	}
	return executor, nil
}

// RecipeID returns the admitted recipe identity.
func (executor *ToolExecutor) RecipeID() artifact.ID {
	if executor == nil {
		return artifact.ID{}
	}
	return executor.admission.Recipe
}

// ExecuteTool runs one call and returns its typed result.
func (executor *ToolExecutor) ExecuteTool(ctx context.Context, call recipe.ToolCall) (recipe.ToolResult, error) {
	execution, err := executor.ExecuteToolOperation(ctx, call)
	return execution.Result, err
}

// ExecuteToolOperation runs one call through the durable operation owner.
func (executor *ToolExecutor) ExecuteToolOperation(ctx context.Context, call recipe.ToolCall) (ToolExecution, error) {
	if executor == nil || executor.runtime == nil || executor.operations == nil || ctx == nil {
		return ToolExecution{}, errors.New("workflow runtime: incomplete tool executor")
	}
	if err := call.Validate(); err != nil {
		return ToolExecution{}, err
	}
	if call.Module != executor.admission.Module {
		return ToolExecution{}, fmt.Errorf("workflow runtime: tool module %q is not admitted", call.Module)
	}
	content, err := call.ArtifactContent()
	if err != nil {
		return ToolExecution{}, err
	}
	type executionOutcome struct {
		execution ToolExecution
		err       error
	}
	completed := make(chan executionOutcome, 1)
	operationID, err := executor.operations.Submit(ctx, operation.Request{
		Task: recipe.TaskInference, Recipe: executor.admission.Recipe,
	}, func(runContext context.Context, reporter operation.Reporter) (operation.Completion, error) {
		result, executeErr := executor.runtime.ExecuteProgram(
			runContext, "tool/"+call.ID, reporter.OperationID(), reporter, executor.program,
			map[recipe.PortName]Value{
				recipe.ToolCallPort: ArtifactValue(recipe.DataToolCall, call, content),
			},
		)
		execution, resultErr := toolExecution(result)
		executeErr = errors.Join(executeErr, resultErr)
		completed <- executionOutcome{execution: execution, err: executeErr}
		return operation.Completion{Run: result.Run.ID, Outputs: execution.Outputs}, executeErr
	})
	if err != nil {
		return ToolExecution{}, err
	}
	status, waitErr := executor.operations.Wait(ctx, operationID)
	if waitErr != nil {
		executor.operations.Cancel(operationID)
		_, _ = executor.operations.Wait(context.Background(), operationID)
		<-completed
		return ToolExecution{}, waitErr
	}
	outcome := <-completed
	if outcome.err != nil {
		return outcome.execution, outcome.err
	}
	if status.State != operation.StateCompleted {
		return outcome.execution, fmt.Errorf("workflow runtime: tool operation ended in state %q", status.State)
	}
	return outcome.execution, nil
}

// Close releases tool operations after cancelling active calls.
func (executor *ToolExecutor) Close() {
	if executor != nil && executor.operations != nil {
		executor.operations.Close()
	}
}

func toolExecution(result Result) (ToolExecution, error) {
	value := result.Outputs[recipe.ToolResultPort]
	datum, ok := value.Single()
	if !ok || value.Kind != recipe.DataToolResult {
		return ToolExecution{Run: result.Run.ID}, errors.New("workflow runtime: tool result is absent")
	}
	output, ok := datum.Value.(recipe.ToolResult)
	if !ok && datum.Content != nil {
		if err := strictjson.DecodeBytes(datum.Content.Data, &output); err != nil {
			return ToolExecution{Run: result.Run.ID}, err
		}
		ok = true
	}
	if !ok {
		return ToolExecution{Run: result.Run.ID}, errors.New("workflow runtime: tool result has invalid type")
	}
	outputs := make([]artifact.ID, 0, 1)
	if datum.Artifact.ID.Valid() {
		outputs = append(outputs, datum.Artifact.ID)
	}
	return ToolExecution{Result: output, Run: result.Run.ID, Outputs: outputs}, nil
}
