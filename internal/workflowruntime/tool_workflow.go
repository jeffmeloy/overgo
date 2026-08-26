package workflowruntime

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"

	"overgo/internal/agenttool"
	"overgo/internal/artifact"
	"overgo/internal/recipe"
	"overgo/internal/strictjson"
)

// ExecuteToolWorkflow runs a closed-world tool graph through the existing
// recipe runtime. The compiled graph owns ordering; exact manuals own effects
// and transports; the normal run record owns execution evidence.
func ExecuteToolWorkflow(
	ctx context.Context,
	store artifact.Repository,
	workflow agenttool.CompiledToolWorkflow,
	tools *agenttool.Executor,
	key string,
	arguments map[recipe.NodeID]json.RawMessage,
) (Result, error) {
	if ctx == nil || store == nil || tools == nil || key == "" {
		return Result{}, errors.New("workflow runtime: tool workflow authority is absent")
	}
	program := workflow.Program()
	runtime, err := NewForProgram(store, program)
	if err != nil {
		return Result{}, err
	}
	for _, module := range workflow.Modules() {
		manual, found := workflow.Manual(module)
		if !found {
			return Result{}, errors.New("workflow runtime: compiled tool manual is absent")
		}
		if err := runtime.Register(module, toolWorkflowAdapter(tools, manual, module)); err != nil {
			return Result{}, err
		}
	}
	calls, err := workflow.Calls(arguments)
	if err != nil {
		return Result{}, err
	}
	inputs := make(map[recipe.PortName]Value, len(calls))
	for port, call := range calls {
		content, err := call.ArtifactContent()
		if err != nil {
			return Result{}, err
		}
		inputs[port] = ArtifactValue(recipe.DataToolCall, call, content)
	}
	definition := program.Definition()
	operation, err := ExecutionID(definition.ID, key)
	if err != nil {
		return Result{}, err
	}
	return runtime.ExecuteProgram(ctx, "tool-workflow/"+operation.String(), operation, nil, program, inputs)
}

func toolWorkflowAdapter(
	tools *agenttool.Executor,
	manual agenttool.Manual,
	module recipe.ModuleID,
) AdapterFunc {
	return func(ctx context.Context, request StepRequest) (map[recipe.PortName]Value, error) {
		value := request.Inputs[recipe.ToolCallPort]
		datum, one := value.Single()
		if !one || value.Kind != recipe.DataToolCall {
			return nil, errors.New("workflow runtime: tool workflow call is absent")
		}
		call, ok := datum.Value.(recipe.ToolCall)
		if !ok && datum.Content != nil {
			if err := strictjson.DecodeBytes(datum.Content.Data, &call); err != nil {
				return nil, err
			}
			ok = true
		}
		if !ok || call.Module != module {
			return nil, fmt.Errorf("workflow runtime: tool call differs from module %q", module)
		}
		output, err := tools.Invoke(ctx, manual, call.Arguments)
		if err != nil {
			return nil, err
		}
		result := recipe.ToolResult{CallID: call.ID, Module: module, Output: output}
		content, err := result.ArtifactContent()
		if err != nil {
			return nil, err
		}
		return map[recipe.PortName]Value{
			recipe.ToolResultPort: ArtifactValue(recipe.DataToolResult, result, content),
		}, nil
	}
}
