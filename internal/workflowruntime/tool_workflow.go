package workflowruntime

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"slices"
	"strings"

	"overgo/internal/agenttool"
	"overgo/internal/artifact"
	"overgo/internal/recipe"
	"overgo/internal/runrecord"
	"overgo/internal/strictjson"
)

// toolCallPrefix opens the ID agenttool's CompiledToolWorkflow.Calls gives each call, before the step it calls.
const toolCallPrefix = "call."

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
		if err := runtime.Register(module, toolWorkflowAdapter(store, tools, manual, module)); err != nil {
			return Result{}, err
		}
	}
	calls, err := workflow.Calls(arguments)
	if err != nil {
		return Result{}, err
	}
	inputs := make(map[recipe.PortName]Value, len(calls))
	sent := make(map[recipe.NodeID]artifact.ID, len(calls))
	for port, call := range calls {
		content, err := call.ArtifactContent()
		if err != nil {
			return Result{}, err
		}
		inputs[port] = ArtifactValue(recipe.DataToolCall, call, content)
		sent[recipe.NodeID(strings.TrimPrefix(call.ID, toolCallPrefix))] = content.Descriptor.ID
	}
	definition := program.Definition()
	operation, err := ExecutionID(definition.ID, key)
	if err != nil {
		return Result{}, err
	}
	// A key that ran answers the calls it ran: a step's recorded outputs replay by operation and node, so a changed
	// call under the key would take them as its own (SQA-328).
	for step, call := range sent {
		receipt, ran, err := runrecord.ResolveStageReceipt(ctx, store, operation, step)
		if err != nil {
			return Result{}, fmt.Errorf("workflow runtime: step %s's receipt under key %q cannot be read: %w", step, key, err)
		}
		at := slices.IndexFunc(receipt.Inputs, func(binding runrecord.StageBinding) bool { return binding.Port == recipe.ToolCallPort })
		if ran && (at < 0 || !slices.Equal(receipt.Inputs[at].Artifacts, []artifact.ID{call})) {
			return Result{}, fmt.Errorf("workflow runtime: step %s ran under key %q with another call; run the changed call under a key of its own", step, key)
		}
	}
	return runtime.ExecuteProgram(ctx, "tool-workflow/"+operation.String(), operation, nil, program, inputs)
}

func toolWorkflowAdapter(
	store artifact.Reader,
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
		if err := agenttool.CheckArgvAuthority(ctx, store, manual); err != nil {
			return nil, err
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
