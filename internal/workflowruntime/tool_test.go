package workflowruntime

import (
	"context"
	"encoding/json"
	"errors"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/overgodb"
	"overgo/internal/recipe"
	"overgo/internal/strictjson"
	"overgo/internal/testcheck"
	"overgo/internal/testutil"
)

const (
	testToolModule    recipe.ModuleID = "sum"
	testToolRetention                 = 4
)

type toolArguments struct {
	Left  int `json:"left"`
	Right int `json:"right"`
}

type toolOutput struct {
	Value int `json:"value"`
}

func TestTypedToolExecution(t *testing.T) {
	executor := toolExecutorFixture(t, typedToolAdapter(
		func(_ context.Context, value toolArguments) (toolOutput, error) {
			return toolOutput{Value: value.Left + value.Right}, nil
		}))
	call := testcheck.Must(recipe.NewToolCall("call", testToolModule, json.RawMessage(`{"left":2,"right":3}`)))(t)
	result := testcheck.Must(executor.ExecuteTool(t.Context(), call))(t)
	var output toolOutput
	if err := json.Unmarshal(result.Output, &output); err != nil || output.Value != 5 {
		t.Fatalf("output = %s, error = %v", result.Output, err)
	}
}

func TestLongRunningToolOperation(t *testing.T) {
	started := make(chan struct{})
	executor := toolExecutorFixture(t, typedToolAdapter(
		func(ctx context.Context, _ toolArguments) (toolOutput, error) {
			close(started)
			<-ctx.Done()
			return toolOutput{}, ctx.Err()
		}))
	call := testcheck.Must(recipe.NewToolCall("long", testToolModule, json.RawMessage(`{}`)))(t)
	ctx, cancel := context.WithCancel(t.Context())
	done := make(chan error, 1)
	go func() {
		_, executeErr := executor.ExecuteTool(ctx, call)
		done <- executeErr
	}()
	<-started
	cancel()
	if err := <-done; !errors.Is(err, context.Canceled) {
		t.Fatalf("execute error = %v", err)
	}
}

func toolExecutorFixture(
	t *testing.T,
	execute func(context.Context, recipe.ToolCall) (recipe.ToolResult, error),
) *ToolExecutor {
	t.Helper()
	store := testcheck.Must(overgodb.Open(t.TempDir()))(t)
	model := testutil.ArtifactID(t, artifact.KindModel, "tool-model")
	if _, err := store.Commit(t.Context(), artifact.Batch{
		Key: "tool/model", Artifacts: []artifact.Descriptor{{ID: model}},
	}); err != nil {
		t.Fatal(err)
	}
	module := recipe.ToolModule(testToolModule, recipe.PlacementHost)
	node := recipe.Node{ID: "execute", Module: module.ID, Placement: recipe.PlacementHost}
	definition := testcheck.Must(recipe.NewDefinitionWithDependencies(
		recipe.TaskInference,
		[]recipe.Dependency{{Role: recipe.DependencyModel, Artifact: model}},
		[]recipe.Node{node}, nil,
		[]recipe.Input{{Name: recipe.ToolCallPort, Data: recipe.DataToolCall, Target: recipe.Endpoint{Node: node.ID, Port: recipe.ToolCallPort}}},
		[]recipe.Output{{Name: recipe.ToolResultPort, Data: recipe.DataToolResult, Source: recipe.Endpoint{Node: node.ID, Port: recipe.ToolResultPort}}},
	))(t)
	catalog := testcheck.Must(recipe.NewCatalog(module))(t)
	program := testcheck.Must(recipe.CompileProgram(definition, catalog))(t)
	executor := testcheck.Must(NewToolExecutor(store, program, testToolRetention, execute))(t)
	t.Cleanup(func() { executor.Close(); store.Close() })
	return executor
}

func typedToolAdapter[Arguments, Output any](
	execute func(context.Context, Arguments) (Output, error),
) func(context.Context, recipe.ToolCall) (recipe.ToolResult, error) {
	return func(ctx context.Context, call recipe.ToolCall) (recipe.ToolResult, error) {
		var arguments Arguments
		if err := strictjson.DecodeBytes(call.Arguments, &arguments); err != nil {
			return recipe.ToolResult{}, err
		}
		output, err := execute(ctx, arguments)
		if err != nil {
			return recipe.ToolResult{}, err
		}
		encoded, err := json.Marshal(output)
		if err != nil {
			return recipe.ToolResult{}, err
		}
		return recipe.ToolResult{CallID: call.ID, Module: call.Module, Output: encoded}, nil
	}
}
