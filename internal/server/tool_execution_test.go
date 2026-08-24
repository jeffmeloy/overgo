package server

import (
	"context"
	"encoding/json"
	"testing"

	"overgo/internal/inference"
	"overgo/internal/recipe"
)

type parityToolExecutor struct{}

func (parityToolExecutor) ExecuteTool(_ context.Context, call recipe.ToolCall) (recipe.ToolResult, error) {
	return recipe.ToolResult{
		CallID: call.ID, Module: call.Module, Output: json.RawMessage(`{"source":"shared"}`),
	}, nil
}

func TestProtocolToolExecutionParity(t *testing.T) {
	handler := &Handler{tools: parityToolExecutor{}}
	chat := []inference.ChatMessage{{
		Role: inference.ChatRoleAssistant,
		ToolCalls: []inference.ChatToolCall{{
			ID: "chat", Type: inference.ChatToolTypeFunction,
			Function: inference.ChatToolFunction{Name: "lookup", Arguments: `{}`},
		}},
	}}
	responses, err := handler.parseResponsesMessages(context.Background(), json.RawMessage(
		`[{"type":"function_call","call_id":"responses","name":"lookup","arguments":"{}"}]`,
	), "")
	if err != nil {
		t.Fatal(err)
	}
	anthropic, err := handler.parseAnthropicMessages(context.Background(), nil, json.RawMessage(
		`[{"role":"assistant","content":[{"type":"tool_use","id":"anthropic","name":"lookup","input":{}}]}]`,
	))
	if err != nil {
		t.Fatal(err)
	}
	for _, messages := range [][]inference.ChatMessage{chat, responses, anthropic} {
		resolved, err := handler.executePendingTools(context.Background(), messages)
		if err != nil {
			t.Fatal(err)
		}
		last := resolved[len(resolved)-1]
		if last.Role != inference.ChatRoleTool || last.Content != `{"source":"shared"}` {
			t.Fatalf("resolved message = %+v", last)
		}
	}
}
