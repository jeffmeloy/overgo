package server

import (
	"context"
	"encoding/json"
	"fmt"
	"slices"

	"overgo/internal/inference"
	"overgo/internal/recipe"
)

func (h *Handler) executePendingTools(ctx context.Context, messages []inference.ChatMessage) ([]inference.ChatMessage, error) {
	if h == nil || h.tools == nil || len(messages) == 0 {
		return messages, nil
	}
	last := messages[len(messages)-1]
	if last.Role != inference.ChatRoleAssistant || len(last.ToolCalls) == 0 {
		return messages, nil
	}
	resolved := slices.Clone(messages)
	for _, pending := range last.ToolCalls {
		call, err := recipe.NewToolCall(
			pending.ID, recipe.ModuleID(pending.Function.Name), json.RawMessage(pending.Function.Arguments),
		)
		if err != nil {
			return nil, fmt.Errorf("server tool call %q: %w", pending.ID, err)
		}
		result, err := h.tools.ExecuteTool(ctx, call)
		if err != nil {
			return nil, fmt.Errorf("server tool call %q: %w", pending.ID, err)
		}
		content := string(result.Output)
		var text string
		if json.Unmarshal(result.Output, &text) == nil {
			content = text
		}
		resolved = append(resolved, inference.ChatMessage{
			Role: inference.ChatRoleTool, Content: content, ToolCallID: result.CallID,
		})
	}
	return resolved, nil
}
