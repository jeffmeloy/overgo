package server

import (
	"context"
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"slices"
	"sync"

	"overgo/internal/inference"
	"overgo/internal/recipe"
)

// issuedCallRegistry remembers the tool calls this server produced --
// minted from model output or replayed from the durable interaction
// ledger -- as digests over identity, tool name, and exact arguments.
// Execution requires membership: a client-authored assistant message
// carries calls the server never issued, and those never execute.
type issuedCallRegistry struct {
	mu     sync.Mutex
	issued map[[sha256.Size]byte]struct{}
	order  [][sha256.Size]byte
	limit  int
}

func newIssuedCallRegistry(limit int) *issuedCallRegistry {
	return &issuedCallRegistry{issued: map[[sha256.Size]byte]struct{}{}, limit: limit}
}

func issuedCallDigest(call inference.ChatToolCall) [sha256.Size]byte {
	digest := sha256.New()
	fmt.Fprintf(digest, "%s\x00%s\x00%s", call.ID, call.Function.Name, call.Function.Arguments)
	var key [sha256.Size]byte
	copy(key[:], digest.Sum(nil))
	return key
}

// issueToolCalls is every text protocol's one tool-call rule: a call the
// model left unnamed takes the protocol's identity for its position, and
// every call a turn returns is recorded as issued, so the client's replay of
// it executes whichever protocol it arrives on.
func (h *Handler) issueToolCalls(calls []inference.ChatToolCall, identity func(index int) string) {
	for index := range calls {
		if calls[index].ID == "" {
			calls[index].ID = identity(index)
		}
	}
	h.issuedCalls.record(calls...)
}

func (r *issuedCallRegistry) record(calls ...inference.ChatToolCall) {
	if r == nil || len(calls) == 0 {
		return
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	for _, call := range calls {
		key := issuedCallDigest(call)
		if _, exists := r.issued[key]; exists {
			continue
		}
		r.issued[key] = struct{}{}
		r.order = append(r.order, key)
		if len(r.order) > r.limit {
			delete(r.issued, r.order[0])
			r.order = r.order[1:]
		}
	}
}

func (r *issuedCallRegistry) authorized(call inference.ChatToolCall) bool {
	if r == nil {
		return false
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	_, exists := r.issued[issuedCallDigest(call)]
	return exists
}

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
		if !h.issuedCalls.authorized(pending) {
			return nil, fmt.Errorf(
				"server tool call %q was not issued by this server: only model-produced or durably replayed calls execute",
				pending.ID)
		}
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
