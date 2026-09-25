package server

import (
	"net/http"
	"strings"
	"testing"

	"overgo/internal/inference"
)

// TestTextProtocolsShareOneTurn holds every text protocol to one tool-call
// rule: whichever protocol returns a tool call, buffered or streamed, names
// it with that protocol's identity and records exactly that call (identity,
// tool and arguments) as issued, so the client's replay of it executes. The
// Anthropic protocol once returned calls it never recorded, and its own
// history's tool results were refused.
func TestTextProtocolsShareOneTurn(t *testing.T) {
	t.Parallel()
	const call = `<tool_call><function=weather><parameter=city>Paris</parameter></function></tool_call>`
	chatTools := `"tools":[{"type":"function","function":{"name":"weather","description":"forecast","parameters":{"type":"object","properties":{"city":{"type":"string"}}}}}]`
	anthropicTools := `"tools":[{"name":"weather","description":"forecast","input_schema":{"type":"object","properties":{"city":{"type":"string"}}}}]`
	for _, protocol := range []struct {
		name, path, body, identity string
	}{
		{"chat", "/v1/chat/completions", `{"max_tokens":1,"tool_choice":"required",` + chatTools + `,"messages":[{"role":"user","content":"weather?"}]}`, "call_"},
		{"chat stream", "/v1/chat/completions", `{"max_tokens":1,"stream":true,"tool_choice":"required",` + chatTools + `,"messages":[{"role":"user","content":"weather?"}]}`, "call_"},
		{"responses", "/v1/responses", `{"max_output_tokens":1,"store":false,"tool_choice":"required","tools":[{"type":"function","name":"weather","description":"forecast","parameters":{"type":"object","properties":{"city":{"type":"string"}}}}],"input":"weather?"}`, "call_"},
		{"responses stream", "/v1/responses", `{"max_output_tokens":1,"store":false,"stream":true,"tool_choice":"required","tools":[{"type":"function","name":"weather","description":"forecast","parameters":{"type":"object","properties":{"city":{"type":"string"}}}}],"input":"weather?"}`, "call_"},
		{"anthropic", "/v1/messages", `{"max_tokens":1,"tool_choice":{"type":"any"},` + anthropicTools + `,"messages":[{"role":"user","content":"weather?"}]}`, "toolu_"},
		{"anthropic stream", "/v1/messages", `{"max_tokens":1,"stream":true,"tool_choice":{"type":"any"},` + anthropicTools + `,"messages":[{"role":"user","content":"weather?"}]}`, "toolu_"},
	} {
		t.Run(protocol.name, func(t *testing.T) {
			t.Parallel()
			handler := newTestHandler(t, &fakeGenerator{pieces: []string{call}})
			answer := serveTestRequest(handler, http.MethodPost, protocol.path, protocol.body)
			if answer.Code != http.StatusOK {
				t.Fatalf("status=%d body=%s", answer.Code, answer.Body)
			}
			body := answer.Body.String()
			start := strings.Index(body, `:"`+protocol.identity)
			if start < 0 {
				t.Fatalf("no %s identity in %s", protocol.identity, body)
			}
			id, _, _ := strings.Cut(body[start+2:], `"`)
			issued := inference.ChatToolCall{ID: id, Type: inference.ChatToolTypeFunction,
				Function: inference.ChatToolFunction{Name: "weather", Arguments: `{"city":"Paris"}`}}
			if !handler.issuedCalls.authorized(issued) {
				t.Fatalf("%s returned call %s without recording it as issued", protocol.name, id)
			}
		})
	}
}
