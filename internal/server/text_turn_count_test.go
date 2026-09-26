package server

import (
	"encoding/json"
	"net/http"
	"testing"
)

// TestTextTurnCountsOnce holds each protocol's input-token count to the turn
// its generation route builds: the same body, tools, system prompt and all,
// counts exactly the prompt tokens the turn then reports generating from.
func TestTextTurnCountsOnce(t *testing.T) {
	t.Parallel()
	chatTools := `"tools":[{"type":"function","function":{"name":"weather","description":"forecast","parameters":{"type":"object","properties":{"city":{"type":"string"}}}}}]`
	responseTools := `"tools":[{"type":"function","name":"weather","description":"forecast","parameters":{"type":"object","properties":{"city":{"type":"string"}}}}]`
	anthropicTools := `"tools":[{"name":"weather","description":"forecast","input_schema":{"type":"object","properties":{"city":{"type":"string"}}}}]`
	for _, protocol := range []struct {
		name, generate, count, withTools, withoutTools string
	}{
		{"chat", "/v1/chat/completions", "/v1/chat/completions/input_tokens",
			`{"max_tokens":1,` + chatTools + `,"messages":[{"role":"system","content":"be brief"},{"role":"user","content":"weather in Paris?"}]}`,
			`{"max_tokens":1,"messages":[{"role":"system","content":"be brief"},{"role":"user","content":"weather in Paris?"}]}`},
		{"responses", "/v1/responses", "/v1/responses/input_tokens",
			`{"max_output_tokens":1,"store":false,"instructions":"be brief",` + responseTools + `,"input":"weather in Paris?"}`,
			`{"max_output_tokens":1,"store":false,"instructions":"be brief","input":"weather in Paris?"}`},
		{"anthropic", "/v1/messages", "/v1/messages/count_tokens",
			`{"max_tokens":1,"system":"be brief",` + anthropicTools + `,"messages":[{"role":"user","content":"weather in Paris?"}]}`,
			`{"max_tokens":1,"system":"be brief","messages":[{"role":"user","content":"weather in Paris?"}]}`},
	} {
		t.Run(protocol.name, func(t *testing.T) {
			t.Parallel()
			handler := newTestHandler(t, &fakeGenerator{pieces: []string{"sunny"}})
			counted := func(body string) int {
				t.Helper()
				answer := serveTestRequest(handler, http.MethodPost, protocol.count, body)
				var count struct {
					InputTokens int `json:"input_tokens"`
				}
				if answer.Code != http.StatusOK || json.Unmarshal(answer.Body.Bytes(), &count) != nil || count.InputTokens == 0 {
					t.Fatalf("count status=%d body=%s", answer.Code, answer.Body)
				}
				return count.InputTokens
			}
			withTools := counted(protocol.withTools)
			counted(protocol.withoutTools)
			answer := serveTestRequest(handler, http.MethodPost, protocol.generate, protocol.withTools)
			var generated struct {
				Usage struct {
					PromptTokens int `json:"prompt_tokens"`
					InputTokens  int `json:"input_tokens"`
				} `json:"usage"`
			}
			if answer.Code != http.StatusOK || json.Unmarshal(answer.Body.Bytes(), &generated) != nil {
				t.Fatalf("generation status=%d body=%s", answer.Code, answer.Body)
			}
			prompt := max(generated.Usage.PromptTokens, generated.Usage.InputTokens)
			if prompt != withTools {
				t.Errorf("the count read %d tokens and the turn generated from %d", withTools, prompt)
			}
		})
	}
}
