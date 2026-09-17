package server

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
	"testing"

	"overgo/internal/capabilityruntime"
	"overgo/internal/inference"
	"overgo/internal/tokenizer"
)

type protocolCacheGenerator struct {
	*recipeInspectorGenerator
	supported bool
	warm      bool
}

func (g *protocolCacheGenerator) SupportsPromptCache() bool { return g.supported }

func (g *protocolCacheGenerator) Generate(ctx context.Context, prompt string, options inference.GenerateOptions) ([]tokenizer.TokenID, string, error) {
	// Report a partial hit so protocols must distinguish evaluated tokens,
	// cached tokens and the whole prompt rather than infer a full hit.
	g.promptCached = 0
	if options.CachePrompt && g.warm {
		g.promptCached = len(options.PromptTokenIDs) - 1
	}
	g.warm = options.CachePrompt
	return g.recipeInspectorGenerator.Generate(ctx, prompt, options)
}

func TestProtocolPromptCacheReuse(t *testing.T) {
	for _, protocol := range []struct{ path, input, budget string }{
		{"/v1/chat/completions", `"messages":[{"role":"user","content":"hi"}]`, "max_tokens"},
		{"/v1/responses", `"input":"hi"`, "max_output_tokens"},
		{"/v1/messages", `"messages":[{"role":"user","content":"hi"}]`, "max_tokens"},
	} {
		for _, stream := range []bool{false, true} {
			for _, supported := range []bool{false, true} {
				t.Run(fmt.Sprintf("%s/stream=%t/support=%t", protocol.path, stream, supported), func(t *testing.T) {
					generator := &protocolCacheGenerator{recipeInspectorGenerator: responseRecipeGenerator(t, &fakeGenerator{}), supported: supported}
					handler := newTestHandlerWithRepository(t, generator)
					for turn := range 2 {
						body := fmt.Sprintf(`{%s,"%s":2,"stream":%t}`, protocol.input, protocol.budget, stream)
						response := serveTestRequest(handler, http.MethodPost, protocol.path, body)
						if response.Code != http.StatusOK {
							t.Fatalf("status=%d body=%s", response.Code, response.Body)
						}
						if generator.cachePrompt != supported {
							t.Fatalf("cache enabled=%t support=%t", generator.cachePrompt, supported)
						}
						want := 0
						if supported && turn > 0 {
							want = len(generator.promptIDs) - 1
						}
						payload := response.Body.String()
						if stream {
							payload = ""
							for line := range strings.SplitSeq(response.Body.String(), "\n") {
								if !strings.HasPrefix(line, "data: ") {
									continue
								}
								data := strings.TrimSpace(strings.TrimPrefix(line, "data: "))
								if protocol.path == "/v1/messages" && strings.Contains(data, `"message_start"`) {
									payload = data
									break
								}
								if protocol.path != "/v1/messages" && strings.Contains(data, `"usage"`) {
									payload = data
								}
							}
						}
						var result struct {
							Usage   json.RawMessage `json:"usage"`
							Message struct {
								Usage json.RawMessage `json:"usage"`
							} `json:"message"`
							Response struct {
								Usage json.RawMessage `json:"usage"`
							} `json:"response"`
						}
						if err := json.Unmarshal([]byte(payload), &result); err != nil {
							t.Fatalf("payload=%s: %v", payload, err)
						}
						usage := result.Usage
						if len(result.Message.Usage) > 0 {
							usage = result.Message.Usage
						}
						if len(result.Response.Usage) > 0 {
							usage = result.Response.Usage
						}
						var counts struct {
							PromptTokenDetails responseInputTokenDetails `json:"prompt_tokens_details"`
							InputTokenDetails  responseInputTokenDetails `json:"input_tokens_details"`
							CacheRead          int                       `json:"cache_read_input_tokens"`
							Input              int                       `json:"input_tokens"`
							Prompt             int                       `json:"prompt_tokens"`
						}
						if err := json.Unmarshal(usage, &counts); err != nil {
							t.Fatalf("missing usage: %s (%v)", payload, err)
						}
						cached, total := counts.PromptTokenDetails.CachedTokens, counts.Prompt
						if protocol.path == "/v1/responses" {
							cached, total = counts.InputTokenDetails.CachedTokens, counts.Input
						}
						if protocol.path == "/v1/messages" {
							cached, total = counts.CacheRead, counts.Input+counts.CacheRead
						}
						if cached != want || total != len(generator.promptIDs) {
							t.Fatalf("turn=%d usage=%s want cached=%d total=%d", turn, usage, want, len(generator.promptIDs))
						}
					}
				})
			}
		}
	}
	t.Run("selected-generator", func(t *testing.T) {
		// The inspected model supports reuse, but the selected execution owner
		// (as with continuous generation) does not advertise that capability.
		inspected := &protocolCacheGenerator{recipeInspectorGenerator: responseRecipeGenerator(t, &fakeGenerator{}), supported: true}
		handler := newTestHandler(t, inspected)
		selected := &fakeGenerator{}
		sessions, err := capabilityruntime.NewResidentModelSessionDirector[Generator]("inference", "compiled", 1, selected)
		if err != nil {
			t.Fatal(err)
		}
		handler.sessions = sessions
		response := serveTestRequest(handler, http.MethodPost, "/v1/chat/completions", `{"messages":[{"role":"user","content":"hi"}],"max_tokens":1}`)
		if response.Code != http.StatusOK || selected.cachePrompt {
			t.Fatalf("selected policy: cache=%t response=%s", selected.cachePrompt, response.Body)
		}
	})
	t.Run("empty-stream", func(t *testing.T) {
		generator := &protocolCacheGenerator{recipeInspectorGenerator: responseRecipeGenerator(t, &fakeGenerator{}), supported: true}
		handler := newTestHandler(t, generator)
		response := serveTestRequest(handler, http.MethodPost, "/v1/messages", `{"messages":[{"role":"user","content":"hi"}],"max_tokens":0,"stream":true}`)
		body := response.Body.String()
		if response.Code != http.StatusOK || strings.Count(body, "event: message_start") != 1 ||
			strings.Index(body, "event: message_start") > strings.Index(body, "event: message_delta") ||
			!strings.Contains(body, "event: message_stop") {
			t.Fatalf("empty stream lost its start or terminal event: %s", body)
		}
	})
}
