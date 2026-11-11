package server

import (
	"net/http"
	"testing"
)

// TestAChatToolRequestMayAskAThinkingDepth holds the chat route to the depth a thinking request asks for beside tools
// (loop-proof/thinking-depth): the local worker always lists its tools, and a chat request that listed any was refused
// for a chat_template_kwargs key beyond enable_thinking, though the prompt's formatting passes reasoning_effort to the
// template and the Responses route sends it beside tools. The request is served, and its depth reaches the template.
func TestAChatToolRequestMayAskAThinkingDepth(t *testing.T) {
	t.Parallel()
	generator := &fakeGenerator{pieces: []string{"done"}}
	handler := newTestHandler(t, generator)
	response := serveTestRequest(handler, http.MethodPost, "/v1/chat/completions", `{"max_tokens":1,`+
		`"tools":[{"type":"function","function":{"name":"weather","description":"forecast","parameters":{"type":"object","properties":{"city":{"type":"string"}}}}}],`+
		`"messages":[{"role":"user","content":"weather?"}],"chat_template_kwargs":{"enable_thinking":true,"reasoning_effort":"low"}}`)
	if response.Code != http.StatusOK {
		t.Fatalf("a tool request asking a depth was answered %d: %s", response.Code, response.Body.String())
	}
	generator.mu.Lock()
	defer generator.mu.Unlock()
	if !generator.chatOptions.EnableThinking || generator.chatOptions.TemplateKwargs["reasoning_effort"] != "low" {
		t.Fatalf("the template was asked %+v; want thinking with reasoning_effort low", generator.chatOptions)
	}
}
