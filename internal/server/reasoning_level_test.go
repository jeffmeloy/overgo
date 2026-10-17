package server

import (
	"context"
	"encoding/json"
	"net/http"
	"slices"
	"sync"
	"testing"

	"overgo/internal/inference"
	"overgo/internal/tokenizer"
)

// reasoningTemplate reads the thinking switch and the reasoning level, and
// renders high and xhigh alike, in letters the hermetic vocabulary holds.
const reasoningTemplate = "{% if enable_thinking %}d{% endif %}" +
	"{% if reasoning_effort == 'high' or reasoning_effort == 'xhigh' %}cc{% elif reasoning_effort == 'medium' %}ca{% elif reasoning_effort %}c{% endif %}" +
	joinedMessagesTemplate

// promptRecorder serves a runner and keeps each prompt it generates from.
type promptRecorder struct {
	*inference.Runner
	mu      sync.Mutex
	prompts []string
}

func (recorder *promptRecorder) Generate(ctx context.Context, prompt string, options inference.GenerateOptions) ([]tokenizer.TokenID, string, error) {
	recorder.mu.Lock()
	recorder.prompts = append(recorder.prompts, prompt)
	recorder.mu.Unlock()
	return recorder.Runner.Generate(ctx, prompt, options)
}

// TestReasoningLevelDeclared: the capability document offers a reasoning
// control only when the served model's chat template reads one -- Off when
// the thinking switch changes the prompt, and each level whose prompt no
// earlier level renders -- and a turn's chosen level reaches the template.
func TestReasoningLevelDeclared(t *testing.T) {
	t.Parallel()
	offered := func(handler *Handler) *workspaceReasoning {
		t.Helper()
		var manifest workspaceManifestResponse
		if err := json.Unmarshal(serveTestRequest(handler, http.MethodGet, "/workspace/manifest", "").Body.Bytes(), &manifest); err != nil || manifest.Model == nil {
			t.Fatalf("manifest = %+v, %v", manifest, err)
		}
		return manifest.Model.Reasoning
	}
	recorder := &promptRecorder{Runner: openTemplateRunner(t, 64, reasoningTemplate)}
	handler := newTestHandler(t, recorder)
	if reasoning := offered(handler); reasoning == nil || !reasoning.Off || !slices.Equal(reasoning.Levels, []string{"low", "medium", "high"}) {
		t.Fatalf("offered reasoning = %+v", reasoning)
	}
	turn := func(effort string) string {
		t.Helper()
		response := serveTestRequest(handler, http.MethodPost, "/v1/responses",
			`{"model":"`+testModelID+`","input":"ab","temperature":0,"max_output_tokens":1,"reasoning":{"effort":"`+effort+`"}}`)
		if response.Code != http.StatusOK {
			t.Fatalf("effort %s: %d %s", effort, response.Code, response.Body.String())
		}
		recorder.mu.Lock()
		defer recorder.mu.Unlock()
		return recorder.prompts[len(recorder.prompts)-1]
	}
	if prompt := turn("high"); prompt != "dccab" {
		t.Fatalf("a high turn's prompt = %q", prompt)
	}
	if prompt := turn("medium"); prompt != "dcaab" {
		t.Fatalf("a medium turn's prompt = %q", prompt)
	}
	if prompt := turn("none"); prompt != "ab" {
		t.Fatalf("an off turn's prompt = %q", prompt)
	}
	for _, thinking := range []string{"", `,"thinking":{"type":"disabled"}`} {
		response := serveTestRequest(handler, http.MethodPost, "/v1/messages",
			`{"model":"`+testModelID+`","messages":[{"role":"user","content":"ab"}],"temperature":0,"max_tokens":1`+thinking+`}`)
		if response.Code != http.StatusOK {
			t.Fatalf("messages thinking %q: %d %s", thinking, response.Code, response.Body.String())
		}
		recorder.mu.Lock()
		prompt := recorder.prompts[len(recorder.prompts)-1]
		recorder.mu.Unlock()
		if prompt != "ab" {
			t.Fatalf("messages thinking %q rendered %q, want no thinking", thinking, prompt)
		}
	}
	if reasoning := offered(newTestHandler(t, openSessionRunner(t, 64))); reasoning != nil {
		t.Fatalf("a template reading no reasoning offered %+v", reasoning)
	}
}
