package server

import (
	"encoding/json"
	"net/http"
	"slices"
	"strings"
	"testing"
)

// TestFrontPageInspect pins inspecting a turn (professional GUI campaign,
// gui-workbench/inspect-turn): a stored turn's run record answers with a
// status from the declared vocabulary, its prompt and completion, and the
// identities behind it (model, trace, receipt); a turn still in flight is
// running; an unknown turn is not found; the materialized chain names the
// response behind every message; and the front page opens the inspectors
// over that record in a side panel without leaving the conversation.
func TestFrontPageInspect(t *testing.T) {
	handler := newTestHandlerWithRepository(t, responseRecipeGenerator(t, &fakeGenerator{}))
	defer handler.Close()
	first := serveTestRequest(handler, http.MethodPost, "/v1/responses", `{"input":"hello there world","max_output_tokens":1}`)
	if first.Code != http.StatusOK {
		t.Fatalf("response status = %d body=%s", first.Code, first.Body.String())
	}
	var stored responsesResponse
	if err := json.Unmarshal(first.Body.Bytes(), &stored); err != nil {
		t.Fatal(err)
	}

	inspected := serveTestRequest(handler, http.MethodGet, "/interactions/inspect?response="+stored.ID, "")
	if inspected.Code != http.StatusOK {
		t.Fatalf("inspect status = %d body=%s", inspected.Code, inspected.Body.String())
	}
	var record turnInspection
	if err := json.Unmarshal(inspected.Body.Bytes(), &record); err != nil {
		t.Fatal(err)
	}
	if record.Status != turnStatusDone || !slices.Equal(record.Statuses, turnStatuses) || record.Prompt != "hello there world" ||
		record.Completion == "" || !record.Model.Valid() || !record.Trace.Valid() || !record.Receipt.Valid() {
		t.Fatalf("inspection = %+v", record)
	}

	handler.inflight.begin("resp_running", handler.config.MaxStoredResponses)
	running := serveTestRequest(handler, http.MethodGet, "/interactions/inspect?response=resp_running", "")
	if err := json.Unmarshal(running.Body.Bytes(), &record); err != nil || record.Status != turnStatusRunning {
		t.Fatalf("running inspection = %+v (%v) body=%s", record, err, running.Body.String())
	}
	for failure, status := range map[string]string{"context canceled": turnStatusCancelled, "deadline exceeded": turnStatusTimeout, "admission refused": turnStatusRefused, "cuda fault": turnStatusError} {
		if got := turnFailureStatus(failure); got != status {
			t.Errorf("turnFailureStatus(%q) = %s, want %s", failure, got, status)
		}
	}
	if missing := serveTestRequest(handler, http.MethodGet, "/interactions/inspect?response=resp_missing", ""); missing.Code != http.StatusNotFound {
		t.Fatalf("missing inspection status = %d", missing.Code)
	}

	messages := serveTestRequest(handler, http.MethodGet, "/interactions/messages?response="+stored.ID, "")
	var chain conversationMessagesResponse
	if err := json.Unmarshal(messages.Body.Bytes(), &chain); err != nil {
		t.Fatal(err)
	}
	if len(chain.Messages) != 2 || chain.Messages[1].Role != "assistant" || chain.Messages[1].Response != stored.ID {
		t.Fatalf("chain = %+v", chain.Messages)
	}

	get := func(path string) string { return serveTestRequest(handler, http.MethodGet, path, "").Body.String() }
	for _, needle := range []string{`id="inspector"`} {
		if !strings.Contains(get("/"), needle) {
			t.Errorf("shell missing %q", needle)
		}
	}
	chat := get("/mod/chat.js")
	for _, needle := range []string{`"/interactions/inspect?response="`, "overgo.embed(", "record.statuses", "assistant.response = latest", ".response = message.response", `"lens"`, `"states"`, `"attention"`, `"model"`, `"vocab"`, `"tensors"`} {
		if !strings.Contains(chat, needle) {
			t.Errorf("chat missing %q", needle)
		}
	}
	if !strings.Contains(get("/composer.js"), "overgo.inspectTurn(message.response)") {
		t.Error("the thread does not offer inspection on an assistant turn")
	}
	if !strings.Contains(get("/boot.js"), "function embed(id, host, seed)") {
		t.Error("boot.js cannot embed a tab's inspector into another host")
	}
	for _, module := range []string{"/mod/analyze_logits.js", "/mod/analyze_states.js", "/mod/analyze_attention.js"} {
		if !strings.Contains(get(module), "overgo.analysisSurface(panel, seed,") {
			t.Errorf("%s does not accept a seeded prompt", module)
		}
	}
	if !strings.Contains(get("/style.css"), ".inspector") {
		t.Error("style lacks the side panel")
	}
}
