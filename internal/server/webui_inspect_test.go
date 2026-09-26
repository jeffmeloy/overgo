package server

import (
	"encoding/json"
	"net/http"
	"slices"
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
	t.Parallel()
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
	if record.Status != turnStatusIncomplete || record.IncompleteDetails == nil || record.IncompleteDetails.Reason != "max_output_tokens" || !slices.Equal(record.Statuses, turnStatuses) || record.Prompt != "hello there world" ||
		record.Completion == "" || !record.Model.Valid() || !record.Trace.Valid() || !record.Receipt.Valid() {
		t.Fatalf("inspection = %+v", record)
	}

	handler.inflight.begin("resp_running", handler.config.MaxStoredResponses, nil)
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
}
