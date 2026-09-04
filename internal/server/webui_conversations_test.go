package server

import (
	"encoding/json"
	"net/http"
	"strings"
	"testing"
)

// TestFrontPageConversations pins durable conversations (professional GUI
// campaign, gui-conversations/durable-sessions): a stored response chain
// is listed as one conversation with a derived title, its messages
// materialize for resumption, a label renames and archives it as a new
// record, a completed turn replays through the follow route, and the
// client keeps no conversation state of its own: chat rides the Responses
// API with previous_response_id, reattaches through follow, and the rail
// lists conversations from the server.
func TestFrontPageConversations(t *testing.T) {
	handler := newTestHandlerWithRepository(t, responseRecipeGenerator(t, &fakeGenerator{}))
	defer handler.Close()
	first := serveTestRequest(handler, http.MethodPost, "/v1/responses", `{"input":"hello there world","max_output_tokens":1}`)
	if first.Code != http.StatusOK {
		t.Fatalf("first response status = %d body=%s", first.Code, first.Body.String())
	}
	var initial responsesResponse
	if err := json.Unmarshal(first.Body.Bytes(), &initial); err != nil {
		t.Fatal(err)
	}
	second := serveTestRequest(handler, http.MethodPost, "/v1/responses",
		`{"input":"next","max_output_tokens":1,"previous_response_id":"`+initial.ID+`"}`)
	if second.Code != http.StatusOK {
		t.Fatalf("second response status = %d body=%s", second.Code, second.Body.String())
	}
	var continued responsesResponse
	if err := json.Unmarshal(second.Body.Bytes(), &continued); err != nil {
		t.Fatal(err)
	}

	listed := serveTestRequest(handler, http.MethodGet, "/interactions", "")
	if listed.Code != http.StatusOK {
		t.Fatalf("list status = %d body=%s", listed.Code, listed.Body.String())
	}
	var list conversationListResponse
	if err := json.Unmarshal(listed.Body.Bytes(), &list); err != nil {
		t.Fatal(err)
	}
	if len(list.Conversations) != 1 || list.Conversations[0].Root != initial.ID || list.Conversations[0].Latest != continued.ID ||
		list.Conversations[0].Turns != 2 || list.Conversations[0].Title != "hello there world" {
		t.Fatalf("conversations = %+v", list.Conversations)
	}

	messages := serveTestRequest(handler, http.MethodGet, "/interactions/messages?response="+continued.ID, "")
	if messages.Code != http.StatusOK {
		t.Fatalf("messages status = %d body=%s", messages.Code, messages.Body.String())
	}
	var chain conversationMessagesResponse
	if err := json.Unmarshal(messages.Body.Bytes(), &chain); err != nil {
		t.Fatal(err)
	}
	if chain.Root != initial.ID || len(chain.Messages) != 4 || chain.Messages[0].Content != "hello there world" || chain.Messages[2].Content != "next" {
		t.Fatalf("chain = %+v", chain)
	}

	labelled := serveTestRequest(handler, http.MethodPost, "/interactions/label", `{"root":"`+initial.ID+`","title":"Greeting","archived":true}`)
	if labelled.Code != http.StatusOK {
		t.Fatalf("label status = %d body=%s", labelled.Code, labelled.Body.String())
	}
	relisted := serveTestRequest(handler, http.MethodGet, "/interactions", "")
	if err := json.Unmarshal(relisted.Body.Bytes(), &list); err != nil {
		t.Fatal(err)
	}
	if len(list.Conversations) != 1 || list.Conversations[0].Title != "Greeting" || !list.Conversations[0].Archived {
		t.Fatalf("labelled conversations = %+v", list.Conversations)
	}
	if missing := serveTestRequest(handler, http.MethodPost, "/interactions/label", `{"root":"resp_999","title":"x"}`); missing.Code != http.StatusNotFound {
		t.Errorf("label of an unknown conversation status = %d", missing.Code)
	}

	follow := serveTestRequest(handler, http.MethodGet, "/interactions/follow?response="+continued.ID, "")
	if follow.Code != http.StatusOK || !strings.Contains(follow.Body.String(), "event: response.output_text.delta") ||
		!strings.Contains(follow.Body.String(), "event: response.completed") {
		t.Fatalf("follow of a durable turn = %d %s", follow.Code, follow.Body.String())
	}
	if unknown := serveTestRequest(handler, http.MethodGet, "/interactions/follow?response=resp_999", ""); unknown.Code != http.StatusNotFound {
		t.Errorf("follow of an unknown turn status = %d", unknown.Code)
	}

	sources := webuiJavaScript(t)
	chat := sources["mod/chat.js"]
	for _, needle := range []string{`"/v1/responses"`, "previous_response_id", `"/interactions/follow?response="`, `"/interactions/messages?response="`, "overgo.streams.responses(", "INFLIGHT_STORAGE", "context meter", "instructions"} {
		if !strings.Contains(chat, needle) {
			t.Errorf("chat.js lacks %s", needle)
		}
	}
	if strings.Contains(chat, `"/v1/chat/completions"`) {
		t.Error("chat.js still streams the stateless chat completion; conversations ride the Responses API")
	}
	boot := sources["boot.js"]
	for _, needle := range []string{`"/interactions"`, `"/interactions/label"`, "conversation-list", "openConversation"} {
		if !strings.Contains(boot, needle) {
			t.Errorf("boot.js lacks the conversation rail (%s)", needle)
		}
	}
	if !strings.Contains(sources["composer.js"], "response.output_text.delta") {
		t.Error("composer.js lacks the Responses stream adapter")
	}
}
