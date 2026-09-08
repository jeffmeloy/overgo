package server

import (
	"encoding/json"
	"net/http"
	"net/url"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/overgodb"
	"overgo/internal/runrecord"
	"overgo/internal/testutil"
)

// A real store with more leaves than one configured page, duplicate titles,
// a parent chain longer than a page, and an independently identified model.
func conversationHistoryFixture(t testing.TB) (*Handler, string, string) {
	t.Helper()
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = store.Close() })
	h := newTestHandlerForRepository(t, store, responseRecipeGenerator(t, &fakeGenerator{pieces: []string{"Stored answer"}}))
	h.config.MaxStoredResponses = 2
	respond := func(input, previous string) string {
		body, err := json.Marshal(map[string]any{"input": input, "previous_response_id": previous, "max_output_tokens": 1})
		if err != nil {
			t.Fatal(err)
		}
		response := serveTestRequest(h, http.MethodPost, "/v1/responses", string(body))
		var result responsesResponse
		if response.Code != http.StatusOK || json.Unmarshal(response.Body.Bytes(), &result) != nil {
			t.Fatalf("seed response: %d %s", response.Code, response.Body)
		}
		return result.ID
	}
	root := respond("Older title", "")
	respond("Duplicate", "")
	respond("Duplicate", "")
	respond("Recent title", "")
	latest := root
	for range 4 {
		latest = respond("Next history turn", latest)
	}
	otherRecipe := testutil.ArtifactID(t, artifact.KindRecipe, "history-other-recipe")
	otherModel := testutil.ArtifactID(t, artifact.KindModel, "history-other-model")
	if _, err := artifact.CommitBatch(t.Context(), store, artifact.Batch{Key: "history/other", Artifacts: []artifact.Descriptor{{ID: otherRecipe}, {ID: otherModel}}}); err != nil {
		t.Fatal(err)
	}
	if _, err := runrecord.PublishInteraction(t.Context(), store, runrecord.Interaction{Response: "resp_other", Recipe: otherRecipe, Model: otherModel, Node: "respond"},
		[]runrecord.InteractionMessage{{Role: "user", Content: "Other model history"}, {Role: "assistant", Content: "An answer from another model"}}, runrecord.OutcomeSucceeded); err != nil {
		t.Fatal(err)
	}
	return h, root, latest
}

func TestConversationHistoryPaging(t *testing.T) {
	h, root, latest := conversationHistoryFixture(t)
	list := func(path string) conversationListResponse {
		t.Helper()
		response := serveTestRequest(h, http.MethodGet, path, "")
		var result conversationListResponse
		if response.Code != http.StatusOK || json.Unmarshal(response.Body.Bytes(), &result) != nil {
			t.Fatalf("list: %d %s", response.Code, response.Body)
		}
		return result
	}
	first := list("/interactions?view=active")
	if len(first.Conversations) != 2 || first.Next == "" {
		t.Fatalf("first page: %+v", first)
	}
	seen := map[string]bool{}
	page := first
	for {
		for _, item := range page.Conversations {
			if seen[item.Latest] {
				t.Fatalf("duplicate leaf: %s", item.Latest)
			}
			seen[item.Latest] = true
			if item.Latest == latest && (item.Root != root || item.Turns != 5 || item.Title != "Older title") {
				t.Fatalf("page-truncated ancestry: %+v", item)
			}
		}
		if page.Next == "" {
			break
		}
		page = list("/interactions?view=active&cursor=" + url.QueryEscape(page.Next))
	}
	if len(seen) != 5 || !seen[latest] {
		t.Fatalf("history lost leaves: %v", seen)
	}
	if matches := list("/interactions?q=duplicate&view=active"); len(matches.Conversations) != 2 {
		t.Fatalf("duplicate-title search: %+v", matches)
	}
	if empty := list("/interactions?q=absent&view=active"); len(empty.Conversations) != 0 || empty.Next != "" {
		t.Fatalf("empty search: %+v", empty)
	}
	for _, path := range []string{"/interactions?limit=0", "/interactions?view=wrong", "/interactions?cursor=bad", "/interactions?view=active&q=changed&cursor=" + url.QueryEscape(first.Next)} {
		if response := serveTestRequest(h, http.MethodGet, path, ""); response.Code != http.StatusBadRequest {
			t.Errorf("invalid query %s: %d", path, response.Code)
		}
	}
	label := `{"root":"` + root + `","title":"Renamed","archived":true}`
	for range 2 {
		if response := serveTestRequest(h, http.MethodPost, "/interactions/label", label); response.Code != http.StatusOK {
			t.Fatalf("idempotent label: %d %s", response.Code, response.Body)
		}
	}
	if response := serveTestRequest(h, http.MethodGet, "/interactions?view=active&cursor="+url.QueryEscape(first.Next), ""); response.Code != http.StatusConflict {
		t.Fatalf("stale page accepted: %d", response.Code)
	}
	archived := list("/interactions?view=archived")
	if len(archived.Conversations) != 1 || archived.Conversations[0].Root != root {
		t.Fatalf("archive: %+v", archived)
	}
	if response := serveTestRequest(h, http.MethodPost, "/interactions/label", `{"root":"`+root+`","title":"Renamed","archived":false}`); response.Code != http.StatusOK {
		t.Fatal(response.Body)
	}
	if restored := list("/interactions?view=active&q=renamed"); len(restored.Conversations) != 1 || restored.Conversations[0].Latest != latest {
		t.Fatalf("restore: %+v", restored)
	}
	for range 2 {
		for _, archived := range []bool{true, false} {
			body, err := json.Marshal(map[string]any{"root": root, "title": "Renamed", "archived": archived})
			if err != nil {
				t.Fatal(err)
			}
			if response := serveTestRequest(h, http.MethodPost, "/interactions/label", string(body)); response.Code != http.StatusOK {
				t.Fatal(response.Body)
			}
			label, found, err := resolveConversationLabel(t.Context(), h.repository, root)
			if err != nil || !found || label.Archived != archived {
				t.Fatalf("repeated archive/restore: %+v %v", label, err)
			}
		}
	}
	for _, id := range []string{latest, "resp_other"} {
		response := serveTestRequest(h, http.MethodGet, "/interactions/messages?response="+id, "")
		var messages conversationMessagesResponse
		if response.Code != http.StatusOK || json.Unmarshal(response.Body.Bytes(), &messages) != nil || len(messages.Messages) == 0 || !messages.Model.Valid() || !messages.Recipe.Valid() {
			t.Fatalf("read stored transcript: %d %s", response.Code, response.Body)
		}
		if id == latest && (len(messages.Messages) != 10 || messages.Root != root) {
			t.Fatalf("materialized chain: %+v", messages)
		}
	}
}

func TestConversationHistoryRefresh(t *testing.T) {
	root := t.TempDir()
	writer, err := overgodb.Open(root)
	if err != nil {
		t.Fatal(err)
	}
	defer writer.Close()
	generator := responseRecipeGenerator(t, &fakeGenerator{pieces: []string{"External answer"}})
	producer := newTestHandlerForRepository(t, writer, generator)
	reader, err := overgodb.Open(root)
	if err != nil {
		t.Fatal(err)
	}
	defer reader.Close()
	consumer := newTestHandlerForRepository(t, reader, generator)
	response := serveTestRequest(producer, http.MethodPost, "/v1/responses", `{"input":"External conversation","max_output_tokens":1}`)
	var turn responsesResponse
	if response.Code != http.StatusOK || json.Unmarshal(response.Body.Bytes(), &turn) != nil {
		t.Fatal(response.Body)
	}
	// The consumer was opened before this turn was committed by another handle.
	response = serveTestRequest(consumer, http.MethodGet, "/interactions/messages?response="+turn.ID, "")
	var transcript conversationMessagesResponse
	if response.Code != http.StatusOK || json.Unmarshal(response.Body.Bytes(), &transcript) != nil || len(transcript.Messages) != 2 {
		t.Fatalf("external transcript refresh: %d %s", response.Code, response.Body)
	}
	for _, label := range []struct {
		handler *Handler
		title   string
	}{{producer, "External rename"}, {consumer, "Local rename"}} {
		response = serveTestRequest(label.handler, http.MethodPost, "/interactions/label", `{"root":"`+turn.ID+`","title":"`+label.title+`"}`)
		if response.Code != http.StatusOK {
			t.Fatalf("external label refresh: %d %s", response.Code, response.Body)
		}
	}
	response = serveTestRequest(producer, http.MethodGet, "/interactions?view=active&q=Local", "")
	var listing conversationListResponse
	if response.Code != http.StatusOK || json.Unmarshal(response.Body.Bytes(), &listing) != nil || len(listing.Conversations) != 1 || listing.Conversations[0].Title != "Local rename" {
		t.Fatalf("external list refresh: %d %s", response.Code, response.Body)
	}
}
