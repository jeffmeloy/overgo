package server

import (
	"context"
	"net/http"
	"strings"
	"testing"

	"overgo/internal/agenttool"
	"overgo/internal/artifact"
	"overgo/internal/overgodb"
)

// TestAgentSessionList pins durable session listing: two sessions with
// different step counts list with their recorded steps and inspection
// state, and a FRESH handler over the same store lists the same
// sessions -- the list is the ledger's memory, not the process's.
func TestAgentSessionList(t *testing.T) {
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	generator := responseRecipeGenerator(t, &fakeGenerator{})
	if _, err := store.Commit(context.Background(), artifact.Batch{
		Key:       "agent/session-list-identity",
		Artifacts: []artifact.Descriptor{{ID: generator.description.Identity.Recipe}},
	}); err != nil {
		t.Fatal(err)
	}
	manuals, err := agenttool.StandardManuals()
	if err != nil {
		t.Fatal(err)
	}
	if _, err := agenttool.PublishManualCatalog(context.Background(), store, manuals); err != nil {
		t.Fatal(err)
	}
	first, err := New(Config{ModelID: testModelID, MaxTokens: testMaxTokens, Repository: store}, generator)
	if err != nil {
		t.Fatal(err)
	}
	for _, step := range []string{`{"session":"alpha","tool":"store.head"}`, `{"session":"alpha","tool":"store.head"}`, `{"session":"beta","tool":"store.head"}`} {
		if code := serveTestRequest(first, http.MethodPost, "/agent/step", step); code.Code != http.StatusOK {
			t.Fatalf("step status=%d body=%s", code.Code, code.Body.String())
		}
	}
	list := serveTestRequest(first, http.MethodGet, "/agent/sessions", "")
	body := list.Body.String()
	if list.Code != http.StatusOK ||
		!strings.Contains(body, `"id":"alpha"`) || !strings.Contains(body, `"id":"beta"`) ||
		!strings.Contains(body, `"steps":2`) || !strings.Contains(body, `"steps":1`) ||
		!strings.Contains(body, `"inspected":true`) {
		t.Fatalf("session list status=%d body=%s", list.Code, body)
	}
	if err := first.Close(); err != nil {
		t.Fatal(err)
	}
	restarted, err := New(Config{ModelID: testModelID, MaxTokens: testMaxTokens, Repository: store}, generator)
	if err != nil {
		t.Fatal(err)
	}
	defer restarted.Close()
	survived := serveTestRequest(restarted, http.MethodGet, "/agent/sessions", "")
	if survived.Code != http.StatusOK ||
		!strings.Contains(survived.Body.String(), `"id":"alpha"`) ||
		!strings.Contains(survived.Body.String(), `"steps":2`) {
		t.Fatalf("restarted session list status=%d body=%s", survived.Code, survived.Body.String())
	}
}
