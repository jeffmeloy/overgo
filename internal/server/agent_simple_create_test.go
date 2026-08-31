package server

import (
	"net/http"
	"strings"
	"testing"

	"overgo/internal/agenttool"
	"overgo/internal/artifact"
	"overgo/internal/overgodb"
)

// TestAgentSimpleCreate pins one-step creation: a name, plain
// instructions, and tool NAMES become a committed prompt artifact, a
// definition bound to the served recipe and the exact tool manuals,
// and an ACTIVE agent -- no identity typed by the operator. The
// created agent immediately admits a step on its granted tool, and an
// unregistered tool name refuses creation outright.
func TestAgentSimpleCreate(t *testing.T) {
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	generator := responseRecipeGenerator(t, &fakeGenerator{})
	if _, err := store.Commit(t.Context(), artifact.Batch{
		Key:       "agent/simple-create-identity",
		Artifacts: []artifact.Descriptor{{ID: generator.description.Identity.Recipe}},
	}); err != nil {
		t.Fatal(err)
	}
	manuals, err := agenttool.StandardManuals()
	if err != nil {
		t.Fatal(err)
	}
	if _, err := agenttool.PublishManualCatalog(t.Context(), store, manuals); err != nil {
		t.Fatal(err)
	}
	handler, err := New(Config{ModelID: testModelID, MaxTokens: testMaxTokens, Repository: store}, generator)
	if err != nil {
		t.Fatal(err)
	}
	defer handler.Close()

	refused := serveTestRequest(handler, http.MethodPost, "/agents/create",
		`{"name":"helper","instructions":"be helpful","tools":["probe.absent"]}`)
	if refused.Code != http.StatusUnprocessableEntity {
		t.Fatalf("unregistered tool create status=%d body=%s", refused.Code, refused.Body.String())
	}

	created := serveTestRequest(handler, http.MethodPost, "/agents/create",
		`{"name":"helper","instructions":"Read the store before answering.","tools":["store.head"]}`)
	if created.Code != http.StatusOK || !strings.Contains(created.Body.String(), `"state":"active"`) {
		t.Fatalf("create status=%d body=%s", created.Code, created.Body.String())
	}

	inventory := serveTestRequest(handler, http.MethodGet, "/agents", "")
	if inventory.Code != http.StatusOK || !strings.Contains(inventory.Body.String(), `"name":"helper"`) {
		t.Fatalf("inventory status=%d body=%s", inventory.Code, inventory.Body.String())
	}

	step := serveTestRequest(handler, http.MethodPost, "/agents/step",
		`{"agent":"helper","session":"first","tool":"store.head"}`)
	if step.Code != http.StatusOK || !strings.Contains(step.Body.String(), `"steps":1`) {
		t.Fatalf("created agent step status=%d body=%s", step.Code, step.Body.String())
	}
}
