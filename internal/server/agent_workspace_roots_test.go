package server

import (
	"net/http"
	"strings"
	"testing"

	"overgo/internal/agenttool"
	"overgo/internal/artifact"
	"overgo/internal/overgodb"
	"overgo/internal/runrecord"
	"overgo/internal/testutil"
)

// TestAgentWorkspaceRootsReadOnlyTheirDocuments lists agent sessions from
// the interaction documents the response root binds, and nothing else the
// store binds under that root: a stray alias to another document does not
// become a session. The session list and the server's start read the root
// this way; paging the whole catalog took tens of seconds on a real store.
func TestAgentWorkspaceRootsReadOnlyTheirDocuments(t *testing.T) {
	t.Parallel()
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	generator := responseRecipeGenerator(t, &fakeGenerator{})
	stray := testutil.ArtifactID(t, artifact.KindEvidence, "not-an-interaction")
	if _, err := store.Commit(t.Context(), artifact.Batch{
		Key:       "agent/roots-identity",
		Artifacts: []artifact.Descriptor{{ID: generator.description.Identity.Recipe}, {ID: stray}},
		Aliases:   []artifact.AliasBinding{{Name: runrecord.InteractionResponseAliasRoot + "stray-step-1", Target: stray}},
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
	if code := serveTestRequest(handler, http.MethodPost, "/agent/step", `{"session":"alpha","tool":"store.head"}`); code.Code != http.StatusOK {
		t.Fatalf("step status=%d body=%s", code.Code, code.Body.String())
	}
	list := serveTestRequest(handler, http.MethodGet, "/agent/sessions", "")
	body := list.Body.String()
	if list.Code != http.StatusOK || !strings.Contains(body, `"id":"alpha"`) || strings.Contains(body, `"id":"stray"`) {
		t.Fatalf("session list status=%d body=%s; want alpha and no stray", list.Code, body)
	}
}
