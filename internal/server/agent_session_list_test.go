package server

import (
	"net/http"
	"strings"
	"testing"

	"overgo/internal/agenttool"
	"overgo/internal/artifact"
	"overgo/internal/overgodb"
	"overgo/internal/runrecord"
	"overgo/internal/testcheck"
	"overgo/internal/testutil"
)

// TestAgentSessionList pins durable session listing: sessions list from the
// interaction documents the response root binds, with their recorded steps
// and inspection state, and a stray alias under that root to another
// document is no session (paging the whole catalog took tens of seconds on
// a real store). A FRESH handler over the same store lists the same sessions
// and resumes one with its step count and inspection state -- the list is
// the ledger's memory, not the process's, so a restart neither resets the
// mutation gate nor forks the interaction chain.
func TestAgentSessionList(t *testing.T) {
	t.Parallel()
	store, err := overgodb.Open(t.TempDir())
	testcheck.NoError(t, err)
	defer store.Close()
	generator := responseRecipeGenerator(t, &fakeGenerator{})
	stray := testutil.ArtifactID(t, artifact.KindEvidence, "not-an-interaction")
	if _, err := store.Commit(t.Context(), artifact.Batch{
		Key:       "agent/session-list-identity",
		Artifacts: []artifact.Descriptor{{ID: generator.description.Identity.Recipe}, {ID: stray}},
		Aliases:   []artifact.AliasBinding{{Name: runrecord.InteractionResponseAliasRoot + "stray-step-1", Target: stray}},
	}); err != nil {
		t.Fatal(err)
	}
	manuals, err := agenttool.StandardManuals()
	testcheck.NoError(t, err)
	if _, err := agenttool.PublishManualCatalog(t.Context(), store, manuals); err != nil {
		t.Fatal(err)
	}
	first, err := New(Config{ModelID: testModelID, MaxTokens: testMaxTokens, Repository: store}, generator)
	testcheck.NoError(t, err)
	for _, step := range []string{`{"session":"alpha","tool":"store.head"}`, `{"session":"alpha","tool":"store.head"}`, `{"session":"beta","tool":"store.head"}`} {
		if code := serveTestRequest(first, http.MethodPost, "/agents/step", step); code.Code != http.StatusOK {
			t.Fatalf("step status=%d body=%s", code.Code, code.Body.String())
		}
	}
	list := serveTestRequest(first, http.MethodGet, "/agents/sessions", "")
	body := list.Body.String()
	if list.Code != http.StatusOK ||
		!strings.Contains(body, `"id":"alpha"`) || !strings.Contains(body, `"id":"beta"`) ||
		!strings.Contains(body, `"steps":2`) || !strings.Contains(body, `"steps":1`) ||
		!strings.Contains(body, `"inspected":true`) || strings.Contains(body, `"id":"stray"`) {
		t.Fatalf("session list status=%d body=%s", list.Code, body)
	}
	testcheck.NoError(t, first.Close())
	restarted, err := New(Config{ModelID: testModelID, MaxTokens: testMaxTokens, Repository: store}, generator)
	testcheck.NoError(t, err)
	defer restarted.Close()
	survived := serveTestRequest(restarted, http.MethodGet, "/agents/sessions", "")
	if survived.Code != http.StatusOK ||
		!strings.Contains(survived.Body.String(), `"id":"alpha"`) ||
		!strings.Contains(survived.Body.String(), `"steps":2`) {
		t.Fatalf("restarted session list status=%d body=%s", survived.Code, survived.Body.String())
	}
	resumed := serveTestRequest(restarted, http.MethodPost, "/agents/step", `{"session":"alpha","tool":"store.head"}`)
	if resumed.Code != http.StatusOK ||
		!strings.Contains(resumed.Body.String(), `"steps":3`) ||
		!strings.Contains(resumed.Body.String(), `"inspected":true`) {
		t.Fatalf("resumed step status=%d body=%s", resumed.Code, resumed.Body.String())
	}
}

// TestListingAgentSessionsRestoresNone: a restarted server lists what the first one listed, and no session was built to do
// it, so a list on a long-lived store costs a read of each session's end.
func TestListingAgentSessionsRestoresNone(t *testing.T) {
	t.Parallel()
	store, err := overgodb.Open(t.TempDir())
	testcheck.NoError(t, err)
	defer store.Close()
	generator := responseRecipeGenerator(t, &fakeGenerator{})
	if _, err := store.Commit(t.Context(), artifact.Batch{
		Key: "agent/session-list-restores-none", Artifacts: []artifact.Descriptor{{ID: generator.description.Identity.Recipe}},
	}); err != nil {
		t.Fatal(err)
	}
	manuals, err := agenttool.StandardManuals()
	testcheck.NoError(t, err)
	if _, err := agenttool.PublishManualCatalog(t.Context(), store, manuals); err != nil {
		t.Fatal(err)
	}
	first, err := New(Config{ModelID: testModelID, MaxTokens: testMaxTokens, Repository: store}, generator)
	testcheck.NoError(t, err)
	defer first.Close()
	for _, step := range []string{`{"session":"alpha","tool":"store.head"}`, `{"session":"alpha","tool":"store.head"}`, `{"session":"beta","tool":"store.head"}`} {
		if response := serveTestRequest(first, http.MethodPost, "/agents/step", step); response.Code != http.StatusOK {
			t.Fatalf("step status=%d body=%s", response.Code, response.Body.String())
		}
	}
	listed := serveTestRequest(first, http.MethodGet, "/agents/sessions", "")
	restarted, err := New(Config{ModelID: testModelID, MaxTokens: testMaxTokens, Repository: store}, generator)
	testcheck.NoError(t, err)
	defer restarted.Close()
	again := serveTestRequest(restarted, http.MethodGet, "/agents/sessions", "")
	if again.Code != http.StatusOK || again.Body.String() != listed.Body.String() {
		t.Fatalf("the restarted list = %d %s, the first = %s", again.Code, again.Body.String(), listed.Body.String())
	}
	if built := len(restarted.agentSessions.sessions); built != 0 {
		t.Fatalf("a list built %d sessions", built)
	}
}
