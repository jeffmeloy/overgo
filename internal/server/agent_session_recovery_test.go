package server

import (
	"encoding/json"
	"net/http"
	"strings"
	"testing"

	"overgo/internal/overgodb"
)

// TestAgentThreadSurvivesRestart: a session's answered chat turns are kept
// beside its tool steps, and a restarted server reads the session's thread
// back in order, its step count resumed; a session whose name extends this
// one's stays out of it.
func TestAgentThreadSurvivesRestart(t *testing.T) {
	t.Parallel()
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	first := newAgentWorkspaceFixture(t, store, nil, nil)
	activateDefinitionFromAPI(t, first.handler, publishAgentFromAPI(t, first, nil, nil), "/agents/activate")
	chat := func(fixture agentWorkspaceFixture, session string, messages ...string) {
		t.Helper()
		turns := make([]map[string]string, len(messages))
		for index, message := range messages {
			turns[index] = map[string]string{"role": "user", "content": message}
		}
		body, err := json.Marshal(map[string]any{"agent": "research-agent", "session": session, "messages": turns})
		if err != nil {
			t.Fatal(err)
		}
		if answered := serveTestRequest(fixture.handler, http.MethodPost, "/agents/chat", string(body)); answered.Code != http.StatusOK || !strings.Contains(answered.Body.String(), `"choices"`) {
			t.Fatalf("agent chat status=%d body=%s", answered.Code, answered.Body.String())
		}
	}
	chat(first, "s1", "hello")
	if step := serveTestRequest(first.handler, http.MethodPost, "/agents/step", `{"agent":"research-agent","session":"s1","tool":"store.head"}`); step.Code != http.StatusOK || !strings.Contains(step.Body.String(), `"steps":1`) {
		t.Fatalf("step status=%d body=%s", step.Code, step.Body.String())
	}
	chat(first, "s1", "hello", "again")
	chat(first, "s1-2", "another session")
	if err := first.handler.Close(); err != nil {
		t.Fatal(err)
	}

	// A restarted server over the same store reads the thread back.
	second := newAgentWorkspaceFixture(t, store, nil, nil)
	read := serveTestRequest(second.handler, http.MethodGet, "/agents/thread?agent=research-agent&session=s1", "")
	var thread agentThreadResponse
	if err := json.Unmarshal(read.Body.Bytes(), &thread); read.Code != http.StatusOK || err != nil {
		t.Fatalf("thread status=%d body=%s", read.Code, read.Body.String())
	}
	kinds := make([]string, len(thread.Entries))
	for index, entry := range thread.Entries {
		kinds[index] = entry.Kind
	}
	if strings.Join(kinds, ",") != "turn,step,turn" || thread.Steps != 1 || thread.Session != "research-agent:s1" {
		t.Fatalf("thread = %+v", thread)
	}
	if thread.Entries[0].User != "hello" || thread.Entries[0].Assistant == "" || thread.Entries[2].User != "again" ||
		thread.Entries[1].Tool != "store.head" || thread.Entries[1].Result == "" || thread.Entries[1].Error {
		t.Fatalf("thread entries = %+v", thread.Entries)
	}
	if step := serveTestRequest(second.handler, http.MethodPost, "/agents/step", `{"agent":"research-agent","session":"s1","tool":"store.head"}`); step.Code != http.StatusOK || !strings.Contains(step.Body.String(), `"steps":2`) {
		t.Fatalf("resumed step status=%d body=%s", step.Code, step.Body.String())
	}
}
