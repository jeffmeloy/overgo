package server

import (
	"net/http"
	"strings"
	"testing"
)

// TestFrontPageAgent holds agent mode's server side: a tool step and the
// session list report the session's step bound beside its steps. The agent
// thread leg drives agent mode on the page; this bans the agent tab from
// calling the approval route itself instead of riding the shared tool-step
// surface.
func TestFrontPageAgent(t *testing.T) {
	t.Parallel()
	fixture := newAgentWorkspaceFixture(t, nil, nil, nil)
	defer fixture.store.Close()
	activateDefinitionFromAPI(t, fixture.handler, publishAgentFromAPI(t, fixture, nil, nil), "/agents/activate")
	step := serveTestRequest(fixture.handler, http.MethodPost, "/agents/step",
		`{"agent":"research-agent","session":"front","tool":"store.head"}`)
	if step.Code != http.StatusOK || !strings.Contains(step.Body.String(), `"steps":1`) || !strings.Contains(step.Body.String(), `"bound":`) {
		t.Fatalf("agent step status=%d body=%s", step.Code, step.Body.String())
	}
	sessions := serveTestRequest(fixture.handler, http.MethodGet, "/agents/sessions", "")
	if sessions.Code != http.StatusOK || !strings.Contains(sessions.Body.String(), `"bound":`) {
		t.Fatalf("session list status=%d body=%s", sessions.Code, sessions.Body.String())
	}

	agent := serveTestRequest(fixture.handler, http.MethodGet, "/mod/agent.js", "").Body.String()
	if strings.Contains(agent, `"/agents/approval"`) {
		t.Error("the agent tab calls the approval route instead of riding the shared tool-step surface")
	}
}
