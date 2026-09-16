package server

import (
	"net/http"
	"strings"
	"testing"
)

// TestFrontPageAgent pins agent mode inside the conversation (professional
// GUI campaign, gui-workbench/agent-in-thread): a tool step and the session
// list report the session's step bound beside its steps, the front page
// offers an agent mode over the active agent definitions and runs the same
// thread through /agents/chat with the shared tool-step surface, that
// surface is one piece composer.js owns (review a grant's binding, execute
// an inspection, approve a mutation bound to the previewed operation
// identity, render results as tool cards, show steps against the bound),
// and the workbench agent tab rides the same piece.
func TestFrontPageAgent(t *testing.T) {
	fixture := newAgentWorkspaceFixture(t, nil, nil, nil)
	defer fixture.store.Close()
	activateDefinitionFromAPI(t, fixture.handler, publishAgentFromAPI(t, fixture, nil, nil), "/agents/activate")
	step := serveTestRequest(fixture.handler, http.MethodPost, "/agents/step",
		`{"agent":"research-agent","session":"front","tool":"store.head"}`)
	if step.Code != http.StatusOK || !strings.Contains(step.Body.String(), `"steps":1`) || !strings.Contains(step.Body.String(), `"bound":`) {
		t.Fatalf("agent step status=%d body=%s", step.Code, step.Body.String())
	}
	sessions := serveTestRequest(fixture.handler, http.MethodGet, "/agent/sessions", "")
	if sessions.Code != http.StatusOK || !strings.Contains(sessions.Body.String(), `"bound":`) {
		t.Fatalf("session list status=%d body=%s", sessions.Code, sessions.Body.String())
	}

	get := func(path string) string {
		return serveTestRequest(fixture.handler, http.MethodGet, path, "").Body.String()
	}
	composer := get("/composer.js")
	for _, needle := range []string{
		"function toolStep(", `"/agents/approval"`, `"/agents/step"`, "request.approval = previewed && previewed.operation",
		"result.bound - result.steps", "options.thread().toolCard(", `class: "card approval"`, "options.onMode(modeSelect.value)",
	} {
		if !strings.Contains(composer, needle) {
			t.Errorf("composer missing %q", needle)
		}
	}
	chat := get("/mod/chat.js")
	for _, needle := range []string{`"/agents"`, `"/agent/tools"`, `"/agents/chat"`, "overgo.toolStep(", `mode === "agent"`, "onMode:", "overgo.streams.reply("} {
		if !strings.Contains(chat, needle) {
			t.Errorf("chat missing %q", needle)
		}
	}
	agent := get("/mod/agent.js")
	if !strings.Contains(agent, "overgo.toolStep(") || strings.Contains(agent, `"/agents/approval"`) {
		t.Error("the agent tab does not ride the shared tool-step surface")
	}
}
