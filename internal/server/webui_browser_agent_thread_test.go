package server

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"testing"

	"overgo/internal/agenttool"
	"overgo/internal/artifact"
	"overgo/internal/testskip"
	"overgo/internal/webuilane"
)

// TestWebUIBrowserAgentThread drives agent mode in the front thread: the
// active agent answers a turn in the conversation, an inspection runs as a
// tool card with the session's remaining steps, and a mutation is refused
// unapproved, previewed with the facts a grant binds, then approved against
// the previewed operation and run as a tool card.
func TestWebUIBrowserAgentThread(t *testing.T) {
	t.Parallel()
	if os.Getenv("OVERGO_WEBUI_LANE") != "1" {
		t.Skip(testskip.Inapplicable + ": the agent thread leg uses Chromium through cmd/webui-lane")
	}
	fixture := newAgentWorkspaceFixture(t, nil, nil, nil)
	defer fixture.store.Close()
	// A mutation manual whose only role is the approval gate; its argv
	// transport runs only under the committed policy naming its program.
	manuals, err := agenttool.StandardManuals()
	if err != nil {
		t.Fatal(err)
	}
	write, err := agenttool.NewManual(agenttool.Manual{
		Name: "store.commit", Description: "A mutation manual for the approval gate.",
		Effect:    agenttool.EffectMutation,
		Ceiling:   agenttool.EffectCeiling{Targets: []agenttool.EffectTargetBinding{{Scope: agenttool.EffectScopeRepository, Value: "overgodb"}}},
		Transport: agenttool.Transport{Kind: agenttool.TransportArgv, Program: "git", Args: []string{"status"}},
	})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := agenttool.PublishArgvPolicy(t.Context(), fixture.store, []string{"git"}); err != nil {
		t.Fatal(err)
	}
	if _, err := agenttool.PublishManualCatalog(t.Context(), fixture.store, append(manuals, write)); err != nil {
		t.Fatal(err)
	}
	definition := AgentDefinitionInput{
		Name: "research-agent", Prompt: fixture.prompt, ModelRecipe: fixture.generator.description.Identity.Recipe,
		ToolManuals: []artifact.ID{fixture.manual, write.ID}, Policies: []artifact.ID{fixture.policy},
	}
	published := serveTestRequest(fixture.handler, http.MethodPost, "/agents/definitions", marshalAutomationJSON(t, definition))
	var created struct {
		ID artifact.ID `json:"id"`
	}
	if err := json.Unmarshal(published.Body.Bytes(), &created); published.Code != http.StatusCreated || err != nil || !created.ID.Valid() {
		t.Fatalf("agent publish status=%d body=%s", published.Code, published.Body.String())
	}
	activateDefinitionFromAPI(t, fixture.handler, created.ID, "/agents/activate")
	server := httptest.NewServer(fixture.handler)
	defer server.Close()
	path, err := webuilane.FindBrowser(os.Getenv("OVERGO_BROWSER"))
	if err != nil {
		t.Fatal(err)
	}
	ctx := t.Context()
	browser, err := webuilane.Open(ctx, path, server.URL)
	if err != nil {
		t.Fatal(err)
	}
	defer browser.Close()
	check := func(expression string) { t.Helper(); assertBrowserPredicate(t, ctx, browser, expression) }
	settle := func(expression string) {
		t.Helper()
		if err := browser.Eventually(ctx, expression); err != nil {
			t.Fatalf("%s: %v", expression, err)
		}
	}
	settle(`!!document.querySelector('.composer textarea') && [...document.querySelectorAll('.composer select[aria-label="mode"] option')].some(option => option.value === 'agent')`)
	check(`(() => {
  window.agentSurface = () => document.querySelector('.agent-session');
  window.agentButton = (label) => [...agentSurface().querySelectorAll('button')].find(button => button.textContent === label);
  window.agentTool = (name) => { const select = agentSurface().querySelector('select[aria-label="tool"]'); select.value = name; select.dispatchEvent(new Event('change', { bubbles: true })); };
  window.agentCards = (name) => [...document.querySelectorAll('.card.tool-call')].filter(card => card.querySelector('.tool-header .mono').textContent === name);
  window.agentCardState = (card) => card.querySelector('.tool-header .tag').textContent;
  window.agentGuard = () => agentSurface().querySelector('[aria-label="Guardrails"]').textContent;
  const mode = document.querySelector('.composer select[aria-label="mode"]'); mode.value = 'agent'; mode.dispatchEvent(new Event('change', { bubbles: true }));
  return true;
})()`)

	// The active agent answers a turn in the front thread.
	settle(`!agentSurface().hidden && agentSurface().querySelector('select[aria-label="agent"]').value === 'research-agent'`)
	check(`(() => {
  const input = document.querySelector('.composer textarea');
  input.value = 'hello agent'; input.dispatchEvent(new Event('input', { bubbles: true }));
  document.querySelector('.send-button').click();
  return true;
})()`)
	settle(`[...document.querySelectorAll('.msg.user')].some(message => message.textContent.includes('hello agent')) && [...document.querySelectorAll('.msg.assistant .body')].some(body => body.textContent.trim() !== '') && !document.querySelector('.send-button').disabled`)

	// An inspection runs as a tool card with the session's remaining steps.
	check(`(() => { agentTool('store.head'); agentButton('Execute inspection').click(); return true; })()`)
	settle(`agentCards('store.head').length === 1 && agentCardState(agentCards('store.head')[0]).startsWith('done') && agentGuard().startsWith('steps 1 / ') && agentGuard().endsWith(' remaining')`)

	// A mutation is refused unapproved and previewed with what a grant binds.
	check(`(() => { agentTool('store.commit'); agentButton('Execute inspection').click(); return true; })()`)
	settle(`agentCards('store.commit').length === 1 && agentCardState(agentCards('store.commit')[0]).startsWith('error')`)
	settle(`(() => { const card = agentSurface().querySelector('.card.approval'); return !!card && card.querySelector('.tag').textContent === 'mutation' &&
  card.textContent.includes('No committed decision yet') && card.textContent.includes('grant binds:') && !agentButton('Approve and execute').disabled && agentButton('Execute inspection').disabled; })()`)

	// Approved against the previewed operation, it runs.
	check(`(() => { agentButton('Approve and execute').click(); return true; })()`)
	settle(`agentCards('store.commit').length === 2 && agentCardState(agentCards('store.commit')[1]).startsWith('done') && agentGuard().startsWith('steps 2 / ') && !agentSurface().querySelector('.card.approval')`)
	webuilane.Leg(t, "agent thread leg", "the active agent answered in the front thread, an inspection ran as a tool card with the remaining steps, and a mutation was refused unapproved, previewed with its binding, then approved against its operation and ran")
}
