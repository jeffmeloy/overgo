package server

import (
	"net/http/httptest"
	"os"
	"testing"

	"overgo/internal/testskip"
	"overgo/internal/webuilane"
)

// TestWebUIBrowserAgentTab drives the agent tab as a person would: the page
// leads with the agents and the chat and folds every operator panel, a new
// agent is created from a name, instructions and a tool and becomes active,
// a message to it gets a reply in the thread, a manual inspection step runs
// in the session and shows in the durable observables and the session list,
// and a retrieval search without a projection is refused where it was asked.
func TestWebUIBrowserAgentTab(t *testing.T) {
	t.Parallel()
	if os.Getenv("OVERGO_WEBUI_LANE") != "1" {
		t.Skip(testskip.Inapplicable + ": the agent tab leg uses Chromium through cmd/webui-lane")
	}
	fixture := newAgentWorkspaceFixture(t, nil, nil, nil)
	defer fixture.store.Close()
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
	settle(`!!document.querySelector('.composer textarea')`)
	check(`(() => { location.hash = 'agent'; return true; })()`)
	settle(`!!document.querySelector('#panel-agent.active details.fold')`)
	check(`(() => {
  window.agentPanel = () => document.querySelector('#panel-agent');
  window.agentFold = (title) => [...agentPanel().querySelectorAll('details.fold')].find(fold => fold.querySelector('summary').textContent === title);
  window.agentButton = (label) => [...agentPanel().querySelectorAll('button')].find(button => button.textContent === label);
  window.agentStatus = () => agentPanel().querySelector('.note').textContent;
  return true;
})()`)
	// The task leads: the agents and the chat open, every operator panel folded.
	check(`agentFold('Your agents').open && agentFold('Chat').open && !agentFold('New agent').open &&
  [...agentPanel().querySelectorAll('details.fold')].filter(fold => fold.querySelector('summary').textContent.startsWith('Advanced:')).every(fold => !fold.open)`)

	// A new agent from a name, instructions and a tool becomes active.
	settle(`[...agentFold('New agent').querySelectorAll('label')].some(label => label.textContent.includes('store.head'))`)
	check(`(() => {
  agentFold('New agent').open = true;
  const name = agentPanel().querySelector('[aria-label="Agent name"]'); name.value = 'lane-agent';
  agentPanel().querySelector('[aria-label="Agent instructions"]').value = 'Answer briefly.';
  [...agentFold('New agent').querySelectorAll('label')].find(label => label.textContent.includes('store.head')).querySelector('input').click();
  agentButton('Create agent').click();
  return true;
})()`)
	settle(`agentStatus() === 'created and activated / lane-agent'`)
	settle(`[...agentFold('Your agents').querySelectorAll('tr')].some(row => row.textContent.includes('lane-agent') && row.textContent.includes('active'))`)

	// A message gets a reply in the thread.
	check(`(() => {
  const input = agentFold('Chat').querySelector('.composer textarea');
  input.value = 'hello agent'; input.dispatchEvent(new Event('input', { bubbles: true }));
  [...agentFold('Chat').querySelectorAll('button')].find(button => button.textContent === 'Send').click();
  return true;
})()`)
	settle(`agentFold('Chat').textContent.includes('hello agent') && agentFold('Chat').querySelectorAll('.msg.assistant, [data-role=assistant]').length > 0`)

	// A manual inspection runs in the session and shows in the observables and the sessions.
	check(`(() => { agentFold('Advanced: manual tool steps and approvals').open = true; return true; })()`)
	settle(`[...agentPanel().querySelectorAll('select[aria-label="tool"] option')].some(option => option.value === 'store.head')`)
	check(`(() => { agentButton('Execute inspection').click(); return true; })()`)
	settle(`agentStatus().startsWith('tool step 1 / ')`)
	settle(`agentFold('Advanced: structured observables').textContent.includes('step 1')`)
	check(`(() => { agentButton('Sessions').click(); return true; })()`)
	settle(`agentFold('Advanced: manual tool steps and approvals').textContent.includes('steps 1')`)

	// A retrieval without a projection is refused where it was asked.
	check(`(() => { window.agentBefore = agentStatus(); agentFold('Advanced: retrieval evidence').open = true; agentButton('Search').click(); return true; })()`)
	settle(`agentStatus() !== agentBefore && agentPanel().querySelector('.note').children.length > 0`)
	webuilane.Leg(t, "agent tab leg", "the agent tab led with its agents and chat, created an active agent, answered a message, ran a manual inspection into the observables and sessions, and refused a retrieval without a projection")
}
