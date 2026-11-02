package server

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"overgo/internal/artifact"
)

// TestWebUIBrowserAgentSwitchMidTurn keeps each agent's history under that agent. A person who switches agents before a
// reply lands used to save the first agent's transcript as the second's, which went out with the second's next turn, and
// lost Stop while the turn ran. After a reload a picked agent answers with the turns the store holds of it.
func TestWebUIBrowserAgentSwitchMidTurn(t *testing.T) {
	t.Parallel()
	requireWebUILane(t, "the agent switch leg uses Chromium through cmd/webui-lane")
	fixture := newAgentWorkspaceFixture(t, nil, nil, nil)
	defer fixture.store.Close()
	for _, name := range []string{"first-agent", "second-agent"} {
		definition := agentDefinitionInput{
			Name: name, Prompt: fixture.prompt, ModelRecipe: fixture.generator.description.Identity.Recipe,
			ToolManuals: []artifact.ID{fixture.manual}, Policies: []artifact.ID{fixture.policy},
		}
		published := serveTestRequest(fixture.handler, http.MethodPost, "/agents/definitions", marshalTestJSON(t, definition))
		var created struct {
			ID artifact.ID `json:"id"`
		}
		if err := json.Unmarshal(published.Body.Bytes(), &created); published.Code != http.StatusCreated || err != nil {
			t.Fatalf("agent publish status=%d body=%s", published.Code, published.Body.String())
		}
		activateAgentFromAPI(t, fixture.handler, created.ID)
	}
	server := httptest.NewServer(fixture.handler)
	defer server.Close()
	_, browser, check, settle := openBrowserLeg(t, server.URL)
	defer browser.Close()
	settle(`!!document.querySelector('.composer textarea')`)

	// Every chat body is kept, one can be held so a switch lands mid-reply, and a read of a stored thread is counted.
	const watchChat = `(() => {
  const real = window.fetch;
  window.chatBodies = [];
  window.fetch = (url, init) => {
    if (String(url).includes('/agents/thread')) return real(url, init).then((response) => { window.threadReads = (window.threadReads || 0) + 1; return response; });
    if (!String(url).endsWith('/agents/chat')) return real(url, init);
    window.chatBodies.push(JSON.parse(init.body));
    if (!window.holdChat) return real(url, init);
    return new Promise((done) => { window.releaseChat = () => done(real(url, init)); });
  };
  return true;
})()`
	check(watchChat)
	check(`(() => { location.hash = 'agent'; return true; })()`)
	settle(`!!document.querySelector('#panel-agent.active details.fold')`)
	check(`(() => {
  window.agentPanel = () => document.querySelector('#panel-agent');
  window.agentFold = (title) => [...agentPanel().querySelectorAll('details.fold')].find(fold => fold.querySelector('summary').textContent === title);
  window.openAgent = (name) => [...agentFold('Your agents').querySelectorAll('tr')].find(row => row.textContent.includes(name)).querySelector('button').click();
  window.sayToAgent = (text) => {
    const input = agentFold('Chat').querySelector('.composer textarea');
    input.value = text; input.dispatchEvent(new Event('input', { bubbles: true }));
    [...agentFold('Chat').querySelectorAll('button')].find(button => button.textContent === 'Send').click();
  };
  window.stopShown = () => { const stop = agentFold('Chat').querySelector('.stop-button'); return !!stop && !stop.hidden; };
  return true;
})()`)
	settle(`[...agentFold('Your agents').querySelectorAll('tr')].some(row => row.textContent.includes('second-agent'))`)

	// The first agent's turn is in flight when the second is opened: Stop stays, and the reply lands under the first.
	check(`(() => { openAgent('first-agent'); window.holdChat = true; sayToAgent('for the first'); return true; })()`)
	settle(`chatBodies.length === 1 && chatBodies[0].agent === 'first-agent' && !!window.releaseChat`)
	check(`(() => { openAgent('second-agent'); return true; })()`)
	settle(`stopShown()`)
	check(`(() => { window.holdChat = false; releaseChat(); return true; })()`)
	settle(`!stopShown() && agentFold('Chat').querySelectorAll('.msg.assistant').length > 0`)

	// The second agent's first turn carries nothing of the first's, and the first's next carries its own.
	check(`(() => { sayToAgent('for the second'); return true; })()`)
	settle(`chatBodies.length === 2 && chatBodies[1].agent === 'second-agent' && chatBodies[1].messages.length === 1`)
	settle(`!stopShown()`)
	check(`(() => { openAgent('first-agent'); sayToAgent('again for the first'); return true; })()`)
	settle(`chatBodies.length === 3 && chatBodies[2].agent === 'first-agent' && chatBodies[2].messages.length === 3`)
	settle(`!stopShown()`)

	// The front thread holds the same, and after a reload a picked agent answers with the turns the store holds of it.
	const frontThread = `(() => {
  location.hash = 'chat';
  window.pickAgent = (name) => { const picker = document.querySelector('.agent-session select[aria-label="agent"]'); picker.value = name; picker.dispatchEvent(new Event('change', { bubbles: true })); };
  window.sayInFront = (text) => {
    const input = document.querySelector('#panel-chat .composer textarea');
    input.value = text; input.dispatchEvent(new Event('input', { bubbles: true }));
    document.querySelector('#panel-chat .send-button').click();
  };
  return true;
})()`
	const agentMode = `(() => {
  const mode = document.querySelector('#panel-chat .composer select[aria-label="mode"]');
  mode.value = 'agent'; mode.dispatchEvent(new Event('change', { bubbles: true }));
  return true;
})()`
	const modeOffered = `!!document.querySelector('#panel-chat .composer select[aria-label="mode"] option[value="agent"]')`
	check(frontThread)
	settle(modeOffered)
	check(agentMode)
	settle(`!document.querySelector('.agent-session').hidden`)
	check(`(() => { pickAgent('first-agent'); window.holdChat = true; window.releaseChat = null; sayInFront('front first'); return true; })()`)
	settle(`chatBodies.length === 4 && chatBodies[3].agent === 'first-agent' && !!window.releaseChat`)
	check(`(() => { pickAgent('second-agent'); window.holdChat = false; releaseChat(); return true; })()`)
	settle(`!document.querySelector('#panel-chat .send-button').hidden`)
	check(`(() => { sayInFront('front second'); return true; })()`)
	settle(`chatBodies.length === 5 && chatBodies[4].agent === 'second-agent' && chatBodies[4].messages.length === 1`)
	settle(`!document.querySelector('#panel-chat .send-button').hidden`)
	check(`(() => { window.reloading = true; location.reload(); return true; })()`)
	settle(`!window.reloading && ` + modeOffered)
	check(watchChat)
	check(frontThread)
	check(agentMode)
	settle(`!document.querySelector('.agent-session').hidden`)
	check(`(() => { window.threadReads = 0; pickAgent('second-agent'); return true; })()`)
	settle(`window.threadReads === 1`)
	check(`(() => { sayInFront('front second again'); return true; })()`)
	settle(`chatBodies.length === 1 && chatBodies[0].agent === 'second-agent' && chatBodies[0].messages.length === 3`)
}
