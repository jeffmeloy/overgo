package server

import (
	"net/http/httptest"
	"os"
	"testing"

	"overgo/internal/overgodb"
	"overgo/internal/testskip"
	"overgo/internal/webuilane"
)

// TestWebUIBrowserAgentRecovery drives an agent session through a reload and
// a streaming tool: after a reload the front thread shows the session's turn
// and tool step again with its step count, and a running tool's output
// appears in its card piece by piece before the step completes.
func TestWebUIBrowserAgentRecovery(t *testing.T) {
	// Serial: the fixture sets PATH.
	if os.Getenv("OVERGO_WEBUI_LANE") != "1" {
		t.Skip(testskip.Inapplicable + ": the agent recovery leg uses Chromium through cmd/webui-lane")
	}
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	fixture, release := streamingAgentFixture(t, store)
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
	const helpers = `(() => {
  window.recoverySurface = () => document.querySelector('.agent-session');
  window.recoveryButton = (label) => [...recoverySurface().querySelectorAll('button')].find(button => button.textContent === label);
  window.recoveryTool = (name) => { const select = recoverySurface().querySelector('select[aria-label="tool"]'); select.value = name; select.dispatchEvent(new Event('change', { bubbles: true })); };
  window.recoveryCards = (name) => [...document.querySelectorAll('.card.tool-call')].filter(card => card.querySelector('.tool-header .mono').textContent === name);
  window.recoveryState = (card) => card.querySelector('.tool-header .tag').textContent;
  window.recoveryGuard = () => recoverySurface().querySelector('[aria-label="Guardrails"]').textContent;
  window.recoveryAgentMode = () => { const mode = document.querySelector('.composer select[aria-label="mode"]'); if (mode.value !== 'agent') { mode.value = 'agent'; mode.dispatchEvent(new Event('change', { bubbles: true })); } return true; };
  return true;
})()`
	open := func() {
		t.Helper()
		settle(`!!document.querySelector('.composer textarea') && [...document.querySelectorAll('.composer select[aria-label="mode"] option')].some(option => option.value === 'agent')`)
		check(helpers)
		check(`recoveryAgentMode()`)
		settle(`!recoverySurface().hidden && [...recoverySurface().querySelectorAll('select[aria-label="tool"] option')].some(option => option.value === 'tool.stream')`)
	}
	open()

	// A turn and a tool step in the session.
	check(`(() => {
  const input = document.querySelector('.composer textarea');
  input.value = 'hello agent'; input.dispatchEvent(new Event('input', { bubbles: true }));
  document.querySelector('.send-button').click();
  return true;
})()`)
	settle(`[...document.querySelectorAll('.msg.assistant .body')].some(body => body.textContent.trim() !== '') && !document.querySelector('.send-button').disabled`)
	check(`(() => { recoveryTool('store.head'); recoveryButton('Execute inspection').click(); return true; })()`)
	settle(`recoveryCards('store.head').length === 1 && recoveryState(recoveryCards('store.head')[0]).startsWith('done') && recoveryGuard().startsWith('steps 1 / ')`)

	// Reloaded, the session's turn and step return with its count.
	if err := browser.Evaluate(ctx, `location.reload()`, nil); err != nil {
		t.Fatal(err)
	}
	open()
	settle(`[...document.querySelectorAll('.msg.user')].some(message => message.textContent.includes('hello agent')) && [...document.querySelectorAll('.msg.assistant .body')].some(body => body.textContent.trim() !== '') &&
  recoveryCards('store.head').length === 1 && recoveryState(recoveryCards('store.head')[0]) === 'done' && recoveryGuard().startsWith('steps 1 / ')`)

	// A running tool's output arrives in its card before the step completes.
	settle(`recoverySurface().dataset.live === 'true'`)
	check(`(() => { recoveryTool('tool.stream'); recoveryButton('Execute inspection').click(); return true; })()`)
	settle(`recoveryCards('tool.stream').length === 1 && recoveryState(recoveryCards('tool.stream')[0]) === 'running' && (recoveryCards('tool.stream')[0].querySelector('.tool-output') || {}).textContent === 'first chunk\n'`)
	release()
	settle(`recoveryState(recoveryCards('tool.stream')[0]).startsWith('done') && recoveryCards('tool.stream')[0].textContent.includes('second chunk') && !recoveryCards('tool.stream')[0].querySelector('.tool-output') && recoveryGuard().startsWith('steps 2 / ')`)
	webuilane.Leg(t, "agent recovery leg", "after a reload the front thread showed the session's turn and tool step again with its step count, and a running tool's output appeared in its card before the step completed")
}
