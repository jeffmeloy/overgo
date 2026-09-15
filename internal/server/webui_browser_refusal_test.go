package server

import (
	"net/http/httptest"
	"os"
	"testing"

	"overgo/internal/operation"
	"overgo/internal/testskip"
	"overgo/internal/webuilane"
)

// TestWebUIBrowserWorkspaceRefusal: a refused workspace stays in the
// navigation and answers a click or a fragment with its reason and the
// action that would enable it; a streaming tab remounts onto one live
// stream; and a decision on a pending approval from the Automations tab
// carries the advertised request and succeeds.
func TestWebUIBrowserWorkspaceRefusal(t *testing.T) {
	if os.Getenv("OVERGO_WEBUI_LANE") != "1" {
		t.Skip(testskip.ShortIntegration + ": workspace refusal runs through cmd/webui-lane")
	}
	fixture := newAutomationServerFixture(t)
	defer fixture.store.Close()
	_, blocked := publishBrowserLaneOperations(t, fixture.handler)
	server := httptest.NewServer(fixture.handler)
	defer server.Close()
	path, err := webuilane.FindBrowser(os.Getenv("OVERGO_BROWSER"))
	if err != nil {
		t.Fatal(err)
	}
	ctx := t.Context()
	browser, err := webuilane.Open(ctx, path, server.URL+"/app.html#chat")
	if err != nil {
		t.Fatal(err)
	}
	defer browser.Close()
	settle := func(expression string) {
		t.Helper()
		if err := browser.Eventually(ctx, expression); err != nil {
			t.Fatalf("%s: %v", expression, err)
		}
	}
	settle(`!!document.querySelector('#panel-chat.active') && window.overgo.errors.length === 0`)
	// 1. The refused Attention tab stays listed, and a click shows its reason and the enabling action instead of the module.
	assertBrowserPredicate(t, ctx, browser, `(() => {
  const button = [...document.querySelectorAll('button.tab')].find((candidate) => candidate.textContent === 'Attention');
  if (!button || button.hidden || button.getAttribute('aria-disabled') !== 'true') return false;
  button.click();
  return true;
})()`)
	settle(`(() => {
  const line = document.querySelector('#panel-attention.active .workspace-refusal');
  return !!line && line.getAttribute('role') === 'status' && line.textContent.includes('unavailable') && line.textContent.includes('Serve a local model');
})()`)
	// The fragment reaches the same refusal, never the module.
	assertBrowserPredicate(t, ctx, browser, `(() => { location.hash = 'chat'; return true; })()`)
	settle(`!!document.querySelector('#panel-chat.active')`)
	assertBrowserPredicate(t, ctx, browser, `(() => { location.hash = 'attention'; return true; })()`)
	settle(`!!document.querySelector('#panel-attention.active .workspace-refusal') && !document.querySelector('#panel-attention select, #panel-attention textarea, #panel-attention canvas')`)
	t.Log("workspace refusal leg: the refused tab answers a click and a fragment with its reason and enabling action")

	// 2. The Inbox streams; a remount (the key change re-reads every mounted tab) leaves exactly one live stream.
	assertBrowserPredicate(t, ctx, browser, `(() => {
  window.liveStreams = [];
  const original = window.fetch;
  window.fetch = function (path, options) {
    if (typeof path === 'string' && path === '/agents/stream') window.liveStreams.push(options && options.signal);
    return original.apply(this, arguments);
  };
  location.hash = 'inbox';
  return true;
})()`)
	settle(`!!document.querySelector('#panel-inbox.active') && window.liveStreams.length === 1`)
	assertBrowserPredicate(t, ctx, browser, `(() => {
  const key = document.getElementById('api-key');
  key.value = '';
  key.dispatchEvent(new Event('change', { bubbles: true }));
  return true;
})()`)
	settle(`window.liveStreams.length === 2 && window.liveStreams.filter((signal) => !signal.aborted).length === 1`)
	t.Log("remount leg: the streaming tab remounted onto one live stream")

	// 3. Granting the blocked operation's recovery from the Automations tab succeeds.
	assertBrowserPredicate(t, ctx, browser, `(() => { location.hash = 'automations'; return true; })()`)
	settle(`[...document.querySelectorAll('#panel-automations.active .card button')].some((button) => button.textContent.startsWith('Grant'))`)
	assertBrowserPredicate(t, ctx, browser, `(() => {
  const grant = [...document.querySelectorAll('#panel-automations .card button')].find((button) => button.textContent.startsWith('Grant'));
  if (!grant) return false;
  grant.click();
  return true;
})()`)
	waitBrowserOperationState(t, fixture.handler, blocked, operation.StateCompleted)
	settle(`window.overgo.errors.length === 0 && !document.querySelector('#panel-automations .note .banner')`)
	t.Log("approval leg: the Automations decision carried the advertised request and the operation completed")
}
