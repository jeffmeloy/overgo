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
	t.Parallel()
	if os.Getenv("OVERGO_WEBUI_LANE") != "1" {
		testskip.NotApplicable(t, "workspace refusal runs through cmd/webui-lane")
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
	// 1. Inspect lists its refused Attention view, and choosing it shows the reason and the enabling action
	// instead of the module.
	assertBrowserPredicate(t, ctx, browser, `(() => { location.hash = 'inspect'; return true; })()`)
	settle(`!!document.querySelector('#panel-inspect.active .view-chooser')`)
	assertBrowserPredicate(t, ctx, browser, `(() => {
  const chip = [...document.querySelectorAll('#panel-inspect .view-chooser button')].find((candidate) => candidate.textContent === 'Attention');
  if (!chip) return false;
  chip.click();
  return true;
})()`)
	settle(`(() => {
  const line = document.querySelector('#panel-inspect.active .workspace-refusal');
  return !!line && line.getAttribute('role') === 'status' && line.textContent.includes('unavailable') && line.textContent.includes('Serve a local model');
})()`)
	// A view's fragment reaches the same refusal, never the module.
	assertBrowserPredicate(t, ctx, browser, `(() => { location.hash = 'chat'; return true; })()`)
	settle(`!!document.querySelector('#panel-chat.active')`)
	assertBrowserPredicate(t, ctx, browser, `(() => { location.hash = 'attention'; return true; })()`)
	settle(`!!document.querySelector('#panel-inspect.active .workspace-refusal') && !document.querySelector('#panel-inspect select, #panel-inspect textarea, #panel-inspect canvas')`)
	webuilane.Leg(t, "workspace refusal leg", "the refused view answers a choice and a fragment with its reason and enabling action")

	// 2. The Inbox listens on the shell's shared runtime stream and opens none of its own; a remount
	// (a key change re-reads every mounted tab and restarts the shared stream) leaves exactly one live one.
	assertBrowserPredicate(t, ctx, browser, `(() => {
  window.liveStreams = {};
  const original = window.fetch;
  window.fetch = function (path, options) {
    if (typeof path === 'string' && path.endsWith('/stream')) (window.liveStreams[path] = window.liveStreams[path] || []).push(options && options.signal);
    return original.apply(this, arguments);
  };
  location.hash = 'inbox';
  return true;
})()`)
	settle(`!!document.querySelector('#panel-inbox.active') && !window.liveStreams['/agents/stream']`)
	assertBrowserPredicate(t, ctx, browser, `(() => {
  const key = document.getElementById('api-key');
  key.value = 'lane-remount';
  key.dispatchEvent(new Event('change', { bubbles: true }));
  return true;
})()`)
	settle(`(window.liveStreams['/runtime/activity/stream'] || []).filter((signal) => !signal.aborted).length === 1 && !window.liveStreams['/agents/stream']`)
	webuilane.Leg(t, "remount leg", "the inbox listens on the shared runtime stream, and a remount leaves exactly one live runtime stream")

	// 3. Granting the blocked operation's recovery from the Automations tab succeeds.
	assertBrowserPredicate(t, ctx, browser, `(() => { location.hash = 'automations'; return true; })()`)
	// The panel re-renders on runtime events, so the button is found and
	// clicked in one evaluation, polled until a render holds it.
	settle(`(() => {
  const grant = [...document.querySelectorAll('#panel-automations.active .card button')].find((button) => button.textContent.startsWith('Grant'));
  if (!grant) return false;
  grant.click();
  return true;
})()`)
	waitBrowserOperationState(t, fixture.handler, blocked, operation.StateCompleted)
	settle(`window.overgo.errors.length === 0 && !document.querySelector('#panel-automations .note .banner')`)
	webuilane.Leg(t, "approval leg", "the Automations decision carried the advertised request and the operation completed")
}
