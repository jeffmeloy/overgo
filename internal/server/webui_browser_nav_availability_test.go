package server

import (
	"net/http/httptest"
	"os"
	"testing"

	"overgo/internal/webuilane"
)

// TestWebUIBrowserNavAvailability drives the navigation on a plain launch:
// every refused workspace is marked off in the navigation and an open one
// is not, and opening a refused one shows what is missing and how to turn
// it on in plain words, as a status with its own heading.
func TestWebUIBrowserNavAvailability(t *testing.T) {
	t.Parallel()
	requireWebUILane(t, "the nav availability leg uses Chromium through cmd/webui-lane")
	server := httptest.NewServer(newTestHandlerWithRepository(t, &fakeGenerator{}))
	defer server.Close()
	ctx, browser, check, settle := openBrowserLeg(t, server.URL)
	defer browser.Close()
	settle(`document.querySelectorAll('button.tab').length > 10 && [...document.querySelectorAll('button.tab')].some(button => button.getAttribute('aria-disabled') === 'true')`)

	// Refused workspaces are marked off; open ones are not.
	check(`[...document.querySelectorAll('button.tab')].every(button => (getComputedStyle(button, '::after').content === '"off"') === (button.getAttribute('aria-disabled') === 'true'))`)

	// A refused workspace says what is missing and how to turn it on, in plain words.
	check(`(() => { location.hash = 'training-jobs'; return true; })()`)
	settle(`!!document.querySelector('#panel-training-jobs.active .workspace-refusal h2')`)
	check(`(() => {
  const refusal = document.querySelector('#panel-training-jobs .workspace-refusal');
  const text = refusal.textContent;
  return refusal.getAttribute('role') === 'status' && refusal.querySelector('h2').textContent === "Train isn't available on this server"
    && text.includes('Training is not turned on for this server.') && text.includes('Restart the server with training turned on.')
    && !/(^|\s)-[a-z]|go run|workflow\.|capability/.test(text);
})()`)
	for _, viewport := range webuilane.ScreenViewports {
		findings, err := webuilane.CaptureState(ctx, browser, os.Getenv("OVERGO_WEBUI_LANE_SCREENS"), viewport, "refused-workspace")
		if err != nil || len(findings) != 0 {
			t.Fatalf("%s: %v %v", viewport.Name, findings, err)
		}
	}
	// A return to Chat whose manifest is still being read loses to the navigation
	// made after it, whether that manifest changed or not: the response arrives
	// after the page has moved on and neither takes the tab back nor remounts Chat.
	// queued delivers it between the hash's change and its hashchange event, the
	// gap the first page load fell into, where only the hash shows the navigation.
	for _, changed := range []string{"false", "true"} {
		for _, queued := range []string{"false", "true"} {
			check(`(() => {
  const original = overgo.api.get;
  let deliver;
  const held = new Promise(resolve => { deliver = resolve; });
  overgo.api.get = function (path, options) {
    if (path !== '/workspace/manifest') return original.call(this, path, options);
    window.manifestRead = true;
    return held;
  };
  window.deliverManifest = async () => {
    overgo.api.get = original;
    const next = await original.call(overgo.api, '/workspace/manifest');
    if (` + queued + `) location.hash = 'library';
    deliver(` + changed + ` ? {...next, superseded: true} : next);
    await held;
    window.manifestDelivered = true;
  };
  location.hash = 'chat';
  return true;
})()`)
			settle(`window.manifestRead === true`)
			if queued == "false" {
				check(`(() => { location.hash = 'library'; return true; })()`)
				settle(`!!document.querySelector('#panel-library.active')`)
			}
			check(`(() => { deliverManifest(); return true; })()`)
			settle(`window.manifestDelivered === true`)
			settle(`!!document.querySelector('#panel-library.active')`)
			check(`location.hash === '#library' && !document.querySelector('#panel-chat.active')`)
			check(`(() => { delete window.manifestRead; delete window.manifestDelivered; return true; })()`)
		}
	}
	webuilane.Leg(t, "nav availability leg", "every refused workspace was marked off in the navigation and no open one was, and a refused workspace said what is missing and how to turn it on in plain words under its own heading")
}
