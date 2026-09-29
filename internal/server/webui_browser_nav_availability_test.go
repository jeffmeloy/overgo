package server

import (
	"net/http/httptest"
	"os"
	"testing"

	"overgo/internal/testskip"
	"overgo/internal/webuilane"
)

// TestWebUIBrowserNavAvailability drives the navigation on a plain launch:
// every refused workspace is marked off in the navigation and an open one
// is not, and opening a refused one shows what is missing and how to turn
// it on in plain words, as a status with its own heading.
func TestWebUIBrowserNavAvailability(t *testing.T) {
	t.Parallel()
	if os.Getenv("OVERGO_WEBUI_LANE") != "1" {
		t.Skip(testskip.Inapplicable + ": the nav availability leg uses Chromium through cmd/webui-lane")
	}
	server := httptest.NewServer(newTestHandlerWithRepository(t, &fakeGenerator{}))
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
	webuilane.Leg(t, "nav availability leg", "every refused workspace was marked off in the navigation and no open one was, and a refused workspace said what is missing and how to turn it on in plain words under its own heading")
}
