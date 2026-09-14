package server

import (
	"context"
	"errors"
	"net/http/httptest"
	"os"
	"testing"
	"time"

	"overgo/internal/overgodb"
	"overgo/internal/testskip"
	"overgo/internal/webuilane"
)

// TestWebUIBrowserSamplingSettings changes the temperature in Settings,
// sends a turn, and reads the resolved chain the turn's inspector shows.
func TestWebUIBrowserSamplingSettings(t *testing.T) {
	if os.Getenv("OVERGO_WEBUI_LANE") != "1" {
		t.Skip(testskip.ShortIntegration + ": sampling settings run through cmd/webui-lane")
	}
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	handler := newTestHandlerForRepository(t, store, responseRecipeGenerator(t, &fakeGenerator{pieces: []string{"Sampled answer"}}))
	defer handler.Close()
	server := httptest.NewServer(handler)
	defer server.Close()
	path, err := webuilane.FindBrowser(os.Getenv("OVERGO_BROWSER"))
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithTimeoutCause(t.Context(), 60*time.Second, errors.New("sampling settings did not settle"))
	defer cancel()
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
	settle(`!!document.querySelector('#panel-chat.active .composer textarea') && !document.querySelector('.send-button').disabled`)
	assertBrowserPredicate(t, ctx, browser, `(() => {
  const field = document.querySelector('[aria-label="temperature"]');
  field.value = '0.25';
  field.dispatchEvent(new Event('input', { bubbles: true }));
  field.dispatchEvent(new Event('change', { bubbles: true }));
  return field.value === '0.25';
})()`)
	say(t, ctx, browser, "Reply with one word.")
	settle(`document.querySelectorAll('#panel-chat .msg.assistant').length === 1 && document.querySelector('#panel-chat .msg.assistant .body').textContent.trim().length > 0 && !document.querySelector('.send-button').disabled`)
	assertBrowserPredicate(t, ctx, browser, `(() => { const button = document.querySelector('#panel-chat .msg.assistant .role button.link-button'); if (!button) return false; button.click(); return true; })()`)
	settle(`(() => { const text = document.querySelector('#inspector').textContent; return !document.querySelector('#inspector').hidden && text.includes('Sampling') && text.includes('0.25') && text.includes('temperature'); })()`)
	t.Log("sampling settings leg: the temperature set in Settings reaches the resolved chain the turn's inspector shows")
}
