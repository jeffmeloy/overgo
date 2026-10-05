package server

import (
	"net/http/httptest"
	"strconv"
	"testing"
	"time"

	"overgo/internal/overgodb"
	"overgo/internal/webuilane"
)

// firstTokenDelay is how long the fixture holds back every token: the first token the page shows cannot
// have come sooner than this after the turn was sent.
const firstTokenDelay = 250 * time.Millisecond

// TestWebUIBrowserShowsTheFirstTokenTime holds the chat's context meter to the one measure that spans the
// proxy, admission, prefill and paint: after a turn the meter shows the milliseconds from sending the turn
// to its first token, and the fixture's own delay before a token is a lower bound of it.
func TestWebUIBrowserShowsTheFirstTokenTime(t *testing.T) {
	t.Parallel()
	requireWebUILane(t, "the first token leg uses Chromium through cmd/webui-lane")
	repository, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = repository.Close() })
	handler := newTestHandlerForRepository(t, repository, responseRecipeGenerator(t, &fakeGenerator{pieces: []string{"Hel", "lo"}, tokenDelay: firstTokenDelay}))
	defer handler.Close()
	server := httptest.NewServer(handler)
	defer server.Close()
	_, browser, check, settle := openBrowserLeg(t, server.URL+"/")
	defer browser.Close()
	settle(`!!document.querySelector('.composer textarea')`)
	check(`(() => {
  const box = document.querySelector('.composer textarea');
  box.value = 'hi';
  box.dispatchEvent(new Event('input', { bubbles: true }));
  return true;
})()`)
	settle(`!document.querySelector('.send-button').disabled`)
	check(`(() => { document.querySelector('.send-button').click(); return true; })()`)
	settle(`[...document.querySelectorAll('[aria-label="Context meter"] .stat .k')].some((label) => label.textContent === 'First token')`)
	check(`(() => {
  const stat = [...document.querySelectorAll('[aria-label="Context meter"] .stat')].find((node) => node.querySelector('.k').textContent === 'First token');
  const milliseconds = Number(stat.querySelector('.v').firstChild.textContent);
  if (!(milliseconds >= ` + strconv.Itoa(int(firstTokenDelay.Milliseconds())) + `) || milliseconds > 60000) throw new Error('the first token time is ' + milliseconds + ' ms, want at least the fixture delay');
  return true;
})()`)
	webuilane.Leg(t, "first token leg", "a turn sent through the composer showed its first token time in the context meter, at least the fixture's delay before a token")
}
