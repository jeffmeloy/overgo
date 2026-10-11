package server

import (
	"net/http"
	"net/http/httptest"
	"strconv"
	"testing"
	"time"

	"overgo/internal/overgodb"
	"overgo/internal/webuilane"
)

// firstTokenDelay is how long the fixture holds back every token, and every request a turn makes before
// its stream opens: the first token the page shows cannot have come sooner than these after the turn was
// sent.
const firstTokenDelay = 250 * time.Millisecond

// firstTokenWaits is how many of those delays lie between a person's action and the first token, for a
// turn sent and for one continued: the token count or the stored turn, the stream's headers, and the token.
const firstTokenWaits = 3

// heldBack is a server whose answers to what a turn asks before its stream opens, the count of its tokens,
// the stored turn and the stream itself, begin only after delay, or not at all for a request given up.
type heldBack struct {
	http.Handler
	delay time.Duration
}

func (held heldBack) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	switch r.URL.Path {
	case "/v1/responses/input_tokens", "/interactions/messages", "/v1/responses":
		select {
		case <-time.After(held.delay):
		case <-r.Context().Done():
			return
		}
	}
	held.Handler.ServeHTTP(w, r)
}

// TestWebUIBrowserShowsTheFirstTokenTime holds the chat's context meter to one measure for both ways a
// turn begins: after a turn the meter shows the milliseconds from the person's action, the send or the
// Continue, to the turn's first token. The fixture delays the three things that lie between them, the
// count of the turn's tokens or the fetch of the stored turn, the stream's headers, and the token itself,
// and all three are a lower bound of the time shown, for the turn sent and for the turn continued.
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
	server := httptest.NewServer(heldBack{Handler: handler, delay: firstTokenDelay})
	defer server.Close()
	_, browser, check, settle := openBrowserLeg(t, server.URL+"/")
	defer browser.Close()
	const (
		budget = `document.querySelector('#conversation-settings input[aria-label="max tokens"]')`
		// unread is the meter's first token time that no step has read yet: the meter is drawn again when a turn ends.
		unread = `[...document.querySelectorAll('[aria-label="Context meter"] .stat')].find((node) => node.querySelector('.k').textContent === 'First token' && !node.dataset.read)`
		more   = `[...document.querySelectorAll('.chat-log .note button')].find((button) => button.textContent === 'Continue')`
		failed = `document.querySelector('.chat-log [role="alert"]')`
	)
	settle(`!!document.querySelector('.composer textarea') && !!` + budget)
	// A reply of one token is cut at its limit, which is what offers Continue.
	check(`(() => {
  const limit = ` + budget + `; limit.value = '1'; limit.dispatchEvent(new Event('input', { bubbles: true }));
  const box = document.querySelector('.composer textarea');
  box.value = 'hi';
  box.dispatchEvent(new Event('input', { bubbles: true }));
  return true;
})()`)
	settle(`!document.querySelector('.send-button').disabled`)
	check(`(() => { document.querySelector('.send-button').click(); return true; })()`)
	settle(`(!!` + unread + ` && !!` + more + `) || !!` + failed)
	check(`(() => {
  if (` + failed + `) throw new Error('the turn sent failed: ' + ` + failed + `.textContent);
  const stat = ` + unread + `;
  window.firstTokenSent = Number(stat.querySelector('.v').firstChild.textContent);
  stat.dataset.read = '1';
  ` + more + `.click();
  return true;
})()`)
	settle(`!!` + unread + ` || !!` + failed)
	bound := strconv.Itoa(int((firstTokenWaits * firstTokenDelay).Milliseconds()))
	check(`(() => {
  if (` + failed + `) throw new Error('the turn continued failed: ' + ` + failed + `.textContent);
  const sent = window.firstTokenSent, continued = Number(` + unread + `.querySelector('.v').firstChild.textContent);
  for (const milliseconds of [sent, continued]) {
    if (!(milliseconds >= ` + bound + `) || milliseconds > 60000) throw new Error('the first token times are ' + sent + ' ms for the turn sent and ' + continued + ' ms for the turn continued, want each at least the ` + bound + ` ms the fixture holds back between the action and the token');
  }
  return true;
})()`)
	webuilane.Leg(t, "first token leg", "a turn sent through the composer and the same turn continued each showed its first token time in the context meter, timed from the person's action: at least the fixture's delays of the count or the stored turn, the stream's headers and the token")
}
