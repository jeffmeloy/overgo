package server

import (
	"context"
	"encoding/json"
	"net/http/httptest"
	"testing"

	"overgo/internal/overgodbtest"
	"overgo/internal/testcheck"
)

// TestWebUIBrowserToolOutputKeepsAPiecesTail holds a tool card to its tail when one piece of output is longer than the
// tail, as the JSON stream transport admits: the card keeps the tail's end of the piece from a whole code point, says
// that the start is not shown, keeps the next piece after it, and the tool's end removes both.
func TestWebUIBrowserToolOutputKeepsAPiecesTail(t *testing.T) {
	t.Parallel()
	requireWebUILane(t, "the tool output leg uses Chromium through cmd/webui-lane")
	store := overgodbtest.Open(t, t.TempDir())
	server := httptest.NewServer(newTestHandlerForRepository(t, store, responseRecipeGenerator(t, &fakeGenerator{})))
	defer server.Close()
	ctx, browser, check, settle := openBrowserLeg(t, server.URL)
	defer browser.Close()
	settle(`!!window.overgo && !!window.overgo.thread`)
	check(`(() => {
  const host = document.createElement('div');
  document.body.append(host);
  const card = overgo.thread(host).toolCard({ name: 'tool', arguments: {} });
  const noted = () => [...host.querySelectorAll('.note')].some((note) => note.textContent === "Showing the end of this tool's output.");
  const oversized = 'a\u{1F600}' + 'z'.repeat((1 << 20) - 1);
  card.output(oversized);
  card.output('!');
  const exact = host.querySelector('.tool-output').textContent === oversized.slice(3) + '!';
  const shown = noted();
  card.end('done');
  window.measured = { exact, shown, cleared: !host.querySelector('.tool-output') && !noted() };
  return true;
})()`)
	var text string
	testcheck.NoError(t, browser.Evaluate(context.WithoutCancel(ctx), `JSON.stringify(window.measured)`, &text))
	var measured struct{ Exact, Shown, Cleared bool }
	testcheck.NoError(t, json.Unmarshal([]byte(text), &measured))
	if !measured.Exact || !measured.Shown || !measured.Cleared {
		t.Fatalf("one piece longer than the tail then one more: its tail and the next kept whole %t, the notice shown %t, both gone at the end %t", measured.Exact, measured.Shown, measured.Cleared)
	}
}
