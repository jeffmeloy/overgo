package server

import (
	"net/http/httptest"
	"os"
	"testing"
	"time"

	"overgo/internal/webuilane"
)

// requestMonitorTokenDelay makes a generated token take long enough for the monitor to measure a rate.
const requestMonitorTokenDelay = 5 * time.Millisecond

// TestWebUIBrowserRequestMonitor sends two requests from the page, one answered and one
// refused, and reads the Requests view of the API tab: both are listed newest first with their route and status,
// the header names the totals, the error rate and the tokens per second, the Failed filter leaves
// the refused one alone, and the tab's own polling is not among the requests it lists.
func TestWebUIBrowserRequestMonitor(t *testing.T) {
	t.Parallel()
	requireWebUILane(t, "the request monitor leg uses Chromium through cmd/webui-lane")
	handler := newTestHandler(t, &fakeGenerator{pieces: []string{"A", "B", "C"}, tokenDelay: requestMonitorTokenDelay})
	defer handler.Close()
	server := httptest.NewServer(handler)
	defer server.Close()
	path, err := webuilane.FindBrowser(os.Getenv("OVERGO_BROWSER"))
	if err != nil {
		t.Fatal(err)
	}
	ctx := t.Context()
	// The page opens at the view's own address, which a reload or a bookmark does: it lands on the view, not on the first tab.
	browser, err := webuilane.Open(ctx, path, server.URL+"/#requests")
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
  window.requestsPanel = () => document.querySelector('#panel-api');
  window.requestRows = () => [...requestsPanel().querySelectorAll('tr')].filter((row) => row.querySelector('td')).map((row) => [...row.querySelectorAll('td')].map((cell) => cell.textContent));
  window.showing = (value) => { const select = requestsPanel().querySelector('select[aria-label="Show"]'); select.value = value; select.dispatchEvent(new Event('input', { bubbles: true })); return select.value === value; };
  return true;
})()`
	settle(`!!document.querySelector('#panel-api.active select[aria-label="Show"]')`)
	check(helpers)
	// The requests are a view of the API tab: its chooser selects them, and the navigation holds no entry of their own.
	check(`requestsPanel().querySelector('[role="tab"][data-view="requests"]').getAttribute('aria-selected') === 'true' && [...document.querySelectorAll('#sections button')].every((button) => button.textContent !== 'Requests')`)

	// The page sends two requests, one answered and one refused; the open view reads the ring again by itself and lists
	// them newest first, each with its route and status, the view's own reads not among them.
	check(`(async () => {
  const post = (body) => fetch('/completion', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body });
  const answered = await post(JSON.stringify({ prompt: 'hi', n_predict: 3 }));
  const refused = await post('{');
  return answered.status === 200 && refused.status === 400;
})()`)
	settle(`requestRows().some((cells) => cells.includes('/completion') && cells.includes('400'))`)
	check(`(() => {
  const rows = requestRows();
  const refused = rows.findIndex((cells) => cells.includes('/completion') && cells.includes('400'));
  const answered = rows.findIndex((cells) => cells.includes('/completion') && cells.includes('200'));
  return refused >= 0 && answered > refused && rows.every((cells) => !cells.includes('/requests')) && rows[answered].some((text) => /in · 3 out/.test(text));
})()`)

	// The header reads the ring: how many, the error rate, the slowest call and the tokens per second.
	check(`(() => {
  const text = requestsPanel().querySelector('[aria-label="Request totals"]').textContent;
  return /\d+ requests since the server started/.test(text) && /0 in flight/.test(text) && /\d+% failed of the last \d+/.test(text) && /slowest [A-Z]+ \S+ /.test(text) && /\d+\.\d tokens\/s/.test(text);
})()`)

	// The Failed filter leaves the refused request alone; In flight finds none on a quiet server.
	check(`showing('failed')`)
	check(`(() => { const rows = requestRows(); return rows.length > 0 && rows.every((cells) => Number(cells[3]) >= 400) && rows.some((cells) => cells.includes('/completion')); })()`)
	check(`showing('in-flight')`)
	check(`requestsPanel().textContent.includes('No requests match.')`)
}
