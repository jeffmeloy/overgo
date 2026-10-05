package server

import (
	"net/http/httptest"
	"testing"

	"overgo/internal/webuilane"
)

// TestWebUIBrowserRequestsViewKeepsItsRowsWhilePolling holds the Requests view of the API tab to
// what a reader needs while it reads the ring again every two seconds: the table is never
// replaced by the loading placeholder after its first read; the explorer's route filter survives
// a switch to the view and back, with only the cards it matches showing; and a payload that is not
// a list shows the failure card without leaving the view's filter to throw on the next change.
func TestWebUIBrowserRequestsViewKeepsItsRowsWhilePolling(t *testing.T) {
	t.Parallel()
	requireWebUILane(t, "the requests view leg uses Chromium through cmd/webui-lane")
	handler := newTestHandler(t, &fakeGenerator{})
	defer handler.Close()
	server := httptest.NewServer(handler)
	defer server.Close()
	_, browser, check, settle := openBrowserLeg(t, server.URL+"/#requests")
	defer browser.Close()
	settle(`!!document.querySelector('#panel-api.active select[aria-label="Show"]') && window.overgo.errors.length === 0`)
	check(`(async () => { await fetch('/v1/models'); return true; })()`)
	settle(`[...document.querySelectorAll('#panel-api tr')].some((row) => row.querySelector('td') && row.textContent.includes('/v1/models'))`)

	// Two reads of the ring pass with the table in place: no placeholder is ever put back over it.
	check(`(async () => {
  const host = document.querySelector('#panel-api .view-host');
  let placeholders = 0;
  new MutationObserver((records) => { for (const record of records) for (const node of record.addedNodes) if (node.textContent === 'Loading…') placeholders++; }).observe(host, { childList: true, subtree: true });
  await new Promise((resolve) => setTimeout(resolve, 4500));
  const rows = [...host.querySelectorAll('tr')].filter((row) => row.querySelector('td')).length;
  if (placeholders !== 0 || rows === 0) throw new Error('the table was replaced ' + placeholders + ' times by the placeholder, rows ' + rows);
  return true;
})()`)

	// The route filter outlives a switch to the Requests view and back: the cards it hid stay hidden.
	check(`(async () => {
  const panel = document.querySelector('#panel-api');
  const tab = (id) => panel.querySelector('[role="tab"][data-view="' + id + '"]');
  tab('api').click();
  const filter = panel.querySelector('input[aria-label="Filter routes"]');
  filter.value = '/v1/models';
  filter.dispatchEvent(new Event('input', { bubbles: true }));
  const cards = () => [...panel.querySelectorAll('.card[data-route]')];
  const visible = () => cards().filter((card) => card.offsetParent !== null).length;
  const matching = cards().filter((card) => !card.hidden).length;
  if (matching === 0 || matching === cards().length || visible() !== matching) throw new Error('the filter shows ' + visible() + ' of ' + cards().length + ', wants ' + matching);
  tab('requests').click();
  await new Promise((resolve) => setTimeout(resolve, 50));
  if (visible() !== 0) throw new Error('a route card shows over the requests view');
  tab('api').click();
  await new Promise((resolve) => setTimeout(resolve, 50));
  if (visible() !== matching || !/of \d+ routes match/.test(panel.textContent)) throw new Error('after the switch ' + visible() + ' cards show, the filter matched ' + matching);
  tab('requests').click();
  return true;
})()`)
	settle(`!!document.querySelector('#panel-api select[aria-label="Show"]')`)

	// A payload that is not a list is refused at the read: the failure card shows and the next change of the filter draws.
	check(`(async () => {
  const original = overgo.api.get;
  overgo.api.get = async function (path, options) { return path === '/requests' ? {} : original.call(this, path, options); };
  await new Promise((resolve) => setTimeout(resolve, 2600));
  overgo.api.get = original;
  const panel = document.querySelector('#panel-api');
  if (!/Try again/.test(panel.textContent)) throw new Error('a payload that is no list showed no failure card');
  const select = panel.querySelector('select[aria-label="Show"]');
  select.value = 'failed';
  select.dispatchEvent(new Event('input', { bubbles: true }));
  select.value = 'all';
  select.dispatchEvent(new Event('input', { bubbles: true }));
  if (overgo.errors.length !== 0) throw new Error('the filter threw after a bad payload: ' + JSON.stringify(overgo.errors));
  return true;
})()`)
	webuilane.Leg(t, "requests view leg", "two reads of the ring left the table in place with no placeholder over it; the route filter kept its hidden cards through a switch to the requests view and back and no card showed over the view; a payload that was no list showed the failure card and the filter still drew")
}
