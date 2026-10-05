package server

import (
	"net/http/httptest"
	"testing"

	"overgo/internal/overgodb"
	"overgo/internal/webuilane"
)

// TestWebUIBrowserSubmitKeepsItsNumberThroughAGatewayAnswer holds the page's numbered
// submission to what a retry needs: a try answered by a timeout or a gateway (408, 502,
// 503, 504), which may have reached a server that started the run, goes again under the
// same number, while an answer that settles it (a refusal, 400) or a success leaves the
// next try a number of its own.
func TestWebUIBrowserSubmitKeepsItsNumberThroughAGatewayAnswer(t *testing.T) {
	t.Parallel()
	requireWebUILane(t, "the submit number leg uses Chromium through cmd/webui-lane")
	repository, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = repository.Close() })
	handler := newTestHandlerForRepository(t, repository, responseRecipeGenerator(t, &fakeGenerator{}))
	defer handler.Close()
	server := httptest.NewServer(handler)
	defer server.Close()
	_, browser, check, settle := openBrowserLeg(t, server.URL+"/")
	defer browser.Close()
	settle(`!!window.overgo && !!window.overgo.api && typeof window.overgo.api.submit === "function"`)
	check(`(async () => {
  const sent = [];
  const original = window.fetch;
  const answers = (statuses) => { let at = 0; window.fetch = async (path, init) => { sent.push(JSON.parse(init.body).request); const status = statuses[at++]; return new Response(JSON.stringify(status < 400 ? { operation: 'op' } : { error: { message: 'no' } }), { status, headers: { 'Content-Type': 'application/json' } }); }; };
  const attempt = async (path, body) => { try { await overgo.api.submit(path, body); } catch (err) { /* the answer is the point */ } };
  try {
    for (const status of [408, 502, 503, 504]) {
      sent.length = 0;
      answers([status, 202]);
      const path = '/number-' + status;
      await attempt(path, { a: 1 });
      await attempt(path, { a: 1 });
      if (sent.length !== 2 || !sent[0] || sent[0] !== sent[1]) throw new Error('a ' + status + ' answer did not keep the number: ' + JSON.stringify(sent));
    }
    sent.length = 0;
    answers([400, 202]);
    await attempt('/number-refused', { a: 1 });
    await attempt('/number-refused', { a: 1 });
    if (sent.length !== 2 || sent[0] === sent[1]) throw new Error('a refusal kept the number: ' + JSON.stringify(sent));
    sent.length = 0;
    answers([202, 202]);
    await attempt('/number-done', { a: 1 });
    await attempt('/number-done', { a: 1 });
    if (sent.length !== 2 || sent[0] === sent[1]) throw new Error('a success kept the number: ' + JSON.stringify(sent));
  } finally { window.fetch = original; }
  return true;
})()`)
	webuilane.Leg(t, "submit number leg", "a submission answered by a 408, 502, 503 or 504 went again under the same number, and one answered by a 400 or a success took a new number")
}
