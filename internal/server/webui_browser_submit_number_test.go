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

// TestWebUIBrowserSubmitNumberCarriesAComposerRun holds a run started from
// the composer to the request number a Run button's submission carries. The
// server here serves speech alone, so a send is a run of /generation/run. The
// first answer is lost after the server has the request: the page says the
// outcome is unknown and blocks sending. Sent again once the person allows
// it, the same request goes under the same number, the server answers with
// the run it already started, and the model is asked once. Before the
// composer numbered its run, the second send was a second run.
func TestWebUIBrowserSubmitNumberCarriesAComposerRun(t *testing.T) {
	t.Parallel()
	requireWebUILane(t, "the composer number leg uses Chromium through cmd/webui-lane")
	shell, generator := coldStartSpeechFixture(t)
	server := httptest.NewServer(shell)
	defer server.Close()
	ctx, browser, check, settle := openBrowserLeg(t, server.URL)
	defer browser.Close()
	settle(`!!document.querySelector('.mode-controls select[aria-label="voice"]')`)
	check(`(()=>{document.querySelector('.mode-controls select[aria-label="voice"]').focus();return true;})()`)
	pressKey(t, ctx, browser, "ArrowDown", 40)
	settle(`document.querySelector('.mode-controls select[aria-label="voice"]').value === 'alba'`)

	// The first answer is lost after the server has the request; the page records what it posted.
	check(`(() => {
  window.posted = [];
  let lost = true;
  const real = window.fetch.bind(window);
  window.fetch = async (url, options) => {
    if (String(url).endsWith('/generation/run') && options && options.method === 'POST') {
      window.posted.push(JSON.parse(options.body));
      const response = await real(url, options);
      if (lost) { lost = false; throw new TypeError('the response was lost'); }
      return response;
    }
    return real(url, options);
  };
  return true;
})()`)
	const prompt = "Read this aloud once."
	say(t, ctx, browser, prompt)
	settle(`window.posted.length === 1 && /outcome is unknown/.test(document.querySelector('.composer-extras').textContent) && document.querySelector('.composer .send-button').disabled`)

	// Allowed again, the same request goes under the same number and starts no second run.
	check(`(() => { [...document.querySelectorAll('.composer-extras button')].find((button) => button.textContent === 'Allow sending again').click(); return true; })()`)
	say(t, ctx, browser, prompt)
	settle(`window.posted.length === 2 && [...document.querySelectorAll('.msg.error')].some((row) => row.textContent.includes('speech fixture stopped before synthesis'))`)
	check(`(() => {
  const [first, second] = window.posted;
  if (typeof first.request !== 'string' || first.request.length < 16) throw new Error('the composer run carried no request number: ' + JSON.stringify(first));
  if (second.request !== first.request) throw new Error('the run sent again took another number: ' + first.request + ' then ' + second.request);
  return true;
})()`)
	generator.mu.Lock()
	defer generator.mu.Unlock()
	if generator.executions != 1 {
		t.Fatalf("the model was asked %d times for one request sent twice, want once", generator.executions)
	}
	webuilane.Leg(t, "composer number leg", "a composer run whose first answer was lost went again under the same request number and was answered with the run already started, the model asked once")
}
