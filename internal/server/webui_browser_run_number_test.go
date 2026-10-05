package server

import (
	"fmt"
	"net/http/httptest"
	"os"
	"testing"

	"overgo/internal/testskip"
	"overgo/internal/webuilane"
)

// TestWebUIBrowserRunNumber drives a fine-tune whose first submission answers but
// whose response is lost on the way back: the page shows the failure, and the next
// Run of the same submission is sent under the same request number, so the server
// answers it with the operation the first started and the training runs once.
func TestWebUIBrowserRunNumber(t *testing.T) {
	t.Parallel()
	if os.Getenv("OVERGO_WEBUI_LANE") != "1" {
		t.Skip(testskip.Inapplicable + ": the run number leg uses Chromium through cmd/webui-lane")
	}
	fixture := newSFTFixture(t)
	handler := newTestHandlerForRepository(t, fixture.store, &fakeGenerator{}, func(config *Config) { config.Workflows = fixture.workspace })
	server := httptest.NewServer(handler)
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
	const helpers = `(() => {
  window.trainingPanel = () => document.querySelector('#panel-training-jobs');
  window.control = (label) => [...trainingPanel().querySelectorAll('label.control')].find((item) => item.querySelector('span').textContent === label);
  window.runButton = () => [...trainingPanel().querySelectorAll('button')].find((button) => button.textContent === 'Run');
  window.verdict = () => trainingPanel().querySelector('[aria-label="Held-out verdict"]');
  return true;
})()`
	check(`(() => { location.hash = 'training-jobs'; return true; })()`)
	settle(`!!document.querySelector('#panel-training-jobs.active select[aria-label="capability"] option')`)
	check(helpers)
	id := fixture.conversations.String()
	settle(fmt.Sprintf(`!!control('conversations') && [...control('conversations').querySelectorAll('select option')].some((option) => option.value === %q)`, id))
	check(fmt.Sprintf(`(() => {
  const select = control('conversations').querySelector('select');
  select.value = %q;
  select.dispatchEvent(new Event('input', { bubbles: true }));
  control('steps').querySelector('input').value = '4';
  control('output').querySelector('input').value = 'numbered';
  return select.value === %q;
})()`, id, id))

	// The first run's response is lost after the server has it; the page records what it posted.
	check(`(() => {
  window.posted = [];
  let lost = true;
  const real = window.fetch.bind(window);
  window.fetch = async (url, options) => {
    if (String(url).endsWith('/training/run') && options && options.method === 'POST') {
      window.posted.push(JSON.parse(options.body));
      const response = await real(url, options);
      if (lost) { lost = false; throw new TypeError('the response was lost'); }
      return response;
    }
    return real(url, options);
  };
  runButton().click();
  return true;
})()`)
	settle(`window.posted.length === 1 && !runButton().disabled && !verdict()`)
	check(`typeof window.posted[0].request === 'string' && window.posted[0].request.length >= 16`)

	// The same submission again goes under the same number and is answered with the operation already started.
	check(`(() => { runButton().click(); return true; })()`)
	settle(`!!verdict()`)
	check(`window.posted.length === 2 && window.posted[1].request === window.posted[0].request`)
	// One run, though the submission was sent twice.
	check(`(async () => { const operations = await (await fetch('/operations')).json(); return JSON.stringify(operations).split('"task":"training"').length - 1 === 1; })()`)
}
