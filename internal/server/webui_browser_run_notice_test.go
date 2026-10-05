package server

import (
	"fmt"
	"net/http/httptest"
	"os"
	"testing"

	"overgo/internal/testskip"
	"overgo/internal/webuilane"
)

// TestWebUIBrowserRunNotice finishes two short fine-tunes, the first with the page in
// front and the second with it hidden: the first leaves the title and raises nothing,
// the second marks the title and raises one notification, and showing the page again
// clears the mark. The browser's permission is asked once, at the first Run.
func TestWebUIBrowserRunNotice(t *testing.T) {
	t.Parallel()
	if os.Getenv("OVERGO_WEBUI_LANE") != "1" {
		t.Skip(testskip.Inapplicable + ": the run notice leg uses Chromium through cmd/webui-lane")
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
  window.fineTune = (conversations, output) => {
    const select = control('conversations').querySelector('select');
    select.value = conversations;
    select.dispatchEvent(new Event('input', { bubbles: true }));
    control('steps').querySelector('input').value = '4';
    control('output').querySelector('input').value = output;
    runButton().click();
  };
  return true;
})()`
	check(`(() => { location.hash = 'training-jobs'; return true; })()`)
	settle(`!!document.querySelector('#panel-training-jobs.active select[aria-label="capability"] option')`)
	check(helpers)
	id := fixture.conversations.String()
	settle(fmt.Sprintf(`!!control('conversations') && [...control('conversations').querySelectorAll('select option')].some((option) => option.value === %q)`, id))

	// A browser that has not been asked: the stub counts requests and grants the permission when asked.
	check(`(() => {
  window.requested = 0;
  window.notices = [];
  window.baseTitle = document.title;
  window.Notification = class {
    static permission = 'default';
    static requestPermission() { window.requested++; window.Notification.permission = 'granted'; return Promise.resolve('granted'); }
    constructor(text) { window.notices.push(text); }
    addEventListener() {}
    close() {}
  };
  return true;
})()`)

	// The first run ends with the page in front: nothing is marked and nothing is raised.
	check(fmt.Sprintf(`(() => { fineTune(%q, 'noticed-visible'); return true; })()`, id))
	settle(`!!verdict() && !runButton().disabled`)
	check(`window.requested === 1 && window.notices.length === 0 && document.title === window.baseTitle`)

	// The second ends with the page hidden: the title carries a mark and one notification is raised.
	check(fmt.Sprintf(`(() => {
  Object.defineProperty(document, 'hidden', { configurable: true, get: () => true });
  fineTune(%q, 'noticed-hidden');
  return true;
})()`, id))
	settle(`!!verdict() && !runButton().disabled`)
	check(`document.title === '(1) ' + window.baseTitle && window.notices.length === 1 && /completed/.test(window.notices[0])`)
	// The permission is not asked again by the second run.
	check(`window.requested === 1`)

	// Showing the page clears the mark.
	check(`(() => {
  Object.defineProperty(document, 'hidden', { configurable: true, get: () => false });
  document.dispatchEvent(new Event('visibilitychange'));
  return true;
})()`)
	check(`document.title === window.baseTitle`)
}
